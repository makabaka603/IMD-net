import os
import argparse
import yaml
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import IMDNet
from data import PairedImageDataset
from utils import IMDNetLoss, psnr, set_seed


def build_model(cfg):
    return IMDNet(**cfg['model'])


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    vals = []
    for batch in loader:
        inp = batch['input'].to(device)
        tgt = batch['target'].to(device)
        pred = model(inp)
        vals.append(psnr(pred, tgt))
    model.train()
    return sum(vals) / max(len(vals), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/imdnet_base.yaml')
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get('seed', 123))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = cfg.get('save_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    # Gradient accumulation: effective batch = batch_size * accumulation_steps
    batch_size = cfg['train']['batch_size']
    acc_steps = cfg['train'].get('accumulation_steps', 1)
    effective_bs = batch_size * acc_steps
    print(f'Effective batch size: {batch_size} x {acc_steps} = {effective_bs}')

    model = build_model(cfg).to(device)
    criterion = IMDNetLoss(**cfg['loss']).to(device)
    optim = Adam(model.parameters(), lr=cfg['train']['lr'] * acc_steps, betas=(0.9, 0.999))
    sched = CosineAnnealingLR(optim, T_max=cfg['train']['iterations'], eta_min=cfg['train']['min_lr'])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['train'].get('amp', True) and device == 'cuda')

    start_iter = 0
    best_psnr = -1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        optim.load_state_dict(ckpt['optim'])
        sched.load_state_dict(ckpt['sched'])
        start_iter = ckpt.get('iter', 0)
        best_psnr = ckpt.get('best_psnr', -1)

    train_set = PairedImageDataset(cfg['train_input_dir'], cfg['train_target_dir'],
                                    cfg['train']['patch_size'], augment=True)
    val_set = PairedImageDataset(cfg['val_input_dir'], cfg['val_target_dir'],
                                  patch_size=None, augment=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=cfg['train']['num_workers'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

    it = start_iter
    pbar = tqdm(total=cfg['train']['iterations'], initial=start_iter)
    optim.zero_grad(set_to_none=True)

    while it < cfg['train']['iterations']:
        for batch in train_loader:
            it += 1
            inp = batch['input'].to(device, non_blocking=True)
            tgt = batch['target'].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=cfg['train'].get('amp', True) and device == 'cuda'):
                outputs = model(inp, return_aux=True)
                loss, logs = criterion(outputs, tgt)
                loss = loss / acc_steps  # normalise for accumulation

            scaler.scale(loss).backward()

            # Update only after accumulating enough gradients
            if it % acc_steps == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                sched.step()
                optim.zero_grad(set_to_none=True)

            pbar.update(1)
            pbar.set_description(f"loss={loss.item() * acc_steps:.4f} c={logs['charb']:.4f} d={logs['decouple']:.4f}")

            if it % cfg['train']['val_every'] == 0:
                val_psnr = validate(model, val_loader, device)
                print(f'\nIter {it}: val PSNR={val_psnr:.3f} dB')
                is_best = val_psnr > best_psnr
                if is_best:
                    best_psnr = val_psnr
                ckpt = {
                    'iter': it,
                    'model': model.state_dict(),
                    'optim': optim.state_dict(),
                    'sched': sched.state_dict(),
                    'best_psnr': best_psnr,
                    'config': cfg,
                }
                torch.save(ckpt, os.path.join(save_dir, 'imdnet_latest.pth'))
                if is_best:
                    torch.save(ckpt, os.path.join(save_dir, 'imdnet_best.pth'))

            if it % cfg['train']['save_every'] == 0:
                torch.save({'model': model.state_dict(), 'config': cfg},
                           os.path.join(save_dir, f'imdnet_iter_{it}.pth'))

            if it >= cfg['train']['iterations']:
                break

    # Final step if accumulation didn't align
    if it % acc_steps != 0:
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim)
        scaler.update()
        sched.step()

    pbar.close()


if __name__ == '__main__':
    main()
