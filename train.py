# -*- coding: utf-8 -*-

import os
import re


def _valid_thread_value(v):
    if v is None:
        return False
    v = str(v).strip()
    return re.fullmatch(r"[1-9][0-9]*", v) is not None


def _fix_thread_env():
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        if not _valid_thread_value(os.environ.get(key)):
            os.environ[key] = "4"


_fix_thread_env()


import argparse
import yaml
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

try:
    from torchvision.utils import make_grid
except Exception:
    make_grid = None

from models import IMDNet
from data import PairedImageDataset
from utils import IMDNetLoss, psnr, set_seed


def build_model(cfg):
    return IMDNet(**cfg["model"])


def set_torch_threads():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def grads_are_finite(model):
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def float(x):
    try:
        if isinstance(x, torch.Tensor):
            if torch.isfinite(x).all():
                return float(x.detach().cpu())
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def log_images(writer, inp, pred, tgt, step):
    if writer is None:
        return

    inp = inp.detach().clamp(0, 1).cpu()
    pred = pred.detach().clamp(0, 1).cpu()
    tgt = tgt.detach().clamp(0, 1).cpu()

    n = min(inp.shape[0], 4)

    if make_grid is not None:
        grid = make_grid(
            torch.cat([inp[:n], pred[:n], tgt[:n]], dim=0),
            nrow=n,
            padding=2,
        )
        writer.add_image("val/input_output_gt", grid, step)
    else:
        writer.add_image("val/input", inp[0], step)
        writer.add_image("val/output", pred[0], step)
        writer.add_image("val/gt", tgt[0], step)


@torch.no_grad()
def validate(model, loader, device, writer=None, step=0):
    model.eval()

    values = []
    logged = False

    for batch in loader:
        inp = batch["input"].to(device, non_blocking=True)
        tgt = batch["target"].to(device, non_blocking=True)

        pred = model(inp)

        if isinstance(pred, dict):
            pred = pred["out"]

        if not torch.isfinite(pred).all():
            continue

        pred_for_metric = pred.clamp(0, 1)
        tgt_for_metric = tgt.clamp(0, 1)

        values.append(psnr(pred_for_metric, tgt_for_metric))

        if not logged:
            log_images(writer, inp, pred_for_metric, tgt_for_metric, step)
            logged = True

    model.train()

    if len(values) == 0:
        return float("nan")

    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="imdnet_base.yaml")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--resume_full", action="store_true")
    args = parser.parse_args()

    set_torch_threads()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 123))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = cfg.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    tb_dir = "/root/tf-logs/imdnet"
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    batch_size = int(cfg["train"]["batch_size"])
    acc_steps = int(cfg["train"].get("accumulation_steps", 1))
    total_iters = int(cfg["train"]["iterations"])
    val_every = int(cfg["train"].get("val_every", 2000))
    save_every = int(cfg["train"].get("save_every", 2000))
    num_workers = int(cfg["train"].get("num_workers", 4))
    clip_grad = float(cfg["train"].get("clip_grad", 1.0))

    print(f"Device: {device}")
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
    print(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS')}")
    print(f"TensorBoard log dir: {tb_dir}")
    print(f"Effective batch size: {batch_size} x {acc_steps} = {batch_size * acc_steps}")

    model = build_model(cfg).to(device)

    criterion = IMDNetLoss(**cfg["loss"]).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        betas=(0.9, 0.999),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_iters // max(acc_steps, 1)),
        eta_min=float(cfg["train"].get("min_lr", 1e-7)),
    )

    use_amp = bool(cfg["train"].get("amp", False)) and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_iter = 0
    best_psnr = -1.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)

        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
            start_iter = int(ckpt.get("iter", 0))
            best_psnr = float(ckpt.get("best_psnr", -1.0))

            if args.resume_full:
                if "optim" in ckpt:
                    optimizer.load_state_dict(ckpt["optim"])
                if "sched" in ckpt:
                    scheduler.load_state_dict(ckpt["sched"])
        else:
            model.load_state_dict(ckpt, strict=False)

        print(f"[INFO] resumed from iter {start_iter}")

    for p in model.parameters():
        if not torch.isfinite(p).all():
            raise RuntimeError("Model has NaN/Inf parameters before training.")

    train_set = PairedImageDataset(
        cfg["train_input_dir"],
        cfg["train_target_dir"],
        patch_size=cfg["train"]["patch_size"],
        augment=True,
    )

    val_set = PairedImageDataset(
        cfg["val_input_dir"],
        cfg["val_target_dir"],
        patch_size=None,
        augment=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    it = start_iter

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(total=total_iters, initial=start_iter)

    while it < total_iters:
        for batch in train_loader:
            it += 1

            inp = batch["input"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inp, return_aux=True)

                loss, logs = criterion(outputs, tgt)
                loss_log = loss.detach()
                loss = loss / acc_steps

            scaler.scale(loss).backward()

            if it % acc_steps == 0:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    clip_grad,
                )

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            current_lr = optimizer.param_groups[0]["lr"]

            writer.add_scalar("train/loss_total", float(loss_log.detach().cpu()), it)
            writer.add_scalar("train/loss_charbonnier", logs.get("charb", float("nan")), it)
            writer.add_scalar("train/loss_edge", logs.get("edge", float("nan")), it)
            writer.add_scalar("train/loss_fft", logs.get("fft", float("nan")), it)
            writer.add_scalar("train/loss_decouple", logs.get("decouple", float("nan")), it)
            writer.add_scalar("train/lr", current_lr, it)

            pbar.update(1)
            pbar.set_description(
                f"loss={float(loss_log.detach().cpu()):.4f} "
                f"c={logs.get('charb', float('nan')):.4f} "
                f"d={logs.get('decouple', float('nan')):.4f}"
            )

            if it % val_every == 0:
                val_psnr = validate(model, val_loader, device, writer=writer, step=it)

                writer.add_scalar("val/psnr", val_psnr, it)

                print(f"\nIter {it}: val PSNR={val_psnr:.3f} dB")

                if val_psnr == val_psnr:
                    is_best = val_psnr > best_psnr

                    if is_best:
                        best_psnr = val_psnr

                    ckpt = {
                        "iter": it,
                        "model": model.state_dict(),
                        "optim": optimizer.state_dict(),
                        "sched": scheduler.state_dict(),
                        "best_psnr": best_psnr,
                        "config": cfg,
                    }

                    torch.save(ckpt, os.path.join(save_dir, "imdnet_latest.pth"))

                    if is_best:
                        torch.save(ckpt, os.path.join(save_dir, "imdnet_best.pth"))
                        print(f"[INFO] best PSNR updated: {best_psnr:.3f} dB")

            if it % save_every == 0:
                torch.save(
                    {
                        "iter": it,
                        "model": model.state_dict(),
                        "best_psnr": best_psnr,
                        "config": cfg,
                    },
                    os.path.join(save_dir, f"imdnet_iter_{it}.pth"),
                )

            if it >= total_iters:
                break

    pbar.close()
    writer.close()


if __name__ == "__main__":
    main()
