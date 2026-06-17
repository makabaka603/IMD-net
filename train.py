# -*- coding: utf-8 -*-
import builtins

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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def is_main_process():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def setup_distributed():
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) <= 1:
        return 0, 1, False
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    print(f"[DDP] rank={local_rank}, world_size={world_size}")
    return local_rank, world_size, True


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
                return builtins.float(x.detach().cpu())
            return builtins.float("nan")
        return builtins.float(x)
    except Exception:
        return builtins.float("nan")


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
    parser.add_argument("--config", type=str, default="configs/imdnet_base.yaml")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--resume_full", action="store_true")
    args = parser.parse_args()

    set_torch_threads()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 123))

    local_rank, world_size, is_dist = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    save_dir = cfg.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    tb_dir = "/root/tf-logs/imdnet"
    if is_main_process():
        os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir) if is_main_process() else None

    batch_size_total = int(cfg["train"]["batch_size"])
    batch_size = max(1, batch_size_total // max(world_size, 1))
    acc_steps = int(cfg["train"].get("accumulation_steps", 1))
    total_iters = int(cfg["train"]["iterations"])
    val_every = int(cfg["train"].get("val_every", 2000))
    save_every = int(cfg["train"].get("save_every", 2000))
    num_workers = int(cfg["train"].get("num_workers", 4))
    clip_grad = float(cfg["train"].get("clip_grad", 1.0))

    if is_main_process():
        print(f"Device: {device}, world_size: {world_size}, per-GPU batch: {batch_size}, acc: {acc_steps}, effective: {batch_size * world_size * acc_steps}")
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
    print(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS')}")
    print(f"TensorBoard log dir: {tb_dir}")
    print(f"Effective batch size: {batch_size} x {acc_steps} = {batch_size * acc_steps}")

    model = build_model(cfg).to(device)
    if is_dist:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
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
            state_dict = ckpt["model"]
            if all(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            load_target = model.module if is_dist else model
            load_target.load_state_dict(state_dict, strict=False)
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

    if is_dist:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=dist.get_rank(), shuffle=True)
    else:
        train_sampler = None
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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

    pbar = tqdm(total=total_iters, initial=start_iter, disable=not is_main_process())

    while it < total_iters:
        if is_dist and train_sampler is not None:
            train_sampler.set_epoch(it // max(len(train_loader), 1))
        for batch in train_loader:
            it += 1

            inp = batch["input"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inp, return_aux=True)

                out = outputs["out"] if isinstance(outputs, dict) else outputs

                if not torch.isfinite(out).all():
                    print(f"\n[NaN DIAG] iter={it}: NaN in main output! min={out.min():.2e} max={out.max():.2e}", flush=True)
                elif out.abs().max() > 1e4:
                    print(f"\n[VAL DIAG] iter={it}: output too large! min={out.min():.2e} max={out.max():.2e}", flush=True)

                if isinstance(outputs, dict):
                    for j, di_val in enumerate(outputs.get("di_list", [])):
                        if di_val.abs().max() > 1e2:
                            print(f"[VAL DIAG] iter={it}: di[{j}] too large! max={di_val.abs().max():.2e}", flush=True)

                loss, logs = criterion(outputs, tgt)

                if not torch.isfinite(loss):
                    print(f"[NaN DIAG] iter={it}: NaN in loss! loss={loss.item()}", flush=True)
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

            if writer is not None:
                writer.add_scalar("train/loss_total", float(loss_log.detach().cpu()), it)
                writer.add_scalar("train/loss_charbonnier", logs.get("charb", float("nan")), it)
                writer.add_scalar("train/loss_edge", logs.get("edge", float("nan")), it)
                writer.add_scalar("train/loss_fft", logs.get("fft", float("nan")), it)
                writer.add_scalar("train/loss_decouple", logs.get("decouple", float("nan")), it)
                writer.add_scalar("train/lr", current_lr, it)

            if not pbar.disable:
                pbar.update(1)
            if not pbar.disable:
                pbar.set_description(
                f"loss={float(loss_log.detach().cpu()):.4f} "
                f"c={logs.get('charb', float('nan')):.4f} "
                f"d={logs.get('decouple', float('nan')):.4f}"
            )

            if it % val_every == 0 and is_main_process():
                val_psnr = validate(model, val_loader, device, writer=writer, step=it)

                if writer is not None:
                    writer.add_scalar("val/psnr", val_psnr, it)

                if is_main_process():
                    print(f"\nIter {it}: val PSNR={val_psnr:.3f} dB")

                if val_psnr == val_psnr:
                    is_best = val_psnr > best_psnr

                    if is_best:
                        best_psnr = val_psnr

                    ckpt = {
                        "iter": it,
                        "model": model.module.state_dict() if is_dist else model.state_dict(),
                        "optim": optimizer.state_dict(),
                        "sched": scheduler.state_dict(),
                        "best_psnr": best_psnr,
                        "config": cfg,
                    }

                    if is_main_process():
                        torch.save(ckpt, os.path.join(save_dir, "imdnet_latest.pth"))

                    if is_best and is_main_process():
                        torch.save(ckpt, os.path.join(save_dir, "imdnet_best.pth"))
                        print(f"[INFO] best PSNR updated: {best_psnr:.3f} dB")

            if it % save_every == 0 and is_main_process():
                torch.save(
                    {
                        "iter": it,
                        "model": model.module.state_dict() if is_dist else model.state_dict(),
                        "best_psnr": best_psnr,
                        "config": cfg,
                    },
                    os.path.join(save_dir, f"imdnet_iter_{it}.pth"),
                )

            if it >= total_iters:
                break

    if not pbar.disable:
        pbar.close()
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
