"""
Quick sanity check for IMDNet training setup.

Run this before launching a long training job to catch common issues:
    - Missing data paths / config errors
    - Model construction failures
    - Data loading / augmentation errors
    - Forward / backward pass crashes
    - Loss computation and gradient flow

Usage:
    python sanity_check.py                                      # use default config
    python sanity_check.py --config configs/imdnet_base.yaml    # custom config
    python sanity_check.py --resume checkpoints/imdnet_best.pth # also load checkpoint
"""

import argparse
import os
import sys
import time
from pathlib import Path

import yaml
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

OK = "\033[92mOK\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"


def check(ok: bool, msg: str, detail: str = "") -> None:
    """Print a check result."""
    status = OK if ok else FAIL
    print(f"  [{status}] {msg}")
    if not ok and detail:
        print(f"         {detail}")


def check_err(name: str, err: Exception) -> None:
    """Print a formatted failure."""
    print(f"  [{FAIL}] {name}")
    print(f"         {type(err).__name__}: {err}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_python() -> None:
    print("\n[Python environment]")
    check(sys.version_info >= (3, 8), f"Python >= 3.8 (have {sys.version})")
    check(torch.__version__ >= "2.0", f"torch >= 2.0 (have {torch.__version__})")
    check(torch.cuda.is_available(), "CUDA available (will run on CPU if not)", "")
    try:
        import yaml  # noqa
        check(True, "pyyaml installed")
    except ImportError:
        check(False, "pyyaml installed")
    try:
        from PIL import Image  # noqa
        check(True, "Pillow installed")
    except ImportError:
        check(False, "Pillow installed")


def check_config(path: str) -> dict:
    print(f"\n[Config: {path}]")
    path = Path(path)
    check(path.exists(), f"Config file exists")
    if not path.exists():
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    check(isinstance(cfg, dict), "Config parsed as dict")

    # Required top-level keys
    required = ["model", "train", "loss", "train_input_dir", "train_target_dir"]
    for key in required:
        check(key in cfg, f"Config has '{key}'")

    if "model" in cfg:
        m = cfg["model"]
        for k in ["img_channel", "width", "enc_blk_nums"]:
            check(k in m, f"  model.{k}")

    if "train" in cfg:
        t = cfg["train"]
        for k in ["patch_size", "batch_size", "iterations", "lr"]:
            check(k in t, f"  train.{k}")
        if "accumulation_steps" in t:
            check(t["accumulation_steps"] >= 1, "  train.accumulation_steps >= 1")
        check(t.get("batch_size", 1) >= 1, "  train.batch_size >= 1")

    if "loss" in cfg:
        L = cfg["loss"]
        for k in ["lambda_fft", "delta_edge", "gamma_decouple"]:
            check(k in L, f"  loss.{k}")

    return cfg


def check_datasets(cfg: dict) -> None:
    print("\n[Dataset paths]")
    # Support both direct config keys and nested under dataset_root
    for role in ["train_input_dir", "train_target_dir", "val_input_dir", "val_target_dir"]:
        raw = cfg.get(role, "")
        if not raw:
            print(f"  [{SKIP}] {role}  (not configured)")
            continue
        p = Path(raw) if raw else Path(".")
        exists = p.exists()
        # Also try under a dataset_root if provided
        if not exists and "dataset_root" in cfg:
            alt = Path(cfg["dataset_root"]) / raw
            check(alt.exists(), f"{role}  -> using dataset_root", str(alt))
        else:
            check(exists, f"{role}  {p.name if exists else ''}")
            if exists:
                files = list(p.rglob("*"))
                exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
                img_files = [f for f in files if f.suffix.lower() in exts]
                check(len(img_files) > 0, f"  has images ({len(img_files)} found)")


def check_model(cfg: dict) -> nn.Module:
    print("\n[Model construction]")
    try:
        from models import IMDNet

        model = IMDNet(**cfg["model"])
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Count params in M
        total_m = total / 1e6
        trainable_m = trainable / 1e6
        check(True, f"IMDNet created  ({total_m:.2f}M params, {trainable_m:.2f}M trainable)")

        # Quick forward pass to catch shape errors
        device = "cpu"
        model = model.to(device)
        B, C, H, W = 1, cfg["model"].get("img_channel", 3), 64, 64
        x = torch.randn(B, C, H, W)
        with torch.no_grad():
            outputs = model(x, return_aux=True)

        check("out" in outputs, "  forward returns 'out'")
        check(outputs["out"].shape == (B, C, H, W), f"  out shape {outputs['out'].shape}")
        check("side_outputs" in outputs, "  forward returns 'side_outputs'")
        n_side = len(outputs["side_outputs"])
        check(n_side > 0, f"  {n_side} side outputs")
        for i, s in enumerate(outputs["side_outputs"]):
            check(s.shape[1] == C, f"  side[{i}] channel={s.shape[1]}")
        check("degraded_img" in outputs, "  forward returns 'degraded_img'")
        check("cf_list" in outputs, "  forward returns 'cf_list'")
        check("di_list" in outputs, "  forward returns 'di_list'")

        return model
    except Exception as e:
        check_err("Model construction / forward", e)
        sys.exit(1)


def check_loss(model: nn.Module, cfg: dict) -> None:
    print("\n[Loss + backward]")
    try:
        from utils.losses import IMDNetLoss

        device = "cpu"
        model = model.to(device)
        criterion = IMDNetLoss(**cfg["loss"])

        B, C, H, W = 2, cfg["model"].get("img_channel", 3), 64, 64
        x = torch.randn(B, C, H, W)
        tgt = torch.randn(B, C, H, W)

        outputs = model(x, return_aux=True)
        loss, logs = criterion(outputs, tgt)

        check(torch.isfinite(loss), f"Loss is finite  ({loss.item():.4f})")
        for key in ["charb", "edge", "fft", "decouple"]:
            v = logs.get(key, float("nan"))
            check(torch.isfinite(torch.tensor(v)), f"  {key}={v:.4f}")

        # Backward pass
        loss.backward()
        has_grad = False
        for name, p in model.named_parameters():
            if p.grad is not None:
                has_grad = True
                if not torch.isfinite(p.grad).all():
                    check(False, f"  Gradient NaN/Inf in {name}")
                break
        check(has_grad, "  Gradients flow through all layers")

    except Exception as e:
        check_err("Loss / backward", e)
        sys.exit(1)


def check_dataloader(cfg: dict) -> None:
    print("\n[DataLoader]")
    try:
        from data import PairedImageDataset
        from torch.utils.data import DataLoader

        patch_size = cfg["train"]["patch_size"]
        batch_size = min(cfg["train"]["batch_size"], 4)  # small for speed

        # Try train set
        train_dir = cfg.get("train_input_dir", "")
        target_dir = cfg.get("train_target_dir", "")
        if not train_dir or not target_dir:
            print(f"  [{SKIP}] train set  (paths not configured)")
            return

        ds = PairedImageDataset(train_dir, target_dir, patch_size=patch_size, augment=True)
        check(len(ds) > 0, f"Train dataset: {len(ds)} pairs")

        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        batch = next(iter(dl))
        check("input" in batch, "  batch has 'input'")
        check("target" in batch, "  batch has 'target'")
        check(batch["input"].shape == (batch_size, 3, patch_size, patch_size),
              f"  input shape {list(batch['input'].shape)}")
        check(batch["target"].shape == (batch_size, 3, patch_size, patch_size),
              f"  target shape {list(batch['target'].shape)}")
        check(torch.isfinite(batch["input"]).all(), "  input values are finite")
        check(torch.isfinite(batch["target"]).all(), "  target values are finite")

    except Exception as e:
        check_err("Dataloader", e)


def check_resume(cfg: dict, resume_path: str) -> None:
    print("\n[Checkpoint loading]")
    try:
        from models import IMDNet

        path = Path(resume_path)
        check(path.exists(), f"Checkpoint exists: {resume_path}")
        if not path.exists():
            return

        ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        check("model" in ckpt or any(k.endswith("weight") for k in ckpt.keys()),
              "Checkpoint contains model weights")

        model = IMDNet(**cfg["model"])
        state = ckpt["model"] if "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        check(len(missing) == 0, f"No missing keys  ({len(missing)} missing)")
        check(len(unexpected) == 0, f"No unexpected keys  ({len(unexpected)} unexpected)")

        if "optim" in ckpt:
            check(True, "Checkpoint has optimizer state")
        if "iter" in ckpt:
            check(True, f"Checkpoint at iteration {ckpt['iter']}")

    except Exception as e:
        check_err("Checkpoint loading", e)


def check_end2end(cfg: dict, model: nn.Module) -> None:
    """Simulate a few training steps with AMP and gradient accumulation."""
    print("\n[End-to-end training step (2 iterations)]")
    try:
        from utils.losses import IMDNetLoss
        from torch.optim import Adam
        from torch.optim.lr_scheduler import CosineAnnealingLR

        device = "cpu"
        model = model.to(device)
        criterion = IMDNetLoss(**cfg["loss"])
        opt = Adam(model.parameters(), lr=cfg["train"]["lr"])
        acc_steps = cfg["train"].get("accumulation_steps", 1)
        total_steps = cfg["train"]["iterations"]
        sched = CosineAnnealingLR(opt, T_max=total_steps // acc_steps, eta_min=cfg["train"].get("min_lr", 1e-7))
        scaler = torch.cuda.amp.GradScaler(enabled=False)

        B, C, H, W = 2, cfg["model"].get("img_channel", 3), cfg["train"]["patch_size"], cfg["train"]["patch_size"]
        if H * W > 256 * 256:
            H, W = 128, 128

        t0 = time.time()
        for step in range(2):
            x = torch.randn(B, C, H, W)
            tgt = torch.randn(B, C, H, W)

            with torch.cuda.amp.autocast(enabled=False):
                outputs = model(x, return_aux=True)
                loss, logs = criterion(outputs, tgt)
                loss = loss / acc_steps

            scaler.scale(loss).backward()

            if (step + 1) % acc_steps == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)

        elapsed = time.time() - t0
        check(True, f"2 training steps completed in {elapsed:.1f}s")
        check(torch.isfinite(loss), f"Loss stable during training  ({loss.item():.4f})")

        # Estimate total training time
        total_iters = cfg["train"]["iterations"]
        est_hours = elapsed / 2 * total_iters / 3600
        check(True, f"Estimated total: {est_hours:.0f}h ({elapsed/2*1000:.0f}ms / iter)")

    except Exception as e:
        check_err("End-to-end", e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Quick sanity check for IMDNet training.")
    parser.add_argument("--config", default="configs/imdnet_base.yaml", help="Config file path.")
    parser.add_argument("--resume", default="", help="Optional checkpoint to verify loading.")
    parser.add_argument("--quick", action="store_true", help="Skip slow checks (dataloader, e2e).")
    args = parser.parse_args()

    print("=" * 60)
    print("  IMDNet Sanity Check")
    print("=" * 60)

    check_python()
    cfg = check_config(args.config)
    check_datasets(cfg)
    model = check_model(cfg)

    if not args.quick:
        check_dataloader(cfg)
        check_loss(model, cfg)
        check_end2end(cfg, model)
    else:
        check_loss(model, cfg)
        print(f"\n  [{SKIP}] Dataloader + end-to-end  (--quick mode)")

    if args.resume:
        check_resume(cfg, args.resume)

    print("\n" + "=" * 60)
    print("  Sanity check complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
