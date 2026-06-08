"""
Per-combo evaluation for IMDNet on the MDIR benchmark.

Evaluates the model on all 7 degradation combinations:
    H_R_N, H_R, H_N, R_N, H, R, N

Usage:
    python evaluate_mdir.py --weights checkpoints/imdnet_best.pth --data_root datasets/MDIR/test
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from models import IMDNet
from utils import load_image_tensor, psnr, ssim

# The 7 degradation combinations from the paper
COMBOS = ["H_R_N", "H_R", "H_N", "R_N", "H", "R", "N"]


@torch.no_grad()
def evaluate_combo(model, input_dir, target_dir, device, desc=""):
    """Evaluate PSNR and SSIM for a single combo."""
    input_dir = Path(input_dir)
    target_dir = Path(target_dir)
    if not input_dir.is_dir() or not target_dir.is_dir():
        return None

    img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    input_files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in img_exts])

    if not input_files:
        return None

    psnr_vals, ssim_vals = [], []
    for inp_path in tqdm(input_files, desc=desc, leave=False):
        name = inp_path.name
        tgt_path = target_dir / name
        if not tgt_path.exists():
            continue

        inp = load_image_tensor(inp_path, device=device)
        tgt = load_image_tensor(tgt_path, device=device)
        pred = model(inp)

        psnr_vals.append(psnr(pred, tgt))
        ssim_vals.append(ssim(pred, tgt))

    if not psnr_vals:
        return None
    return sum(psnr_vals) / len(psnr_vals), sum(ssim_vals) / len(ssim_vals)


def main():
    parser = argparse.ArgumentParser(description="MDIR per-combo evaluation.")
    parser.add_argument("--weights", required=True, help="Checkpoint path.")
    parser.add_argument("--data_root", required=True, help="Root dir containing combo subdirs.")
    parser.add_argument("--config", default="", help="Config path (default: from checkpoint).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    ckpt = torch.load(args.weights, map_location=device)
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = ckpt.get("config", None)
    if cfg is None:
        raise RuntimeError("No config found. Provide --config.")

    model = IMDNet(**cfg["model"]).to(device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Model loaded from {args.weights}")

    # Evaluate each combo
    data_root = Path(args.data_root)
    results = {}
    for combo in COMBOS:
        input_dir = data_root / combo / "input"
        target_dir = data_root / combo / "target"
        result = evaluate_combo(model, input_dir, target_dir, device, desc=combo)
        if result is not None:
            results[combo] = result
            print(f"  {combo:>8}: PSNR={result[0]:.4f}  SSIM={result[1]:.4f}")
        else:
            print(f"  {combo:>8}: SKIPPED (no data found)")

    # Summary
    if results:
        avg_psnr = sum(r[0] for r in results.values()) / len(results)
        avg_ssim = sum(r[1] for r in results.values()) / len(results)
        print(f"\n{'Average':>8}: PSNR={avg_psnr:.4f}  SSIM={avg_ssim:.4f}")


if __name__ == "__main__":
    main()
