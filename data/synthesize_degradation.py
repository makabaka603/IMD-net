"""
Synthetic multi-degradation data generator for IMDNet.

Creates paired data:
    input  = degraded image
    target = original clean image

Supported degradation ingredients:
    H: haze
    R: rain streaks
    N: additive Gaussian noise

Combinations (matching the MDIR paper setting):
    H_R_N, H_R, H_N, R_N, H, R, N

Example:
    python data/synthesize_degradation.py \
        --clean_dir datasets/clean_images \
        --output_dir datasets/MDIR \
        --mode all \
        --train_ratio 0.9 \
        --num_train_variants 2 \
        --seed 1
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VALID_COMBOS = {"H", "R", "N", "H_R", "H_N", "R_N", "H_R_N"}
DEFAULT_COMBOS = ["H_R_N", "H_R", "H_N", "R_N", "H", "R", "N"]


def list_images(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_rgb(img: Image.Image, path: Path) -> None:
    ensure_dir(path.parent)
    img.save(path)


def safe_stem(path: Path) -> str:
    """Build a unique stem from the last path components."""
    parts = path.with_suffix("").parts
    relevant = [p for p in parts if p != path.anchor][-3:]
    return "__".join(relevant)


def to_float01(img: Image.Image) -> np.ndarray:
    return np.asarray(img).astype(np.float32) / 255.0


def from_float01(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# Depth map (for haze)
# ---------------------------------------------------------------------------

def random_depth_map(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Create a simple pseudo depth map in [0, 1] for haze simulation."""
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    vertical = 1.0 - yy
    cx, cy = rng.uniform(0.25, 0.75), rng.uniform(0.15, 0.85)
    radial = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radial = radial / (radial.max() + 1e-6)

    low_res = rng.random((max(4, h // 64), max(4, w // 64)), dtype=np.float32)
    noise_img = Image.fromarray((low_res * 255).astype(np.uint8)).resize(
        (w, h), Image.BICUBIC
    )
    noise_img = noise_img.filter(
        ImageFilter.GaussianBlur(radius=max(3, min(h, w) // 32))
    )
    smooth_noise = np.asarray(noise_img).astype(np.float32) / 255.0

    depth = 0.55 * vertical + 0.25 * radial + 0.20 * smooth_noise
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    return depth.astype(np.float32)


# ---------------------------------------------------------------------------
# Haze
# ---------------------------------------------------------------------------

def add_haze(
    img: Image.Image,
    rng: np.random.Generator,
    beta: float | None = None,
    airlight: float | None = None,
) -> Image.Image:
    """Add synthetic haze via atmospheric scattering model."""
    arr = to_float01(img)
    h, w = arr.shape[:2]
    beta = rng.uniform(0.35, 2.20) if beta is None else float(beta)
    A = rng.uniform(0.78, 1.0) if airlight is None else float(airlight)
    depth = random_depth_map(h, w, rng)
    t = np.exp(-beta * depth)[..., None]
    hazy = arr * t + A * (1.0 - t)
    return from_float01(hazy)


# ---------------------------------------------------------------------------
# Rain
# ---------------------------------------------------------------------------

def add_rain(
    img: Image.Image,
    rng: np.random.Generator,
    streak_count: int | None = None,
    angle_deg: float | None = None,
) -> Image.Image:
    """Add synthetic rain streaks."""
    arr = to_float01(img)
    h, w = arr.shape[:2]

    strength = rng.uniform(0.2, 1.0)
    if streak_count is None:
        streak_count = int((0.0007 + 0.0045 * strength) * h * w)
    streak_count = max(1, streak_count)

    length_min = max(8, int(0.012 * min(h, w)))
    length_max = max(
        length_min + 1, int((0.035 + 0.055 * strength) * min(h, w))
    )
    width = 1 if strength < 0.55 else int(rng.integers(1, 3))
    angle = rng.uniform(-25, 25) if angle_deg is None else float(angle_deg)

    rain_mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(rain_mask)
    for _ in range(streak_count):
        x = rng.integers(-w // 10, w + w // 10)
        y = rng.integers(-h // 10, h + h // 10)
        length = int(rng.integers(length_min, length_max + 1))
        theta = math.radians(90 + angle + float(rng.normal(0, 4)))
        dx = math.cos(theta) * length
        dy = math.sin(theta) * length
        brightness = int(rng.integers(120, 256))
        draw.line((x, y, x + dx, y + dy), fill=brightness, width=width)

    blur_radius = 0.4 + 0.8 * strength
    rain_mask = rain_mask.filter(
        ImageFilter.GaussianBlur(radius=blur_radius)
    )
    rain = np.asarray(rain_mask).astype(np.float32) / 255.0

    alpha = 0.12 + 0.32 * strength
    rainy = (
        arr * (1.0 - alpha * rain[..., None] * 0.35)
        + alpha * rain[..., None]
    )
    return from_float01(rainy)


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

def add_noise(
    img: Image.Image,
    rng: np.random.Generator,
    sigma: float | None = None,
) -> Image.Image:
    """Add additive white Gaussian noise (range 0-50 in 8-bit scale)."""
    arr = to_float01(img)
    s = rng.uniform(0, 50) if sigma is None else float(sigma)
    s = np.clip(s, 0, 50) / 255.0
    noisy = arr + rng.normal(0.0, s, size=arr.shape).astype(np.float32)
    return from_float01(noisy)


# ---------------------------------------------------------------------------
# Combo dispatcher
# ---------------------------------------------------------------------------

def apply_combo(
    img: Image.Image, combo: str, rng: np.random.Generator
) -> Image.Image:
    """Apply degradation combination in fixed order H -> R -> N."""
    parts = set(combo.split("_"))
    out = img.copy()
    if "H" in parts:
        out = add_haze(out, rng)
    if "R" in parts:
        out = add_rain(out, rng)
    if "N" in parts:
        out = add_noise(out, rng)
    return out


# ---------------------------------------------------------------------------
# Split & generation
# ---------------------------------------------------------------------------

def split_files(
    files: Sequence[Path], train_ratio: float, seed: int
) -> Tuple[List[Path], List[Path]]:
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)
    n_train = int(len(files) * train_ratio)
    return files[:n_train], files[n_train:]


def _rand_combo(rng: np.random.Generator, combos: Sequence[str]) -> str:
    return combos[int(rng.integers(0, len(combos)))]


def generate_flat_split(
    files: Sequence[Path],
    out_root: Path,
    split: str,
    rng: np.random.Generator,
    num_variants: int,
    combos: Sequence[str],
) -> None:
    input_dir = out_root / split / "input"
    target_dir = out_root / split / "target"
    ensure_dir(input_dir)
    ensure_dir(target_dir)

    for clean_path in tqdm(files, desc=f"Generating {split}"):
        clean = load_rgb(clean_path)
        for v in range(num_variants):
            combo = _rand_combo(rng, combos)
            degraded = apply_combo(clean, combo, rng)
            stem = safe_stem(clean_path)
            base = f"{stem}__{combo}__v{v:02d}.png"
            save_rgb(degraded, input_dir / base)
            save_rgb(clean, target_dir / base)


def generate_combo_tests(
    files: Sequence[Path],
    out_root: Path,
    split: str,
    rng: np.random.Generator,
    combos: Sequence[str],
) -> None:
    for combo in combos:
        input_dir = out_root / split / combo / "input"
        target_dir = out_root / split / combo / "target"
        ensure_dir(input_dir)
        ensure_dir(target_dir)
        for clean_path in tqdm(files, desc=f"Generating {split}/{combo}"):
            clean = load_rgb(clean_path)
            degraded = apply_combo(clean, combo, rng)
            name = f"{safe_stem(clean_path)}.png"
            save_rgb(degraded, input_dir / name)
            save_rgb(clean, target_dir / name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_combos(raw: str) -> List[str]:
    """Parse and validate comma-separated combo strings."""
    combos = [c.strip().upper() for c in raw.split(",") if c.strip()]
    for c in combos:
        if c not in VALID_COMBOS:
            raise ValueError(
                f"Unknown combo {c!r}. "
                f"Valid options: {', '.join(sorted(VALID_COMBOS))}"
            )
    return combos


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic MDIR paired data."
    )
    parser.add_argument("--clean_dir", required=True, help="Clean images dir.")
    parser.add_argument(
        "--output_dir", required=True, help="Output dataset directory."
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["train", "val", "test", "all"],
        help="Which part to generate.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Clean-image ratio for train when mode=all.",
    )
    parser.add_argument(
        "--num_train_variants",
        type=int,
        default=2,
        help="Degraded copies per clean train image.",
    )
    parser.add_argument(
        "--num_val_variants",
        type=int,
        default=1,
        help="Degraded copies per clean val image.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output_dir before generation.",
    )
    parser.add_argument(
        "--combos",
        type=str,
        default=",".join(DEFAULT_COMBOS),
        help="Comma-separated combinations.",
    )
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    out_root = Path(args.output_dir)
    combos = parse_combos(args.combos)

    files = list_images(clean_dir)
    if not files:
        raise RuntimeError(f"No images found in {clean_dir}")

    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)
    ensure_dir(out_root)

    rng = np.random.default_rng(args.seed)

    if args.mode == "train":
        generate_flat_split(
            files, out_root, "train", rng, args.num_train_variants, combos
        )
    elif args.mode == "val":
        generate_flat_split(
            files, out_root, "val", rng, args.num_val_variants, combos
        )
    elif args.mode == "test":
        generate_combo_tests(files, out_root, "test", rng, combos)
    else:
        train_files, val_files = split_files(
            files, args.train_ratio, args.seed
        )
        generate_flat_split(
            train_files, out_root, "train", rng, args.num_train_variants, combos
        )
        generate_flat_split(
            val_files, out_root, "val", rng, args.num_val_variants, combos
        )
        generate_combo_tests(val_files, out_root, "test", rng, combos)

    print(f"Done. Dataset written to: {out_root}")
    print("Train/val format: split/input + split/target")
    print("Test format: test/<combo>/input + test/<combo>/target")


if __name__ == "__main__":
    main()
