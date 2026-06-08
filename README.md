# IMDNet PyTorch Reimplementation (modular)

This is a clean, runnable PyTorch implementation of **Learning to Restore Multi-Degraded Images via Ingredient Decoupling and Task-Aware Path Adaptation**.

It implements the paper's core modules:

- DIDBlock: degradation ingredient decoupling block
- FBlock: learnable multi-level degradation information fusion (softmax normalised)
- TABlock: task-aware path adaptation block with sparse branch activation
- IMDNet: encoder-decoder restoration network with multi-scale side outputs
- Multi-term loss: Charbonnier + edge + FFT + decoupling (cos²) loss

> Note: The paper does not disclose every low-level implementation detail. This repository is a faithful engineering implementation of the described equations and figures, designed to be directly trainable and easy to modify.

## Install

```bash
pip install torch torchvision pillow tqdm pyyaml numpy
```

## Dataset

### Option A: Download existing MDIR data

Place paired images in the following structure:

```text
datasets/MDIR/
  train/
    input/   degraded images
    target/  clean images
  val/
    input/
    target/
```

Filenames in `input` and `target` should match.

### Option B: Synthesise your own (recommended for quick start)

Use the built-in degradation synthesiser to generate multi-degradation pairs (haze + rain + noise) from clean images:

```bash
# Generate all splits (train/val/test) with 7 degradation combinations
python data/synthesize_degradation.py \
    --clean_dir datasets/clean_images \
    --output_dir datasets/MDIR \
    --mode all \
    --train_ratio 0.9 \
    --num_train_variants 3 \
    --seed 42

# Generate only training data with custom combos
python data/synthesize_degradation.py \
    --clean_dir datasets/clean_images \
    --output_dir datasets/MDIR \
    --mode train \
    --num_train_variants 5 \
    --combos H_R_N,H_R,H_N,R_N,H,R,N
```

Supported degradations: **H** (haze), **R** (rain streaks), **N** (Gaussian noise).
7 combinations: H_R_N, H_R, H_N, R_N, H, R, N — matching the paper's MDIR setting.

## Train

```bash
python train.py --config configs/imdnet_base.yaml
```

To resume from a checkpoint:

```bash
python train.py --config configs/imdnet_base.yaml --resume checkpoints/imdnet_latest.pth
```

**Gradient accumulation**: the default config uses `batch_size: 4` and `accumulation_steps: 8`,
giving an effective batch size of 32. Adjust `accumulation_steps` based on your GPU memory.

## Inference

```bash
python infer.py --weights checkpoints/imdnet_best.pth --input test/input --output results
```

## Evaluate

### Per-combo evaluation (MDIR benchmark)

Evaluate all 7 degradation combinations in one pass:

```bash
python evaluate_mdir.py \
    --weights checkpoints/imdnet_best.pth \
    --data_root datasets/MDIR/test
```

### Quick PSNR/SSIM

```bash
python evaluate.py --pred results --gt test/target
```
