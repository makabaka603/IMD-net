# IMDNet PyTorch Reimplementation (modular)

This is a clean, runnable PyTorch implementation of **Learning to Restore Multi-Degraded Images via Ingredient Decoupling and Task-Aware Path Adaptation**.

It implements the paper's core modules:

- DIDBlock: degradation ingredient decoupling block
- FBlock: learnable multi-level degradation information fusion
- TABlock: task-aware path adaptation block with sparse branch activation
- IMDNet: encoder-decoder restoration network with multi-scale side outputs
- Multi-term loss: Charbonnier + edge + FFT + decoupling loss

> Note: The paper does not disclose every low-level implementation detail, especially the exact dynamic filter parameterization and branch count. This repository is a faithful engineering implementation of the described equations and figures, designed to be directly trainable and easy to modify.

## Install

```bash
pip install torch torchvision pillow tqdm pyyaml numpy
```

## Dataset format

Use paired folders:

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

## Train

```bash
python train.py --config configs/imdnet_base.yaml
```

## Inference

```bash
python infer.py --weights checkpoints/imdnet_best.pth --input test/input --output results
```

## Evaluate PSNR/SSIM

```bash
python evaluate.py --pred results --gt test/target
```
