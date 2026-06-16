import os
import sys
import argparse
import traceback
import py_compile
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader


def check_compile(files):
    print("\n========== 1. Check Python syntax ==========")
    ok = True
    for f in files:
        if not Path(f).exists():
            print(f"[WARN] Missing file: {f}")
            continue
        try:
            py_compile.compile(f, doraise=True)
            print(f"[OK] {f}")
        except Exception as e:
            ok = False
            print(f"[FAIL] {f}")
            print(e)
    return ok


def load_config(config_path):
    print("\n========== 2. Load YAML config ==========")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise RuntimeError("YAML 配置读取失败：cfg 不是 dict，请检查 configs/imdnet_base.yaml 格式。")

    print("[OK] Config loaded:", config_path)
    print("Top-level keys:", list(cfg.keys()))
    return cfg


def check_cuda():
    print("\n========== 3. Check CUDA ==========")
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("torch cuda version:", torch.version.cuda)

    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"gpu memory: {mem:.2f} GB")
        return torch.device("cuda")

    print("[WARN] CUDA 不可用，将使用 CPU 测试。正式训练必须保证 CUDA 可用。")
    return torch.device("cpu")


def check_dirs(cfg):
    print("\n========== 4. Check dataset paths ==========")
    required = [
        "train_input_dir",
        "train_target_dir",
        "val_input_dir",
        "val_target_dir",
    ]

    for k in required:
        if k not in cfg:
            raise KeyError(f"配置文件缺少字段：{k}")
        p = Path(cfg[k])
        print(f"{k}: {p}")
        if not p.exists():
            raise FileNotFoundError(f"路径不存在：{p}")

    train_input_count = len(list(Path(cfg["train_input_dir"]).glob("*")))
    train_target_count = len(list(Path(cfg["train_target_dir"]).glob("*")))
    val_input_count = len(list(Path(cfg["val_input_dir"]).glob("*")))
    val_target_count = len(list(Path(cfg["val_target_dir"]).glob("*")))

    print("train input files:", train_input_count)
    print("train target files:", train_target_count)
    print("val input files:", val_input_count)
    print("val target files:", val_target_count)

    if train_input_count == 0 or train_target_count == 0:
        raise RuntimeError("训练集 input 或 target 为空，请先检查数据是否放对。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/imdnet_base.yaml")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--random_only", action="store_true",
                        help="不读数据，直接用随机张量测试模型 forward/backward。")
    args = parser.parse_args()

    critical_files = [
        "train.py",
        "infer.py",
        "evaluate.py",
        "evaluate_mdir.py",
        "models/imdnet.py",
        "models/didblock.py",
        "models/tablock.py",
        "models/fblock.py",
        "models/dynamic_filter.py",
        "models/layers.py",
        "data/paired_dataset.py",
        "utils/losses.py",
    ]

    compile_ok = check_compile(critical_files)
    if not compile_ok:
        print("\n[STOP] 有 Python 文件语法检查失败。请先修复代码换行/语法问题，再训练。")
        sys.exit(1)

    cfg = load_config(args.config)
    device = check_cuda()

    print("\n========== 5. Import project modules ==========")
    try:
        from models import IMDNet
        from utils import IMDNetLoss
        print("[OK] Imported IMDNet and IMDNetLoss")
    except Exception:
        print("[FAIL] 项目模块导入失败。错误如下：")
        traceback.print_exc()
        sys.exit(1)

    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {})

    print("\n========== 6. Build model and loss ==========")
    model = IMDNet(**model_cfg).to(device)
    criterion = IMDNetLoss(**loss_cfg).to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[OK] Model built. Params: {param_count:.2f} M")

    if args.random_only:
        print("\n========== 7. Random tensor test ==========")
        x = torch.rand(args.batch_size, 3, args.patch_size, args.patch_size, device=device)
        y = torch.rand(args.batch_size, 3, args.patch_size, args.patch_size, device=device)
        names = ["random_tensor"]
    else:
        check_dirs(cfg)
        print("\n========== 7. Load one mini-batch ==========")
        try:
            from data import PairedImageDataset

            dataset = PairedImageDataset(
                cfg["train_input_dir"],
                cfg["train_target_dir"],
                patch_size=args.patch_size,
                augment=True,
            )

            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )

            batch = next(iter(loader))
            x = batch["input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            names = batch.get("name", ["unknown"])

            print("[OK] Batch loaded")
            print("names:", names[:3] if isinstance(names, list) else names)
        except Exception:
            print("[FAIL] 数据读取失败。错误如下：")
            traceback.print_exc()
            sys.exit(1)

    print("input shape:", tuple(x.shape))
    print("target shape:", tuple(y.shape))
    print("input min/max:", float(x.min()), float(x.max()))
    print("target min/max:", float(y.min()), float(y.max()))

    print("\n========== 8. Forward + loss + backward ==========")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

    try:
        optimizer.zero_grad(set_to_none=True)

        outputs = model(x, return_aux=True)
        loss, logs = criterion(outputs, y)

        print("[OK] Forward success")
        print("main output shape:", tuple(outputs["out"].shape))
        print("side output shapes:", [tuple(s.shape) for s in outputs.get("side_outputs", [])])
        print("cf shapes:", [tuple(t.shape) for t in outputs.get("cf_list", [])])
        print("di shapes:", [tuple(t.shape) for t in outputs.get("di_list", [])])
        print("loss:", float(loss.detach()))
        print("logs:", logs)

        if not torch.isfinite(loss):
            raise RuntimeError("loss 出现 NaN 或 Inf。")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        print("[OK] Backward + optimizer step success")

    except RuntimeError as e:
        print("[FAIL] Forward/backward 失败。错误如下：")
        print(e)

        if "out of memory" in str(e).lower():
            print("\n显存不足处理建议：")
            print("1. 先用 --patch_size 64")
            print("2. 或者 --batch_size 1")
            print("3. 正式训练时再逐步调回 patch_size=256")
        sys.exit(1)

    print("\n========== 9. Save temporary checkpoint ==========")
    save_dir = Path("checkpoints_sanity")
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "sanity_test.pth"

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "loss": float(loss.detach().cpu()),
        },
        ckpt_path,
    )

    print("[OK] Checkpoint saved:", ckpt_path)

    print("\n========== Sanity check finished ==========")
    print("结果：临时测试通过，可以开始短训练。")


if __name__ == "__main__":
    main()
