import os
import argparse
from glob import glob
import torch
import yaml
from tqdm import tqdm

from models import IMDNet
from utils import load_image_tensor, save_image_tensor

EXTS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.webp')


def collect_images(path):
    if os.path.isfile(path):
        return [path]
    files = []
    for e in EXTS:
        files.extend(glob(os.path.join(path, e)))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.weights, map_location=device)
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = ckpt.get('config', None)
    if cfg is None:
        raise RuntimeError('No config found in checkpoint. Provide --config.')

    model = IMDNet(**cfg['model']).to(device)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    os.makedirs(args.output, exist_ok=True)
    files = collect_images(args.input)
    with torch.no_grad():
        for p in tqdm(files):
            x = load_image_tensor(p, device=device)
            y = model(x)
            save_image_tensor(y, os.path.join(args.output, os.path.basename(p)))


if __name__ == '__main__':
    main()
