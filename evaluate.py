import os
import argparse
from glob import glob
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from utils import psnr, ssim

EXTS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.webp')


def list_images(root):
    files = []
    for e in EXTS:
        files.extend(glob(os.path.join(root, e)))
    return sorted(files)


def load(path, device):
    return TF.to_tensor(Image.open(path).convert('RGB')).unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True)
    parser.add_argument('--gt', required=True)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pred_files = list_images(args.pred)
    vals_p, vals_s = [], []
    for pp in pred_files:
        name = os.path.basename(pp)
        gp = os.path.join(args.gt, name)
        if not os.path.exists(gp):
            continue
        p, g = load(pp, device), load(gp, device)
        vals_p.append(psnr(p, g))
        vals_s.append(ssim(p, g))
    print(f'Images: {len(vals_p)}')
    print(f'PSNR: {sum(vals_p)/len(vals_p):.4f}')
    print(f'SSIM: {sum(vals_s)/len(vals_s):.4f}')


if __name__ == '__main__':
    main()
