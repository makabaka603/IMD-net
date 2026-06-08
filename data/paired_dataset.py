import os
import random
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


def list_images(root):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob(os.path.join(root, f'*{ext}')))
    return sorted(files)


class PairedImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, patch_size=256, augment=True):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.patch_size = patch_size
        self.augment = augment
        input_files = list_images(input_dir)
        target_names = set(os.path.basename(p) for p in list_images(target_dir))
        self.pairs = []
        for ip in input_files:
            name = os.path.basename(ip)
            if name in target_names:
                self.pairs.append((ip, os.path.join(target_dir, name)))
        if not self.pairs:
            raise RuntimeError(f'No paired images found in {input_dir} and {target_dir}')

    def __len__(self):
        return len(self.pairs)

    def _load(self, path):
        return Image.open(path).convert('RGB')

    def _random_crop(self, inp, tgt):
        w, h = inp.size
        ps = self.patch_size
        if w < ps or h < ps:
            scale = max(ps / w, ps / h)
            nw, nh = int(w * scale + 0.5), int(h * scale + 0.5)
            inp = inp.resize((nw, nh), Image.BICUBIC)
            tgt = tgt.resize((nw, nh), Image.BICUBIC)
            w, h = inp.size
        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)
        return inp.crop((x, y, x + ps, y + ps)), tgt.crop((x, y, x + ps, y + ps))

    def __getitem__(self, idx):
        inp_path, tgt_path = self.pairs[idx]
        inp, tgt = self._load(inp_path), self._load(tgt_path)
        if self.patch_size is not None:
            inp, tgt = self._random_crop(inp, tgt)
        if self.augment:
            if random.random() < 0.5:
                inp, tgt = TF.hflip(inp), TF.hflip(tgt)
            if random.random() < 0.5:
                inp, tgt = TF.vflip(inp), TF.vflip(tgt)
            if random.random() < 0.5:
                inp, tgt = TF.rotate(inp, 90), TF.rotate(tgt, 90)
        inp = TF.to_tensor(inp)
        tgt = TF.to_tensor(tgt)
        return {'input': inp, 'target': tgt, 'name': os.path.basename(inp_path)}
