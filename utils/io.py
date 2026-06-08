import os
import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_image_tensor(x, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = x.detach().clamp(0, 1).cpu()
    if x.ndim == 4:
        x = x[0]
    img = TF.to_pil_image(x)
    img.save(path)


def load_image_tensor(path, device='cpu'):
    img = Image.open(path).convert('RGB')
    return TF.to_tensor(img).unsqueeze(0).to(device)
