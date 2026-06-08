import math
import torch
import torch.nn.functional as F


def psnr(pred, target, max_val=1.0):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 99.0
    return 20 * math.log10(max_val / math.sqrt(mse))


def _gaussian_window(window_size=11, sigma=1.5, channel=3, device='cpu'):
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] @ g[None, :]).view(1, 1, window_size, window_size)
    return w.repeat(channel, 1, 1, 1)


def ssim(pred, target, window_size=11):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    c = pred.shape[1]
    window = _gaussian_window(window_size, channel=c, device=pred.device)
    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=c)
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=c)
    mu1_sq, mu2_sq, mu12 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=c) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=c) - mu12
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    val = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return val.mean().item()
