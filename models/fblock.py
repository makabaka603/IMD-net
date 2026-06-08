import torch
import torch.nn as nn
import torch.nn.functional as F


class FBlock(nn.Module):
    """Fusion block for multi-level DI aggregation.

    Resizes all degradation features to the target scale, projects them to the
    target channel width, and combines them with softmax-normalised weights.
    """
    def __init__(self, in_channels_list, out_channels: int):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        # Log-scale weights -- softmax normalisation ensures non-negative, sum=1
        self.log_weights = nn.Parameter(torch.zeros(len(in_channels_list), out_channels, 1, 1))
        self.out = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, di_list, target_size):
        fused = 0.0
        # Softmax over the DI-level dimension for each channel
        w = F.softmax(self.log_weights, dim=0)  # [N_levels, C, 1, 1]
        for di, proj, wi in zip(di_list, self.proj, w):
            y = proj(di)
            if y.shape[-2:] != target_size:
                y = F.interpolate(y, size=target_size, mode='bilinear', align_corners=False)
            fused = fused + y * wi.unsqueeze(0)
        return self.out(fused)
