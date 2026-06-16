import torch
import torch.nn as nn
import torch.nn.functional as F


class FBlock(nn.Module):
    """
    Fusion block for multi-level degradation information.

    Paper role:
        aggregate degradation information DI from all encoder levels
        using learnable matrices / weights initialized to 1.
    """

    def __init__(self, in_channels_list, out_channels: int):
        super().__init__()

        self.proj = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in in_channels_list]
        )

        self.weights = nn.Parameter(
            torch.ones(len(in_channels_list), out_channels, 1, 1)
        )

        self.out = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, di_list, target_size):
        fused = None

        for di, proj, w in zip(di_list, self.proj, self.weights):
            y = proj(di)

            if y.shape[-2:] != target_size:
                y = F.interpolate(
                    y,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )

            term = y * w.unsqueeze(0)

            if fused is None:
                fused = term
            else:
                fused = fused + term

        fused = fused / max(len(di_list), 1)

        out = self.out(fused)

        return out
