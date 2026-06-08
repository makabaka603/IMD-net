import torch
import torch.nn as nn
from .layers import NAFBlock, SimpleGate
from .dynamic_filter import DynamicFilter


class StatisticalCoefficient(nn.Module):
    """SC(.) in Eq. (3): combine GAP and STD statistics through 1x1 conv + SG."""
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels * 2, 8)
        self.avg_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            SimpleGate(),
            nn.Conv2d(hidden // 2, channels, 1),
        )
        self.std_conv1 = nn.Conv2d(channels, hidden, 1)
        self.std_gate = SimpleGate()
        self.std_conv2 = nn.Conv2d(hidden // 2, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.avg_branch(x)
        std = torch.sqrt(torch.var(x, dim=(2, 3), keepdim=True, unbiased=False) + 1e-6)
        std = self.std_conv2(self.std_gate(self.std_conv1(std)))
        return avg + std


class DIDBlock(nn.Module):
    """Degradation Ingredient Decoupling Block (Paper Eq. 1-4).

    Supports optional multi-input fusion: fc_1x1([E_{l-1}, Convs(I_l)]) before NAFBlock.

    Outputs:
        e: encoded feature for the next encoder level
        cf: decoupled clean feature for skip connection
        di: degradation ingredient feature for path adaptation
    """
    def __init__(self, channels: int, has_fusion: bool = False):
        super().__init__()
        self.has_fusion = has_fusion
        if has_fusion:
            # Paper Eq. (1): 1x1 conv to fuse concatenated encoder feat + processed degraded image
            self.img_fusion = nn.Conv2d(channels * 2, channels, 1)
        self.spatial = NAFBlock(channels)
        self.df = DynamicFilter(channels)
        self.sc_h = StatisticalCoefficient(channels)
        self.sc_l = StatisticalCoefficient(channels)
        self.sc_e = StatisticalCoefficient(channels)
        self.refine_di = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, img_feat: torch.Tensor = None):
        if img_feat is not None and self.has_fusion:
            x = self.img_fusion(torch.cat([x, img_feat], dim=1))
        e = self.spatial(x)
        fh, fl = self.df(e)
        coeff = (self.sc_h(fh) + self.sc_l(fl) + self.sc_e(e)) / 6.0
        di = self.refine_di(coeff * e)
        cf = e - di
        return e, cf, di


class DIDStage(nn.Module):
    """Stack several DIDBlocks; return last e/cf/di.

    The first DIDBlock in the stage receives the multi-input fusion
    (degraded image features concatenated with encoder features, per Paper Eq. 1).
    """
    def __init__(self, channels: int, num_blocks: int, img_channel: int = 3):
        super().__init__()
        # Paper: Convs(I_l) — process downsampled degraded image before concatenation
        self.degraded_stem = nn.Sequential(
            nn.Conv2d(img_channel, channels, 3, 1, 1),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.blocks = nn.ModuleList([DIDBlock(channels, has_fusion=(i == 0)) for i in range(num_blocks)])

    def forward(self, x: torch.Tensor, degraded: torch.Tensor = None):
        img_feat = self.degraded_stem(degraded) if degraded is not None else None
        cf = di = None
        for blk in self.blocks:
            x, cf, di = blk(x, img_feat)
            img_feat = None  # only inject before the first block
        return x, cf, di
