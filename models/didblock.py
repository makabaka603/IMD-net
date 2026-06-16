import torch
import torch.nn as nn

from .layers import NAFBlock, SimpleGate
from .dynamic_filter import DynamicFilter


class StatisticalCoefficient(nn.Module):
    """
    SC(.) in the paper:

        SC(.) =
            Conv1x1(SG(Conv1x1(GAP(.))))
          + Conv1x1(SG(Conv1x1(STD(.))))

    GAP and STD both produce channel-wise statistics.
    """

    def __init__(self, channels: int):
        super().__init__()

        hidden = channels * 2

        self.avg_conv1 = nn.Conv2d(channels, hidden, 1)
        self.avg_gate = SimpleGate()
        self.avg_conv2 = nn.Conv2d(hidden // 2, channels, 1)

        self.std_conv1 = nn.Conv2d(channels, hidden, 1)
        self.std_gate = SimpleGate()
        self.std_conv2 = nn.Conv2d(hidden // 2, channels, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.avg_pool(x)
        avg = self.avg_conv2(self.avg_gate(self.avg_conv1(avg)))

        var = torch.var(x.float(), dim=(2, 3), keepdim=True, unbiased=False)
        std = torch.sqrt(var + 1e-6).to(x.dtype)
        std = self.std_conv2(self.std_gate(self.std_conv1(std)))

        return avg + std


class DIDBlock(nn.Module):
    """
    Degradation Ingredient Decoupling Block.

    Paper equations:
        E  = NAFBlock(...)
        FH, FL = DF(E)
        DI = (SC(FH) + SC(FL) + SC(E)) / 6 * E
        CF = E - DI
    """

    def __init__(self, channels: int):
        super().__init__()

        self.spatial = NAFBlock(channels)
        self.df = DynamicFilter(channels)

        self.sc_h = StatisticalCoefficient(channels)
        self.sc_l = StatisticalCoefficient(channels)
        self.sc_e = StatisticalCoefficient(channels)

    def forward(self, x: torch.Tensor):
        e = self.spatial(x)

        fh, fl = self.df(e)

        coeff = (self.sc_h(fh) + self.sc_l(fl) + self.sc_e(e)) / 6.0

        di = coeff * e
        cf = e - di

        return e, cf, di


class DIDStage(nn.Module):
    """
    A stack of DIDBlocks.

    The paper introduces low-resolution degraded image features into the main path.
    This stage injects degraded image features once before the DIDBlock stack.
    """

    def __init__(self, channels: int, num_blocks: int, img_channel: int = 3):
        super().__init__()

        self.degraded_embed = nn.Sequential(
            nn.Conv2d(img_channel, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

        self.fuse = nn.Conv2d(channels * 2, channels, 1)

        self.blocks = nn.ModuleList(
            [DIDBlock(channels) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor, degraded: torch.Tensor = None):
        if degraded is not None:
            d = self.degraded_embed(degraded)
            if d.shape[-2:] != x.shape[-2:]:
                d = torch.nn.functional.interpolate(
                    d,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x = self.fuse(torch.cat([x, d], dim=1))

        cf = None
        di = None

        for block in self.blocks:
            x, cf, di = block(x)

        return x, cf, di