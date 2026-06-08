import torch
import torch.nn as nn
from .layers import LayerNorm2d, SimpleGate, SCA


class FunctionalBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            SimpleGate(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x):
        return self.net(x)


class TABlock(nn.Module):
    """Task Adaptation Block with sparse branch activation (Paper Eq. 5-9).

    Degradation-aware context injection, branch gating with thresholded fusion.
    Multiple activated branches accumulate their contributions.
    """
    def __init__(self, channels: int, branch_num: int = 4, tau: float = 0.2):
        super().__init__()
        self.tau = tau
        self.branch_num = branch_num
        self.norm = LayerNorm2d(channels)
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.Conv2d(channels * 2, channels * 2, 3, 1, 1, groups=channels * 2),
            SimpleGate(),
            SCA(channels),
            nn.Conv2d(channels, channels, 1),
        )
        self.di_proj = nn.Conv2d(channels, channels, 1)
        self.general = FunctionalBranch(channels)
        self.branches = nn.ModuleList([FunctionalBranch(channels) for _ in range(branch_num)])
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, branch_num, 1), nn.Sigmoid())
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, di):
        dc = self.context(self.norm(x)) + self.di_proj(di)
        y = self.general(dc + x)
        weights = self.gate(dc)  # B,N,1,1
        for i, branch in enumerate(self.branches):
            wi = weights[:, i:i+1]
            mask = (wi >= self.tau).to(y.dtype)
            # Accumulate activated branch outputs (fix: replaced replacement with accumulation)
            y = y + mask * wi * branch(y)
        return x + self.out(y), weights


class TAStage(nn.Module):
    def __init__(self, channels: int, num_blocks: int, branch_num: int = 4, tau: float = 0.2):
        super().__init__()
        self.blocks = nn.ModuleList([TABlock(channels, branch_num, tau) for _ in range(num_blocks)])

    def forward(self, x, di):
        gates = []
        for blk in self.blocks:
            x, w = blk(x, di)
            gates.append(w)
        return x, gates
