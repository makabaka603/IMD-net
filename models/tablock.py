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

    - Training: soft gating (all branches contribute with learned weights).
    - Inference: true sparse activation (branches below tau are skipped).
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

        # Training: soft gating (all branches contribute)
        # Inference: true sparse ¡ª skip branches below tau entirely
        for i, branch in enumerate(self.branches):
            wi = weights[:, i:i+1]
            if self.training:
                y = y + wi * branch(y)
            else:
                # Truly skip: don't compute branch(y) if no sample activates it
                batch_active = (wi >= self.tau)
                if batch_active.any():
                    mask = batch_active.to(y.dtype)
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
