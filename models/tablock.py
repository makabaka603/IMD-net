import torch
import torch.nn as nn

from .layers import LayerNorm2d, SimpleGate, SCA


class FunctionalBranch(nn.Module):
    """
    Block(.) in TABlock:

        Block(.) = Conv1x1(SG(Conv1x1(.)))
    """

    def __init__(self, channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels * 2, 1)
        self.sg = SimpleGate()
        self.conv2 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return self.conv2(self.sg(self.conv1(x)))


class TABlock(nn.Module):
    """
    Task Adaptation Block.

    Paper equations:
        DC = Conv1x1(SCA(SG(DWConv3x3(Conv1x1(LN(D))) + DI)))
        DX0 = Block0(DC + D)
        Wn = Sigmoid(Conv1x1(GAP(DC)))
        DXn = Wn * Blockn(DXn-1), if Wn >= tau
            = DXn-1, otherwise
    """

    def __init__(self, channels: int, branch_num: int = 4, tau: float = 0.2):
        super().__init__()

        self.channels = channels
        self.branch_num = branch_num
        self.tau = tau

        self.norm = LayerNorm2d(channels)

        self.context_conv1 = nn.Conv2d(channels, channels * 2, 1)
        self.context_dw = nn.Conv2d(
            channels * 2,
            channels * 2,
            3,
            1,
            1,
            groups=channels * 2,
        )
        self.context_sg = SimpleGate()
        self.context_sca = SCA(channels)
        self.context_conv2 = nn.Conv2d(channels, channels, 1)

        self.di_proj = nn.Conv2d(channels, channels, 1)

        self.general_branch = FunctionalBranch(channels)

        self.branches = nn.ModuleList(
            [FunctionalBranch(channels) for _ in range(branch_num)]
        )

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_num, 1),
            nn.Sigmoid(),
        )

        self.out = nn.Conv2d(channels, channels, 1)

    def context(self, x, di):
        y = self.norm(x)
        y = self.context_conv1(y)
        y = self.context_dw(y)
        y = self.context_sg(y)
        y = y + self.di_proj(di)
        y = self.context_sca(y)
        y = self.context_conv2(y)
        return y

    def forward(self, x: torch.Tensor, di: torch.Tensor):
        dc = self.context(x, di)

        y = self.general_branch(dc + x)

        weights = self.gate(dc)

        for i, branch in enumerate(self.branches):
            wi = weights[:, i:i + 1]

            branch_out = branch(y)
            candidate = wi * branch_out

            # Replace only non-finite elements, do not clamp normal activations.
            if self.training:
                mask = (wi >= self.tau).to(y.dtype)
                y = mask * candidate + (1.0 - mask) * y
            else:
                if (wi >= self.tau).any():
                    mask = (wi >= self.tau).to(y.dtype)
                    y = mask * candidate + (1.0 - mask) * y

        out = self.out(y)
        res = x + out

        return res, weights


class TAStage(nn.Module):
    def __init__(
        self,
        channels: int,
        num_blocks: int,
        branch_num: int = 4,
        tau: float = 0.2,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TABlock(
                    channels=channels,
                    branch_num=branch_num,
                    tau=tau,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor, di: torch.Tensor):
        gates = []

        for block in self.blocks:
            x, w = block(x, di)
            gates.append(w)

        return x, gates