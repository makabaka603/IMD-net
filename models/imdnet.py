import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ConvDown, PixelShuffleUp, pad_to_multiple
from .didblock import DIDStage
from .tablock import TAStage
from .fblock import FBlock


class IMDNet(nn.Module):
    def __init__(
        self,
        img_channel: int = 3,
        width: int = 32,
        enc_blk_nums=(4, 4, 4, 8),
        middle_blk_num: int = 8,
        dec_blk_nums=(2, 2, 2, 2),
        branch_num: int = 4,
        tau: float = 0.2,
    ):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1)

        channels = [width, width * 2, width * 4, width * 8]
        # Encoder with multi-input mechanism (Paper Eq. 1):
        # each DIDStage receives downsampled degraded image processed by Convs
        self.encoders = nn.ModuleList([
            DIDStage(ch, n, img_channel) for ch, n in zip(channels, enc_blk_nums)
        ])
        self.downs = nn.ModuleList([ConvDown(channels[i], channels[i + 1]) for i in range(3)])
        self.middle = DIDStage(channels[-1], middle_blk_num, img_channel)

        # Decoder from deepest to shallowest: 256->128->64->32 when width=32
        dec_channels = list(reversed(channels))  # [256, 128, 64, 32]
        # Include middle block DI channels in FBlock fusion (Paper: multi-scale DI aggregation)
        di_channels = list(channels) + [channels[-1]]  # [32, 64, 128, 256, 256]
        self.fblocks = nn.ModuleList([
            FBlock(di_channels, ch) for ch in dec_channels
        ])
        self.up_after_stage = nn.ModuleList([
            PixelShuffleUp(dec_channels[i], dec_channels[i + 1]) for i in range(3)
        ])
        self.skip_proj = nn.ModuleList([
            nn.Conv2d(dec_channels[i] + dec_channels[i], dec_channels[i], 1) for i in range(4)
        ])
        self.decoders = nn.ModuleList([
            TAStage(ch, n, branch_num, tau) for ch, n in zip(dec_channels, dec_blk_nums)
        ])
        self.side_out = nn.ModuleList([nn.Conv2d(ch, img_channel, 3, 1, 1) for ch in dec_channels])

    def forward(self, inp: torch.Tensor, return_aux: bool = False):
        x, h, w = pad_to_multiple(inp, 16)
        feat = self.intro(x)

        # Generate multi-scale degraded inputs for the multi-input mechanism
        # (Paper: "Low-resolution degraded images are introduced into the main path")
        degraded_scales = [x]
        for _ in range(3):
            degraded_scales.append(
                F.interpolate(degraded_scales[-1], scale_factor=0.5, mode='bilinear', align_corners=False)
            )

        enc_feats, cf_list, di_list = [], [], []
        for i, enc in enumerate(self.encoders):
            feat, cf, di = enc(feat, degraded_scales[i])
            enc_feats.append(feat)
            cf_list.append(cf)
            di_list.append(di)
            if i < len(self.downs):
                feat = self.downs[i](feat)

        # Middle block also receives lowest-res degraded input
        feat, cf_mid, di_mid = self.middle(feat, degraded_scales[-1])

        # Include middle block DI for multi-scale fusion (Paper: aggregate DI across all levels)
        di_all = di_list + [di_mid]

        dec_feats = []
        gate_maps = []
        side_outputs = []
        rev_cf = list(reversed(cf_list))
        out = feat

        for i, dec in enumerate(self.decoders):
            # Align with corresponding clean skip feature
            skip = rev_cf[i]
            if out.shape[-2:] != skip.shape[-2:]:
                out = F.interpolate(out, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            out = self.skip_proj[i](torch.cat([out, skip], dim=1))
            fused_di = self.fblocks[i](di_all, target_size=out.shape[-2:])
            out, gates = dec(out, fused_di)
            dec_feats.append(out)
            gate_maps.extend(gates)
            side = self.side_out[i](out)
            side = F.interpolate(side, size=x.shape[-2:], mode='bilinear', align_corners=False)
            side_outputs.append(side + x)
            if i < len(self.up_after_stage):
                out = self.up_after_stage[i](out)

        residual = self.ending(out)
        restored = residual + x
        restored = restored[:, :, :h, :w]
        side_outputs = [s[:, :, :h, :w] for s in side_outputs]

        if return_aux:
            return {
                'out': restored,
                'side_outputs': side_outputs,
                'cf_list': cf_list,
                'di_list': di_list,
                'gates': gate_maps,
            }
        return restored
