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
        self.encoders = nn.ModuleList([
            DIDStage(ch, n, img_channel) for ch, n in zip(channels, enc_blk_nums)
        ])
        self.downs = nn.ModuleList([ConvDown(channels[i], channels[i + 1]) for i in range(3)])
        self.middle = DIDStage(channels[-1], middle_blk_num, img_channel)

        dec_channels = list(reversed(channels))
        di_channels = list(channels) + [channels[-1]]
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
        # Side outputs at native decoder scales for multi-scale supervision
        self.side_out = nn.ModuleList([nn.Conv2d(ch, img_channel, 3, 1, 1) for ch in dec_channels])

    def forward(self, inp: torch.Tensor, return_aux: bool = False):
        x, h, w = pad_to_multiple(inp, 16)
        feat = self.intro(x)

        # Multi-scale degraded inputs
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

        feat, cf_mid, di_mid = self.middle(feat, degraded_scales[-1])
        di_all = di_list + [di_mid]

        dec_feats = []
        gate_maps = []
        side_outputs = []
        rev_cf = list(reversed(cf_list))
        out = feat

        for i, dec in enumerate(self.decoders):
            skip = rev_cf[i]
            if out.shape[-2:] != skip.shape[-2:]:
                out = F.interpolate(out, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            out = self.skip_proj[i](torch.cat([out, skip], dim=1))
            fused_di = self.fblocks[i](di_all, target_size=out.shape[-2:])
            out, gates = dec(out, fused_di)
            dec_feats.append(out)
            gate_maps.extend(gates)
            # Side output at native decoder scale (multi-scale supervision)
            side = self.side_out[i](out)
            side_outputs.append(side)  # no resize -- loss will downsample target
            if i < len(self.up_after_stage):
                out = self.up_after_stage[i](out)

        residual = self.ending(out)
        restored = residual + x
        restored = restored[:, :, :h, :w]

        if return_aux:
            return {
                'out': restored,
                'side_outputs': side_outputs,
                'cf_list': cf_list,
                'di_list': di_list,
                'gates': gate_maps,
            }
        return restored
