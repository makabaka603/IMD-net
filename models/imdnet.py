import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvDown, PixelShuffleUp, pad_to_multiple
from .didblock import DIDStage
from .tablock import TAStage
from .fblock import FBlock


class IMDNet(nn.Module):
    """
    IMDNet.

    Paper architecture:
        Encoder DIDBlocks: [4, 4, 4, 8]
        Middle DIDBlocks: 8
        Decoder TABlocks: [2, 2, 2, 2]
    """

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

        self.img_channel = img_channel
        self.width = width

        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1)

        channels = [
            width,
            width * 2,
            width * 4,
            width * 8,
        ]

        self.encoders = nn.ModuleList(
            [
                DIDStage(
                    channels=channels[i],
                    num_blocks=enc_blk_nums[i],
                    img_channel=img_channel,
                )
                for i in range(4)
            ]
        )

        self.downs = nn.ModuleList(
            [
                ConvDown(channels[i], channels[i + 1])
                for i in range(3)
            ]
        )

        self.middle = DIDStage(
            channels=channels[-1],
            num_blocks=middle_blk_num,
            img_channel=img_channel,
        )

        dec_channels = list(reversed(channels))

        di_channels = list(channels) + [channels[-1]]

        self.fblocks = nn.ModuleList(
            [
                FBlock(di_channels, ch)
                for ch in dec_channels
            ]
        )

        self.skip_proj = nn.ModuleList(
            [
                nn.Conv2d(dec_channels[i] + dec_channels[i], dec_channels[i], 1)
                for i in range(4)
            ]
        )

        self.decoders = nn.ModuleList(
            [
                TAStage(
                    channels=dec_channels[i],
                    num_blocks=dec_blk_nums[i],
                    branch_num=branch_num,
                    tau=tau,
                )
                for i in range(4)
            ]
        )

        self.ups = nn.ModuleList(
            [
                PixelShuffleUp(dec_channels[i], dec_channels[i + 1])
                for i in range(3)
            ]
        )

        self.side_out = nn.ModuleList(
            [
                nn.Conv2d(ch, img_channel, 3, 1, 1)
                for ch in dec_channels
            ]
        )

    def forward(self, inp: torch.Tensor, return_aux: bool = False):
        x, h, w = pad_to_multiple(inp, multiple=16)

        feat = self.intro(x)

        degraded_scales = [x]

        for _ in range(3):
            degraded_scales.append(
                F.interpolate(
                    degraded_scales[-1],
                    scale_factor=0.5,
                    mode="bilinear",
                    align_corners=False,
                )
            )

        cf_list = []
        di_list = []

        for i, encoder in enumerate(self.encoders):
            feat, cf, di = encoder(feat, degraded_scales[i])

            cf_list.append(cf)
            di_list.append(di)

            if i < len(self.downs):
                feat = self.downs[i](feat)

        feat, cf_mid, di_mid = self.middle(feat, degraded_scales[-1])

        di_all = di_list + [di_mid]

        rev_cf = list(reversed(cf_list))

        out = feat

        side_outputs = []
        gate_maps = []

        for i, decoder in enumerate(self.decoders):
            skip = rev_cf[i]

            if out.shape[-2:] != skip.shape[-2:]:
                out = F.interpolate(
                    out,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            out = torch.cat([out, skip], dim=1)
            out = self.skip_proj[i](out)

            fused_di = self.fblocks[i](di_all, target_size=out.shape[-2:])

            out, gates = decoder(out, fused_di)
            gate_maps.extend(gates)

            side_residual = self.side_out[i](out)
            side_residual = F.interpolate(
                side_residual,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            side_restored = side_residual + x
            side_outputs.append(side_restored[:, :, :h, :w])

            if i < len(self.ups):
                out = self.ups[i](out)

        residual = self.ending(out)
        restored = residual + x
        restored = restored[:, :, :h, :w]

        if return_aux:
            return {
                "out": restored,
                "side_outputs": side_outputs,
                "cf_list": cf_list,
                "di_list": di_list,
                "gates": gate_maps,
            }

        return restored