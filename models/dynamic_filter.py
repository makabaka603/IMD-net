import torch
import torch.nn as nn


class DynamicFilter(nn.Module):
    """Learnable frequency separator.

    The paper describes DF as a learnable dynamic filter that separates high- and
    low-frequency components. Here we implement it using an input-conditioned
    low-pass/high-pass split in the FFT domain.
    """
    def __init__(self, channels: int, hidden: int = 16):
        super().__init__()
        self.radius_scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )
        self.sharpness = nn.Parameter(torch.tensor(10.0))

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        fft = torch.fft.rfft2(x, norm='ortho')

        yy = torch.fft.fftfreq(h, device=x.device, dtype=x.dtype).view(h, 1)
        xx = torch.fft.rfftfreq(w, device=x.device, dtype=x.dtype).view(1, w // 2 + 1)
        radius = torch.sqrt(xx * xx + yy * yy).view(1, 1, h, w // 2 + 1)

        # sample-wise cutoff in [0.08, 0.42]
        cutoff = 0.08 + 0.34 * self.radius_scale(x).view(b, 1, 1, 1)
        low_mask = torch.sigmoid((cutoff - radius) * self.sharpness.abs())
        high_mask = 1.0 - low_mask

        low = torch.fft.irfft2(fft * low_mask, s=(h, w), norm='ortho')
        high = torch.fft.irfft2(fft * high_mask, s=(h, w), norm='ortho')
        return high, low
