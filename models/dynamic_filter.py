import torch
import torch.nn as nn


class DynamicFilter(nn.Module):
    """
    Learnable dynamic frequency filter.

    Paper role:
        FH, FL = DF(E)

    This implementation keeps the paper intention:
        - input-conditioned learnable frequency split
        - high-frequency / low-frequency decomposition
        - FFT computed in float32 for numerical stability
    """

    def __init__(self, channels: int, hidden: int = 16):
        super().__init__()

        hidden = max(hidden, channels // 4)

        self.cutoff_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

        self.temperature = nn.Parameter(torch.tensor(3.0))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        b, c, h, w = x.shape

        xf = x.float()

        fft = torch.fft.rfft2(xf, norm="ortho")

        yy = torch.fft.fftfreq(h, device=x.device, dtype=xf.dtype).view(1, 1, h, 1)
        xx = torch.fft.rfftfreq(w, device=x.device, dtype=xf.dtype).view(1, 1, 1, w // 2 + 1)
        radius = torch.sqrt(xx * xx + yy * yy + 1e-12)

        cutoff = 0.08 + 0.42 * self.cutoff_predictor(xf)
        cutoff = cutoff.view(b, c, 1, 1)

        temperature = self.temperature.abs().clamp(1.0, 5.0)

        low_mask = torch.sigmoid((cutoff - radius) * temperature)
        high_mask = 1.0 - low_mask

        low = torch.fft.irfft2(fft * low_mask, s=(h, w), norm="ortho")
        high = torch.fft.irfft2(fft * high_mask, s=(h, w), norm="ortho")

        return high.to(dtype), low.to(dtype)