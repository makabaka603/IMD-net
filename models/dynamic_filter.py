import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicFilter(nn.Module):
    """Learnable frequency separator.

    The paper describes DF as a learnable dynamic filter that separates high- and
    low-frequency components. Uses FFT with input-conditioned cutoff.

    NOTE: internal FFT ops always run in float32 to avoid AMP instability.
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
        # Force float32 for FFT ops to avoid AMP numerical issues
        orig_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        b, c, h, w = x_f32.shape

        fft = torch.fft.rfft2(x_f32, norm='ortho')

        yy = torch.fft.fftfreq(h, device=x_f32.device, dtype=torch.float32).view(h, 1)
        xx = torch.fft.rfftfreq(w, device=x_f32.device, dtype=torch.float32).view(1, w // 2 + 1)
        radius = torch.sqrt(xx * xx + yy * yy).view(1, 1, h, w // 2 + 1)

        # sample-wise cutoff in [0.08, 0.42]
        cutoff = 0.08 + 0.34 * self.radius_scale(x.to(orig_dtype)).view(b, 1, 1, 1)
        cutoff = cutoff.to(torch.float32)
        low_mask = torch.sigmoid((cutoff - radius) * self.sharpness.abs().to(torch.float32))
        high_mask = 1.0 - low_mask

        low = torch.fft.irfft2(fft * low_mask, s=(h, w), norm='ortho').to(orig_dtype)
        high = torch.fft.irfft2(fft * high_mask, s=(h, w), norm='ortho').to(orig_dtype)
        return high, low
