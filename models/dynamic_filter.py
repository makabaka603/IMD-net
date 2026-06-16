import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicFilter(nn.Module):
    """
    Spatial-domain dynamic frequency filter (no FFT).

    Decomposes features into low-frequency (smooth) and high-frequency (detail)
    components using a learnable channel-wise blend between the input and a
    fixed Gaussian-blurred version.

    Paper role (same as original):
        FH, FL = DF(E)
        FH = high-frequency features, FL = low-frequency features

    Why spatial domain:
        The original FFT-based implementation produces unstable gradients
        during backpropagation, causing NaN during training. This spatial
        version achieves the same frequency decomposition effect with
        well-behaved gradients.

    How it works:
        - A fixed Gaussian kernel (sigma=2.0, size=15) blurs the input
        - A learnable per-channel gate predicts blending ratios [0, 1]
        - Low = blend * blurred + (1 - blend) * original  (adaptive low-pass)
        - High = original - Low                            (high-pass = detail)
    """

    def __init__(self, channels: int, hidden: int = 16):
        super().__init__()

        hidden = max(hidden, channels // 4)

        # Learnable per-channel blending ratio
        self.blend_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),  # [0, 1]: 0=no blur(sharp), 1=full blur(smooth)
        )

        # Fixed Gaussian blur kernel (moderate blur, covers most cutoff ranges)
        kernel_size = 15
        sigma = 2.0
        self.register_buffer("kernel", self._create_gaussian(kernel_size, sigma))
        self.kernel_size = kernel_size

    def _create_gaussian(self, size: int, sigma: float) -> torch.Tensor:
        """Create a 2D Gaussian kernel."""
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-0.5 * (coords / sigma) ** 2)
        g = g / g.sum()
        kernel = g[:, None] @ g[None, :]
        return kernel.view(1, 1, size, size)

    def forward(self, x: torch.Tensor):
        """Decompose x into high-frequency and low-frequency components."""
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # Apply fixed Gaussian blur as low-pass filter
        kernel = self.kernel.to(device, dtype=dtype).repeat(C, 1, 1, 1)
        pad = self.kernel_size // 2
        x_blur = F.conv2d(x, kernel, padding=pad, groups=C)

        # Predict per-channel blending ratio
        blend = self.blend_net(x)  # (B, C, 1, 1), range [0, 1]

        # Low-frequency: adaptive blend between original and blurred
        low = blend * x_blur + (1.0 - blend) * x

        # High-frequency: detail = original - smooth
        high = x - low

        return high, low
