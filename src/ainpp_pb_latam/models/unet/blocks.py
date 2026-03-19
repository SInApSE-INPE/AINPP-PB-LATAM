from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# -----------------------------
# Building blocks
# -----------------------------
class DoubleConv(nn.Module):
    """
    Two consecutive convolutional layers with BatchNorm and ReLU (classic U-Net pattern).

    Parameters
    ----------
    in_channels:
        Number of input channels.
    out_channels:
        Number of output channels.
    kernel_size:
        Convolution kernel size.
    mid_channels:
        If provided, sets an intermediate channel size for the first convolution.
        If None, defaults to out_channels.
    norm_layer:
        Normalization layer constructor (defaults to nn.BatchNorm2d).
    activation:
        Activation function constructor (defaults to nn.ReLU).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        mid_channels: Optional[int] = None,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size must be a positive odd integer to preserve spatial dimensions via padding."
            )

        mid_channels = out_channels if mid_channels is None else mid_channels
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False
            ),
            norm_layer(mid_channels),
            activation(inplace=True),
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False
            ),
            norm_layer(out_channels),
            activation(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """
    Downsampling block: MaxPool(2) followed by DoubleConv.

    Parameters
    ----------
    in_channels:
        Number of input channels.
    out_channels:
        Number of output channels.
    kernel_size:
        Convolution kernel size used in DoubleConv.
    """

    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """
    Upsampling block: upsample + skip concatenation + DoubleConv.

    Two modes are supported:
    - bilinear=True: uses nn.Upsample + DoubleConv with a mid_channels reduction heuristic.
    - bilinear=False: uses nn.ConvTranspose2d + DoubleConv.

    Notes
    -----
    This block uses padding to handle odd-sized mismatches between skip and upsampled tensors.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # after concat: channels = in_channels + skip_channels
            # mid_channels heuristic: reduce to in_channels // 2 (common in U-Net variants)
            self.conv = DoubleConv(
                in_channels + skip_channels,
                out_channels,
                kernel_size=kernel_size,
                mid_channels=max(in_channels // 2, out_channels),
            )
        else:
            # transposed conv halves channels by design here
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(
                in_channels // 2 + skip_channels, out_channels, kernel_size=kernel_size
            )

    @staticmethod
    def _pad_to_match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Pad tensor x to match spatial size of ref (centered padding)."""
        diff_y = ref.size(2) - x.size(2)
        diff_x = ref.size(3) - x.size(3)

        if diff_y == 0 and diff_x == 0:
            return x

        pad_left = diff_x // 2
        pad_right = diff_x - pad_left
        pad_top = diff_y // 2
        pad_bottom = diff_y - pad_top
        return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self._pad_to_match(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
