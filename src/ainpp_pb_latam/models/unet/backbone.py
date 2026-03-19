from __future__ import annotations

import logging
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ainpp_pb_latam.models.unet.blocks import DoubleConv, DownBlock, UpBlock

logger = logging.getLogger(__name__)


class UNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features: Sequence[int] = (64, 128, 256, 512, 1024),
        kernel_size: int = 3,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.kernel_size = kernel_size
        self.bilinear = bilinear
        self._validate_cfg()

        # Adjustment to bilinear
        features = list(features)
        if bilinear:
            features[-1] = features[-1] // 2

        self.stem = DoubleConv(in_channels, features[0], kernel_size=kernel_size)

        # Encoder
        self.down = nn.ModuleList()
        for i in range(len(features) - 1):
            self.down.append(DownBlock(features[i], features[i + 1], kernel_size=kernel_size))

        # Decoder
        self.up = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.up.append(
                UpBlock(
                    features[i],
                    features[i - 1],
                    features[i - 1],
                    kernel_size=kernel_size,
                    bilinear=bilinear,
                )
            )

        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _validate_cfg(self) -> None:
        if self.in_channels <= 0:
            raise ValueError("in_channels must be > 0.")
        if self.out_channels <= 0:
            raise ValueError("out_channels must be > 0.")
        if len(self.features) < 2:
            raise ValueError("features must have length >= 2.")
        if any(f <= 0 for f in self.features):
            raise ValueError("All feature sizes must be > 0.")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if not isinstance(self.bilinear, bool):
            raise ValueError("bilinear must be a boolean.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []

        x = self.stem(x)
        skips.append(x)

        for down_block in self.down:
            x = down_block(x)
            skips.append(x)

        # The last element is the bottleneck output; do not use it as a skip
        skips = skips[:-1]

        for i, up_block in enumerate(self.up):
            skip = skips[-(i + 1)]
            x = up_block(x, skip)

        return self.head(x)
