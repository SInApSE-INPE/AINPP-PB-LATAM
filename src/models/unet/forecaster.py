from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet.backbone import UNet2D

logger = logging.getLogger(__name__)


class BaseForecaster(nn.Module):
    def _apply_nonnegativity(self, x, mode="relu"):
        if mode == "relu": return F.relu(x)
        if mode == "softplus": return F.softplus(x)
        return x


class UNetMultiHorizon(BaseForecaster):
    def __init__(
        self,
        input_timesteps: int,
        input_channels: int,
        output_timesteps: int,
        output_channels: int = 1,
        features: list = [64, 128, 256, 512],
        kernel_size: int = 3,
        bilinear: bool = True,
        nonnegativity: str = "relu"
    ):
        super().__init__()
        self.output_timesteps = output_timesteps
        self.out_ch = output_channels
        self.nonnegativity = nonnegativity
        
        self.unet = UNet2D(
            in_channels=input_timesteps * input_channels,
            out_channels=output_timesteps * output_channels,
            features=features,
            kernel_size=kernel_size,
            bilinear=bilinear
        )

    def forward(self, x):
        b, tin, cin, h, w = x.shape
        # Flatten temporal dimension
        x_flat = x.reshape(b, tin * cin, h, w)
        y_flat = self.unet(x_flat)
        # Reshape back
        y = y_flat.reshape(b, self.output_timesteps, self.out_ch, h, w)
        return self._apply_nonnegativity(y, self.nonnegativity)


class UNetAutoRegressive(BaseForecaster):
    def __init__(
        self,
        input_timesteps: int,
        input_channels: int,
        output_timesteps: int,
        features: list = [64, 128, 256, 512],
        kernel_size: int = 3,
        bilinear: bool = True,
        nonnegativity: str = "relu"
    ):
        super().__init__()
        self.input_timesteps = input_timesteps
        self.input_channels = input_channels
        self.output_timesteps = output_timesteps
        self.features = features
        self.kernel_size = kernel_size
        self.bilinear = bilinear
        self.nonnegativity = nonnegativity
        
        self.unet = UNet2D(
            in_channels=self.input_timesteps * self.input_channels,
            out_channels=self.input_channels, 
            features=self.features,
            kernel_size=self.kernel_size,
            bilinear=self.bilinear
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Input tensor with shape (B, Tin, Cin, H, W).

        Returns
        -------
        torch.Tensor
            Autoregressive predictions with shape (B, Tout, Cin, H, W).
        """
        if x.dim() != 5:
            raise ValueError(f"Expected a 5D tensor (B, Tin, Cin, H, W). Got shape: {tuple(x.shape)}")

        b, tin, cin, h, w = x.shape
        if tin != self.input_timesteps or cin != self.input_channels:
            raise ValueError(
                f"Expected Tin={self.input_timesteps}, Cin={self.input_channels}, "
                f"but got Tin={tin}, Cin={cin}."
            )

        context = x  # (B, Tin, Cin, H, W)
        preds: List[torch.Tensor] = []

        for _ in range(self.output_timesteps):
            context_flat = context.reshape(b, tin * cin, h, w)
            next_frame = self.unet(context_flat)  # (B, Cin, H, W)
            next_frame = next_frame.unsqueeze(1)  # (B, 1, Cin, H, W)
            preds.append(next_frame)

            # slide window: drop oldest, append latest prediction
            context = torch.cat([context[:, 1:, ...], next_frame], dim=1)

        y = torch.cat(preds, dim=1)  # (B, Tout, Cin, H, W)
        return self._apply_nonnegativity(y)