"""
Forecaster modules based on U-Net architecture.

This module implements two forecasting strategies using a standard 2D U-Net backbone:
1.  UNetMultiHorizon: Direct strategy (predicts all timesteps at once).
2.  UNetAutoRegressive: Recursive strategy (predicts one step at a time, feeding it back).
"""

from __future__ import annotations

import logging
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Assumes UNet2D is defined in the backbone module
from ainpp_pb_latam.models.unet.backbone import UNet2D

logger = logging.getLogger(__name__)


class BaseForecaster(nn.Module):
    """
    Base class for forecasters providing common utility methods.
    """

    def _apply_nonnegativity(self, x: torch.Tensor, mode: str = "relu") -> torch.Tensor:
        """
        Applies a non-linearity to enforce non-negative precipitation values.

        Args:
            x (torch.Tensor): Input tensor.
            mode (str): Activation mode ('relu', 'softplus', or 'none').

        Returns:
            torch.Tensor: Tensor with non-negative values.
        """
        if mode == "relu":
            return F.relu(x)
        if mode == "softplus":
            return F.softplus(x)
        return x


class UNetMultiHorizon(BaseForecaster):
    """
    Direct Multi-Horizon Forecaster using a 2D U-Net backbone.

    This model stacks input timesteps into the channel dimension, passes them
    through a 2D U-Net, and outputs all future timesteps simultaneously
    by predicting (Output_Timesteps * Output_Channels) channels.

    Shape Logic:
        Input: (B, Tin, C, H, W) -> Flat: (B, Tin*C, H, W)
        Output Flat: (B, Tout*C, H, W) -> Reshaped: (B, Tout, C, H, W)
    """

    def __init__(
        self,
        input_timesteps: int,
        input_channels: int,
        output_timesteps: int,
        output_channels: int = 1,
        features: Sequence[int] = (64, 128, 256, 512),
        kernel_size: int = 3,
        bilinear: bool = True,
        nonnegativity: str = "relu",
    ):
        """
        Args:
            input_timesteps (int): Number of input time frames.
            input_channels (int): Number of channels per frame.
            output_timesteps (int): Number of time frames to predict.
            output_channels (int): Number of channels per predicted frame.
            features (Sequence[int]): List of channel depths for U-Net encoder levels.
            kernel_size (int): Convolution kernel size.
            bilinear (bool): If True, use bilinear upsampling; otherwise transposed conv.
            nonnegativity (str): Strategy for output activation ('relu' or 'softplus').
        """
        super().__init__()
        logger.info("Initializing UNetMultiHorizon forecaster.")
        self.output_timesteps = output_timesteps
        self.out_ch = output_channels
        self.nonnegativity = nonnegativity

        # In Direct strategy, the U-Net maps (Tin * Cin) -> (Tout * Cout)
        self.unet = UNet2D(
            in_channels=input_timesteps * input_channels,
            out_channels=output_timesteps * output_channels,
            features=list(features),
            kernel_size=kernel_size,
            bilinear=bilinear,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, Tin, Cin, H, W).

        Returns:
            torch.Tensor: Predicted tensor of shape (B, Tout, Cout, H, W).
        """
        b, tin, cin, h, w = x.shape

        # Flatten temporal dimension into channels
        x_flat = x.reshape(b, tin * cin, h, w)

        # Forward pass through U-Net
        y_flat = self.unet(x_flat)

        # Reshape back to spatiotemporal format
        y = y_flat.reshape(b, self.output_timesteps, self.out_ch, h, w)

        return self._apply_nonnegativity(y, self.nonnegativity)


class UNetAutoRegressive(BaseForecaster):
    """
    Auto-Regressive Forecaster using a 2D U-Net backbone.

    This model predicts one frame at a time. The prediction is fed back
    into the input window (sliding window) to predict the subsequent frame.

    Mechanism:
        1. Input window (t-N ... t) -> Predict (t+1)
        2. New window (t-N+1 ... t, t+1) -> Predict (t+2)
        3. Repeat until Output_Timesteps is reached.
    """

    def __init__(
        self,
        input_timesteps: int,
        input_channels: int,
        output_timesteps: int,
        features: Sequence[int] = (64, 128, 256, 512),
        kernel_size: int = 3,
        bilinear: bool = True,
        nonnegativity: str = "relu",
    ):
        """
        Args:
            input_timesteps (int): Size of the sliding window (Tin).
            input_channels (int): Number of channels per frame.
            output_timesteps (int): Total horizon to predict (Tout).
            features (Sequence[int]): List of channel depths for U-Net levels.
            kernel_size (int): Convolution kernel size.
            bilinear (bool): Upsampling mode.
            nonnegativity (str): Output activation ('relu', 'softplus').
        """
        super().__init__()
        logger.info("Initializing UNetAutoRegressive forecaster.")
        self.input_timesteps = input_timesteps
        self.input_channels = input_channels
        self.output_timesteps = output_timesteps
        self.features = list(features)
        self.kernel_size = kernel_size
        self.bilinear = bilinear
        self.nonnegativity = nonnegativity

        # In AR strategy, the U-Net maps (Tin * Cin) -> (1 * Cin)
        self.unet = UNet2D(
            in_channels=self.input_timesteps * self.input_channels,
            out_channels=self.input_channels,
            features=self.features,
            kernel_size=self.kernel_size,
            bilinear=self.bilinear,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs auto-regressive rollout.

        Args:
            x (torch.Tensor): Input tensor of shape (B, Tin, Cin, H, W).

        Returns:
            torch.Tensor: Recursive predictions of shape (B, Tout, Cin, H, W).

        Raises:
            ValueError: If input shape dimensions do not match the configuration.
        """
        if x.dim() != 5:
            raise ValueError(
                f"Expected a 5D tensor (B, Tin, Cin, H, W). Got shape: {tuple(x.shape)}"
            )

        b, tin, cin, h, w = x.shape
        if tin != self.input_timesteps or cin != self.input_channels:
            raise ValueError(
                f"Expected Tin={self.input_timesteps}, Cin={self.input_channels}, "
                f"but got Tin={tin}, Cin={cin}."
            )

        context = x  # Current sliding window: (B, Tin, Cin, H, W)
        preds: List[torch.Tensor] = []

        # Auto-regressive loop
        for _ in range(self.output_timesteps):
            # Flatten context for the 2D backbone
            context_flat = context.reshape(b, tin * cin, h, w)

            # Predict next frame (B, Cin, H, W)
            next_frame = self.unet(context_flat)

            # Add temporal dimension: (B, 1, Cin, H, W)
            next_frame = next_frame.unsqueeze(1)
            preds.append(next_frame)

            # Update Sliding Window:
            # 1. Drop oldest frame: context[:, 1:, ...]
            # 2. Append new prediction: next_frame
            # 3. Concatenate along time dimension
            context = torch.cat([context[:, 1:, ...], next_frame], dim=1)

        # Concatenate all predictions along time dimension
        y = torch.cat(preds, dim=1)  # (B, Tout, Cin, H, W)

        return self._apply_nonnegativity(y, self.nonnegativity)
