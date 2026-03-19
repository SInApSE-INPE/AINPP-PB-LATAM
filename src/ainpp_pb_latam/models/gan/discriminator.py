"""
Discriminator modules for Spatiotemporal GANs.

This module implements a 3D PatchGAN Discriminator, which is suitable for
video generation tasks like precipitation nowcasting. It judges whether a
sequence of frames is real or fake at the patch level.
"""

import logging
from functools import partial
from typing import List, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PatchDiscriminator3D(nn.Module):
    """
    3D PatchGAN Discriminator.

    This model takes a sequence of frames (Input History + Prediction/Target)
    and outputs a grid of validity scores. Each pixel in the output grid
    represents the 'realness' of a spatiotemporal patch in the input.

    Architecture:
        Series of Conv3d layers with LeakyReLU and Batch/Instance Norm.
        No fully connected layers at the end (fully convolutional).
    """

    def __init__(
        self, input_channels: int = 1, ndf: int = 64, n_layers: int = 3, norm_type: str = "instance"
    ):
        """
        Args:
            input_channels (int): Number of channels (usually Input Channels + Output Channels for cGAN).
            ndf (int): Number of Discriminator Filters in the first layer.
            n_layers (int): Number of downsampling layers.
            norm_type (str): 'instance' or 'batch'. Type of normalization layer to use.
        """
        super().__init__()

        if norm_type == "batch":
            norm_layer = nn.BatchNorm3d
        else:
            norm_layer = partial(nn.InstanceNorm3d, affine=True)

        # 1. Input Layer (No Norm)
        # Kernel: (Time=4, H=4, W=4), Stride: (2, 2, 2)
        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(input_channels, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        # 2. Hidden Layers
        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=False,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        # 3. Intermediate Layer
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=False,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        # 4. Output Layer
        # Maps to a 1-channel prediction map (Real/Fake)
        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input sequence (B, C, T, H, W).
                              Note: Usually concat(History, Prediction, dim=Channel).

        Returns:
            torch.Tensor: Validity map (B, 1, T', H', W').
        """
        return self.model(x)
