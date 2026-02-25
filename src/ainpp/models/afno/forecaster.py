"""
Adaptive Fourier Neural Operator (AFNO) Module for Precipitation Nowcasting.

This module implements the AFNO architecture adapted for 2D spatiotemporal forecasting.
It treats time steps as channels during the processing and reconstructs the temporal
dimension at the head.
"""

import logging
from typing import Tuple, Optional

import torch
import torch.nn as nn

# Assuming the block is available in your project structure
from ainpp.models.afno.blocks import AFNOBlock

logger = logging.getLogger(__name__)

class AFNO2D(nn.Module):
    """
    AFNO model for 2D Precipitation Nowcasting. 
    Propused by "Adaptive Fourier Neural Operator for Efficient Spatio-Temporal 
    Forecasting" (Guibas et al., NeurIPS 2021).

    This architecture uses patch embeddings and spectral processing in the Fourier domain
    to capture global dependencies efficiently.

    Attributes:
        patch_embed (nn.Conv2d): Convolutional layer to create patch embeddings.
        pos_embed (nn.Parameter): Learnable positional embeddings.
        blocks (nn.ModuleList): Stack of AFNO Transformer blocks.
        head (nn.ConvTranspose2d): Reconstruction head (Upsampling).
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (880, 970),
        input_timesteps: int = 6,
        input_channels: int = 1,
        output_timesteps: int = 6,
        output_channels: int = 1,
        embed_dim: int = 256,
        depth: int = 8,
        patch_size: int = 10,
        num_blocks: int = 8,
    ):
        """
        Args:
            img_size (Tuple[int, int]): Dimensions of the input grid (H, W).
            input_timesteps (int): Number of input time frames (Tin).
            input_channels (int): Number of channels per input frame. Defaults to 1.
            output_timesteps (int): Number of output time frames (Tout).
            output_channels (int): Number of channels per output frame. Defaults to 1.
            embed_dim (int): Dimension of the embedding space.
            depth (int): Number of AFNO layers/blocks.
            patch_size (int): Size of the patches (must divide H and W).
            num_blocks (int): Number of frequency blocks in the AFNO layer.
        """
        super().__init__()
        logger.info("Initializing AFNO2D forecaster.")
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_timesteps = input_timesteps
        self.input_channels = input_channels
        self.output_timesteps = output_timesteps
        self.output_channels = output_channels
        
        # Grid dimensions in patch space
        self.h = self.img_size[0] // self.patch_size
        self.w = self.img_size[1] // self.patch_size
        
        # 1. Patch Embedding
        # We flatten Time and Channels into a single dimension for the Conv2d
        # Input: (B, Tin*Cin, H, W) -> Output: (B, embed_dim, h, w)
        # Using Conv2d is generally faster than manual unfolding
        self.patch_embed = nn.Conv2d(
            in_channels=input_timesteps * input_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 2. Positional Embedding
        # Scaled by 0.02 for better initialization stability
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.h, self.w, embed_dim) * 0.02
        )
        
        self.dropout = nn.Dropout(0.1)
        
        # 3. AFNO Blocks Stack
        self.blocks = nn.ModuleList([
            AFNOBlock(
                dim=embed_dim, 
                h=self.h, 
                w=self.w, 
                num_blocks=num_blocks
            )
            for _ in range(depth)
        ])
        
        # 4. Reconstruction Head
        # Upsamples from patch space back to pixel space
        # Output channels = Tout * Cout
        self.head = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=output_timesteps * output_channels,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, Tin, Cin, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, Tout, Cout, H, W).
        """
        # x shape: [B, Tin, Cin, H, W]
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B, T, C, H, W), got shape {x.shape}")
            
        B, T, C, H, W = x.shape
        
        # Flatten Temporal and Channel dimensions
        # [B, T, C, H, W] -> [B, T*C, H, W]
        x = x.reshape(B, T * C, H, W)
        
        # 1. Patch Embedding
        x = self.patch_embed(x)   # [B, embed_dim, h, w]
        x = x.permute(0, 2, 3, 1) # [B, h, w, embed_dim] (Channels last for Transformer)
        
        # 2. Positional Embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # 3. AFNO Transformer Blocks
        for blk in self.blocks:
            x = blk(x)
            
        # 4. Reconstruction
        x = x.permute(0, 3, 1, 2) # Back to [B, embed_dim, h, w]
        x = self.head(x)          # [B, Tout*Cout, H, W]

        # Reshape back to 5D: [B, Tout, Cout, H, W]
        # x = x.unsqueeze(2)
        x = x.reshape(B, self.output_timesteps, self.output_channels, H, W)
        
        return x