"""
Adaptive Fourier Neural Operator (AFNO) Block Module.

This module implements the core building block of the AFNO architecture,
which performs mixing in the frequency domain using FFTs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AFNOBlock(nn.Module):
    """
    Adaptive Fourier Neural Operator (AFNO) Block.

    This block performs token mixing in the Fourier domain. It is designed to be
    computationally efficient by dividing the channel dimension into blocks and
    performing operations on a subset of frequencies.

    Attributes:
        w1 (nn.Parameter): Complex weights for frequency mixing (stored as real view).
        b1 (nn.Parameter): Complex bias for frequency mixing (stored as real view).
        norm1 (nn.LayerNorm): Layer normalization before FFT.
        norm2 (nn.LayerNorm): Layer normalization before MLP.
        mlp (nn.Sequential): Multi-layer perceptron for channel mixing.
    """

    def __init__(
        self, dim: int, h: int, w: int, num_blocks: int = 8, sparsity_threshold: float = 0.01
    ):
        """
        Args:
            dim (int): Input channel dimension. Must be divisible by num_blocks.
            h (int): Height of the input.
            w (int): Width of the input.
            num_blocks (int): Number of blocks to divide the channel dimension into.
                This controls the rank of the spectral transformation.
            sparsity_threshold (float): Threshold for soft-thresholding in the frequency domain.
                (Note: In this implementation, ReLU is used as the non-linearity).
        """
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w
        self.num_blocks = num_blocks

        assert dim % num_blocks == 0, f"dim {dim} must be divisible by num_blocks {num_blocks}"
        self.block_size = dim // num_blocks
        self.scale = 0.02

        # ============================================================
        # AMP / GRADSCALER FIX
        # ============================================================
        # PyTorch's AMP GradScaler (float16) often crashes with complex parameters (cfloat).
        # WORKAROUND: Initialize as complex, but store as a REAL view (float32).
        # We convert back to complex on-the-fly during the forward pass.

        # 1. Weights (w1)
        w1_complex = self.scale * torch.randn(
            h, w // 2 + 1, num_blocks, self.block_size, self.block_size, dtype=torch.cfloat
        )
        self.w1 = nn.Parameter(torch.view_as_real(w1_complex))

        # 2. Bias (b1)
        b1_complex = self.scale * torch.randn(
            h, w // 2 + 1, num_blocks, self.block_size, dtype=torch.cfloat
        )
        self.b1 = nn.Parameter(torch.view_as_real(b1_complex))

        # Normalization and MLP
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim), nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AFNO Block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, H, W, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, H, W, C).
        """
        B, H, W, C = x.shape
        residual = x

        x = self.norm1(x)

        # ============================================================
        # SPECTRAL PROCESSING
        # ============================================================
        # We disable autocast here to ensure FFT operations run in FP32.
        # FFT in FP16 can be unstable or overflow easily.
        with torch.amp.autocast("cuda", enabled=False):
            # Ensure input is float32 for the FFT
            x_fp32 = x.float()

            # 1. 2D FFT
            # Result shape: (B, H, W//2 + 1, C) (Complex)
            x_fft = torch.fft.rfft2(x_fp32, dim=(1, 2), norm="ortho")

            # Reshape for block multiplication
            x_fft = x_fft.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

            # 2. Recover Complex Weights (On-the-fly)
            # The optimizer updates self.w1 (Real), but we view it as Complex here.
            w1_c = torch.view_as_complex(self.w1)
            b1_c = torch.view_as_complex(self.b1)

            # 3. Block Multiplication (Spectral Mixing)
            # Einsum: Batch, Height, Width, NumBlocks, BlockIn -> Batch, Height, Width, NumBlocks, BlockOut
            o1 = torch.einsum("b h w n i, h w n i o -> b h w n o", x_fft, w1_c) + b1_c

            # 4. Non-linearity (Soft-Thresholding equivalent)
            o1 = F.relu(o1.real) + 1j * F.relu(o1.imag)

            # 5. Reshape and IFFT
            o1 = o1.reshape(B, H, W // 2 + 1, C)

            # Inverse FFT to return to spatial domain
            x = torch.fft.irfft2(o1, s=(H, W), dim=(1, 2), norm="ortho")

        # Final connection: Residual + MLP(Norm(Spatial_Mixing))
        return x + residual + self.mlp(self.norm2(x))
