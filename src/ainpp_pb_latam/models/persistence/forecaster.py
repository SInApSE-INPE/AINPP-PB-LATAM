"""
Eulerian persistence baseline.

Repeats the last observed input frame for every lead time. It is the minimum reference
any nowcasting model must beat and the default reference for skill scores.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PersistenceForecaster(nn.Module):
    """
    Eulerian persistence: ``y(t0 + k) = x(t0)`` for every lead time ``k``.

    Args:
        output_timesteps (int): Number of forecast frames (Tout).
    """

    def __init__(self, output_timesteps: int = 6, **_: object) -> None:
        super().__init__()
        self.output_timesteps = int(output_timesteps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, Tin, C, H, W).

        Returns:
            torch.Tensor: Forecast of shape (B, Tout, C, H, W).
        """
        if x.dim() != 5:
            raise ValueError(f"Expected a 5D tensor (B, Tin, C, H, W). Got {tuple(x.shape)}")
        last = x[:, -1:]
        return last.expand(-1, self.output_timesteps, -1, -1, -1).clone()
