"""
Lagrangian persistence (optical flow extrapolation) baseline.

Estimates a motion field from the last input frames with pysteps and advects the last
observed frame along it. This is the classic operational nowcasting reference.
Requires the optional dependency ``pysteps`` (``pip install ainpp-pb-latam[baselines]``).

Note: when evaluated on patches, precipitation advected in from outside the patch is
unknown and pixels advected from outside are set to 0; evaluate on large patches or the
full domain to reduce this border effect.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class OpticalFlowForecaster(nn.Module):
    """
    Lagrangian persistence using a pysteps motion method and semi-Lagrangian extrapolation.

    Args:
        output_timesteps (int): Number of forecast frames (Tout).
        motion_method (str): pysteps motion method name (e.g. 'LK', 'DARTS', 'proesmans').
        n_motion_frames (int): Number of most recent input frames used to estimate motion.
        log_transform (bool): Estimate motion on log1p(rain) for better feature tracking.
    """

    def __init__(
        self,
        output_timesteps: int = 6,
        motion_method: str = "LK",
        n_motion_frames: int = 3,
        log_transform: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        try:
            from pysteps import motion, nowcasts
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise ImportError(
                "OpticalFlowForecaster requires pysteps: pip install 'ainpp-pb-latam[baselines]'"
            ) from exc

        self.output_timesteps = int(output_timesteps)
        self.n_motion_frames = int(n_motion_frames)
        self.log_transform = log_transform
        self._motion = motion.get_method(motion_method)
        self._extrapolate = nowcasts.get_method("extrapolation")

    def _forecast_sample(self, frames: np.ndarray) -> np.ndarray:
        """frames: (Tin, H, W) in mm/h -> (Tout, H, W) in mm/h."""
        recent = frames[-self.n_motion_frames :]
        motion_input = np.log1p(recent) if self.log_transform else recent
        velocity = self._motion(motion_input)
        forecast = self._extrapolate(frames[-1], velocity, self.output_timesteps)
        # Pixels advected from outside the domain are NaN in pysteps
        return np.nan_to_num(np.asarray(forecast), nan=0.0).clip(min=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, Tin, 1, H, W).

        Returns:
            torch.Tensor: Forecast of shape (B, Tout, 1, H, W).
        """
        if x.dim() != 5 or x.shape[2] != 1:
            raise ValueError(f"Expected (B, Tin, 1, H, W). Got {tuple(x.shape)}")

        frames = x[:, :, 0].detach().cpu().numpy().astype(np.float64)
        out = np.stack([self._forecast_sample(sample) for sample in frames])
        return torch.from_numpy(out[:, :, None]).to(device=x.device, dtype=x.dtype)
