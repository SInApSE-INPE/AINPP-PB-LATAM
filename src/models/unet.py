from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# -----------------------------
# Utilities
# -----------------------------
def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _configure_logging(level: int = logging.INFO) -> None:
    """Configure a basic console logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


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
            raise ValueError("kernel_size must be a positive odd integer to preserve spatial dimensions via padding.")

        mid_channels = out_channels if mid_channels is None else mid_channels
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
            norm_layer(mid_channels),
            activation(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
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
            self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels, kernel_size=kernel_size)

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


# -----------------------------
# U-Net core
# -----------------------------
@dataclass(frozen=True)
class UNetConfig:
    """
    Configuration for a flexible 2D U-Net.

    Attributes
    ----------
    in_channels:
        Number of input channels.
    out_channels:
        Number of output channels.
    features:
        Channel widths per level (encoder). Must have length >= 2.
    kernel_size:
        Kernel size for DoubleConv blocks (odd integer recommended).
    bilinear:
        If True, uses bilinear upsampling. Otherwise uses transposed convolutions.
    """

    in_channels: int
    out_channels: int
    features: Sequence[int] = (64, 128, 256, 512, 1024)
    kernel_size: int = 3
    bilinear: bool = True


class UNet2D(nn.Module):
    """
    A configurable 2D U-Net with encoder/decoder and skip connections.

    Input shape:  (B, Cin, H, W)
    Output shape: (B, Cout, H, W)
    """

    def __init__(self, cfg: UNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._validate_cfg()

        features = list(cfg.features)

        # When using bilinear upsampling, a common adjustment is to reduce the bottleneck.
        # This keeps parameter count closer to the transposed-conv variant.
        if cfg.bilinear:
            features[-1] = features[-1] // 2

        self.stem = DoubleConv(cfg.in_channels, features[0], kernel_size=cfg.kernel_size)

        # Encoder
        self.down: nn.ModuleList[nn.Module] = nn.ModuleList()
        for i in range(len(features) - 1):
            self.down.append(DownBlock(features[i], features[i + 1], kernel_size=cfg.kernel_size))

        # Decoder
        self.up: nn.ModuleList[nn.Module] = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.up.append(
                UpBlock(
                    in_channels=features[i],
                    skip_channels=features[i - 1],
                    out_channels=features[i - 1],
                    kernel_size=cfg.kernel_size,
                    bilinear=cfg.bilinear,
                )
            )

        self.head = nn.Conv2d(features[0], cfg.out_channels, kernel_size=1)

    def _validate_cfg(self) -> None:
        if self.cfg.in_channels <= 0:
            raise ValueError("in_channels must be > 0.")
        if self.cfg.out_channels <= 0:
            raise ValueError("out_channels must be > 0.")
        if len(self.cfg.features) < 2:
            raise ValueError("features must have length >= 2.")
        if any(f <= 0 for f in self.cfg.features):
            raise ValueError("All feature sizes must be > 0.")
        if self.cfg.kernel_size <= 0 or self.cfg.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

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


# -----------------------------
# Forecasting wrappers
# -----------------------------
@dataclass(frozen=True)
class MultiHorizonConfig:
    """
    Direct multi-horizon forecasting using a U-Net over a channel-stacked temporal window.

    The model consumes a sequence (Tin) and predicts all future frames (Tout) in one forward pass.

    Input:  (B, Tin, Cin, H, W)
    Output: (B, Tout, Cout, H, W)
    """

    input_timesteps: int = 12
    input_channels: int = 1
    output_timesteps: int = 6
    output_channels: int = 1
    features: Sequence[int] = (64, 128, 256, 512, 1024)
    kernel_size: int = 3
    bilinear: bool = True
    nonnegativity: str = "relu"  # "relu" | "softplus" | "none"


class UNetMultiHorizon(nn.Module):
    """
    Direct (multi-horizon) precipitation forecaster.

    Strategy
    --------
    - Stack the temporal dimension into channels: Tin * Cin channels.
    - Use a 2D U-Net to map to Tout * Cout channels.
    - Reshape back to (B, Tout, Cout, H, W).

    Notes
    -----
    Enforcing non-negativity:
      - "relu" is simple and fast, but can create dead zones.
      - "softplus" is smooth and strictly positive.
      - "none" disables the constraint.
    """

    def __init__(self, cfg: MultiHorizonConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._validate_cfg()

        unet_cfg = UNetConfig(
            in_channels=cfg.input_timesteps * cfg.input_channels,
            out_channels=cfg.output_timesteps * cfg.output_channels,
            features=cfg.features,
            kernel_size=cfg.kernel_size,
            bilinear=cfg.bilinear,
        )
        self.unet = UNet2D(unet_cfg)

    def _validate_cfg(self) -> None:
        if self.cfg.input_timesteps <= 0:
            raise ValueError("input_timesteps must be > 0.")
        if self.cfg.output_timesteps <= 0:
            raise ValueError("output_timesteps must be > 0.")
        if self.cfg.input_channels <= 0:
            raise ValueError("input_channels must be > 0.")
        if self.cfg.output_channels <= 0:
            raise ValueError("output_channels must be > 0.")
        if self.cfg.nonnegativity not in {"relu", "softplus", "none"}:
            raise ValueError("nonnegativity must be one of: {'relu', 'softplus', 'none'}.")

    def _apply_nonnegativity(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.nonnegativity == "relu":
            return F.relu(x)
        if self.cfg.nonnegativity == "softplus":
            return F.softplus(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Input tensor with shape (B, Tin, Cin, H, W).

        Returns
        -------
        torch.Tensor
            Prediction tensor with shape (B, Tout, Cout, H, W).
        """
        if x.dim() != 5:
            raise ValueError(f"Expected a 5D tensor (B, Tin, Cin, H, W). Got shape: {tuple(x.shape)}")

        b, tin, cin, h, w = x.shape
        if tin != self.cfg.input_timesteps or cin != self.cfg.input_channels:
            raise ValueError(
                f"Expected Tin={self.cfg.input_timesteps}, Cin={self.cfg.input_channels}, "
                f"but got Tin={tin}, Cin={cin}."
            )

        x_flat = x.reshape(b, tin * cin, h, w)
        y_flat = self.unet(x_flat)
        y = y_flat.reshape(b, self.cfg.output_timesteps, self.cfg.output_channels, h, w)
        return self._apply_nonnegativity(y)


@dataclass(frozen=True)
class AutoRegressiveConfig:
    """
    Autoregressive multi-step forecasting with a single-step U-Net.

    Input:  (B, Tin, Cin, H, W)
    Output: (B, Tout, Cin, H, W)

    Notes
    -----
    - This version predicts one frame at a time and feeds it back into the context window.
    - Output channels are fixed to input_channels for consistency (typical in autoregression).
    """

    input_timesteps: int = 12
    input_channels: int = 1
    output_timesteps: int = 6
    features: Sequence[int] = (64, 128, 256, 512, 1024)
    kernel_size: int = 3
    bilinear: bool = True
    nonnegativity: str = "relu"  # "relu" | "softplus" | "none"


class UNetAutoRegressive(nn.Module):
    """
    Autoregressive precipitation forecaster (one-step-ahead repeated Tout times).

    The model uses a single-step U-Net that predicts exactly one future frame per iteration.
    """

    def __init__(self, cfg: AutoRegressiveConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._validate_cfg()

        unet_cfg = UNetConfig(
            in_channels=cfg.input_timesteps * cfg.input_channels,
            out_channels=cfg.input_channels,  # one step: predict Cin channels
            features=cfg.features,
            kernel_size=cfg.kernel_size,
            bilinear=cfg.bilinear,
        )
        self.unet = UNet2D(unet_cfg)

    def _validate_cfg(self) -> None:
        if self.cfg.input_timesteps <= 0:
            raise ValueError("input_timesteps must be > 0.")
        if self.cfg.output_timesteps <= 0:
            raise ValueError("output_timesteps must be > 0.")
        if self.cfg.input_channels <= 0:
            raise ValueError("input_channels must be > 0.")
        if self.cfg.nonnegativity not in {"relu", "softplus", "none"}:
            raise ValueError("nonnegativity must be one of: {'relu', 'softplus', 'none'}.")

    def _apply_nonnegativity(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.nonnegativity == "relu":
            return F.relu(x)
        if self.cfg.nonnegativity == "softplus":
            return F.softplus(x)
        return x

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
        if tin != self.cfg.input_timesteps or cin != self.cfg.input_channels:
            raise ValueError(
                f"Expected Tin={self.cfg.input_timesteps}, Cin={self.cfg.input_channels}, "
                f"but got Tin={tin}, Cin={cin}."
            )

        context = x  # (B, Tin, Cin, H, W)
        preds: List[torch.Tensor] = []

        for _ in range(self.cfg.output_timesteps):
            context_flat = context.reshape(b, tin * cin, h, w)
            next_frame = self.unet(context_flat)  # (B, Cin, H, W)
            next_frame = next_frame.unsqueeze(1)  # (B, 1, Cin, H, W)
            preds.append(next_frame)

            # slide window: drop oldest, append latest prediction
            context = torch.cat([context[:, 1:, ...], next_frame], dim=1)

        y = torch.cat(preds, dim=1)  # (B, Tout, Cin, H, W)
        return self._apply_nonnegativity(y)


# -----------------------------
# Smoke test / example usage
# -----------------------------
def _smoke_test(
    device: torch.device,
    input_shape: Tuple[int, int, int, int, int] = (2, 12, 1, 300, 360),
) -> None:
    b, tin, cin, h, w = input_shape
    x = torch.randn(*input_shape, device=device)

    direct_cfg = MultiHorizonConfig(
        input_timesteps=tin,
        input_channels=cin,
        output_timesteps=6,
        output_channels=1,
        features=(64, 128, 256, 512, 1024),
        kernel_size=3,
        bilinear=True,
        nonnegativity="relu",
    )
    model_direct = UNetMultiHorizon(direct_cfg).to(device)

    ar_cfg = AutoRegressiveConfig(
        input_timesteps=tin,
        input_channels=cin,
        output_timesteps=6,
        features=(64, 128, 256, 512, 1024),
        kernel_size=3,
        bilinear=True,
        nonnegativity="relu",
    )
    model_ar = UNetAutoRegressive(ar_cfg).to(device)

    with torch.no_grad():
        y_direct = model_direct(x)
        y_ar = model_ar(x)

    logger.info("Direct model params: %s", f"{count_trainable_parameters(model_direct):,}")
    logger.info("AutoReg model params: %s", f"{count_trainable_parameters(model_ar):,}")
    logger.info("Input shape:  %s", tuple(x.shape))
    logger.info("Direct out:   %s", tuple(y_direct.shape))
    logger.info("AutoReg out:  %s", tuple(y_ar.shape))


if __name__ == "__main__":
    _configure_logging(logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    _smoke_test(device=device)
