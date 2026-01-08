"""
Loss functions module for precipitation nowcasting.

This module implements:
1.  Pixel-wise losses (Weighted MSE/MAE, LogCosh).
2.  Structural losses (SSIM).
3.  Hybrid containers to combine multiple losses via configuration.
"""

import math
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class WeightedMSELoss(nn.Module):
    """
    Mean Squared Error with dynamic weighting based on target intensity.

    Precipitation data is heavily imbalanced (lots of zeros, few heavy rain pixels).
    Standard MSE tends to suppress high-intensity values. This loss applies
    a weight mask: weight = 1 + alpha * target.

    Formula:
        L = mean( (pred - target)^2 * (1 + alpha * target) )
    """

    def __init__(self, alpha: float = 1.0, threshold: float = 0.0):
        """
        Args:
            alpha (float): Scaling factor for heavy rain. 
                           Higher values prioritize heavy rain. Defaults to 1.0.
            threshold (float): Only apply weight for values > threshold. Defaults to 0.0.
        """
        super().__init__()
        self.alpha = alpha
        self.threshold = threshold

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): Predictions (B, ...).
            target (torch.Tensor): Ground truth (B, ...).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        mse = (input - target) ** 2
        
        # Create weight map based on target intensity
        # If target > threshold, weight grows linearly. Else weight is 1.
        if self.alpha > 0:
            weights = torch.ones_like(target)
            mask = target > self.threshold
            weights[mask] = 1.0 + self.alpha * target[mask]
            mse = mse * weights

        return mse.mean()


class LogCoshLoss(nn.Module):
    """
    Log-Cosh Loss.

    A smooth approximation of Huber Loss / MAE.
    - Behaves like MSE (L2) for small errors (x < 1).
    - Behaves like MAE (L1) for large errors (x > 1).
    - Fully differentiable (unlike MAE at 0).
    
    Formula:
        L = log(cosh(pred - target))
    """
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.log(torch.cosh(input - target))
        return loss.mean()


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) Loss.

    Designed to penalize structural differences rather than pixel-wise errors.
    Crucial for preventing the "blurring" effect common in MSE-trained models.

    Note: This implementation handles 5D tensors (B, T, C, H, W) by flattening 
    Batch and Time dimensions during computation.
    """

    def __init__(self, window_size: int = 11, in_channels: int = 1):
        """
        Args:
            window_size (int): Size of the Gaussian window (kernel). Defaults to 11.
            in_channels (int): Number of channels in the images. Defaults to 1.
        """
        super().__init__()
        self.window_size = window_size
        self.channel = in_channels
        self.window = self._create_window(window_size, self.channel)

    def _gaussian(self, window_size: int, sigma: float) -> torch.Tensor:
        gauss = torch.Tensor([
            math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(
        self, 
        img1: torch.Tensor, 
        img2: torch.Tensor, 
        window: torch.Tensor, 
        window_size: int, 
        channel: int
    ) -> torch.Tensor:
        
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): Shape (B, T, C, H, W) or (B, C, H, W).
            target (torch.Tensor): Shape (B, T, C, H, W) or (B, C, H, W).
        """
        # Handle 5D Input (Video/Sequence) -> Flatten to 4D for 2D Conv
        is_5d = input.dim() == 5
        if is_5d:
            b, t, c, h, w = input.shape
            input = input.reshape(b * t, c, h, w)
            target = target.reshape(b * t, c, h, w)

        (_, channel, _, _) = input.size()

        # Update window if device/type/channels changed
        if channel == self.channel and self.window.data.type() == input.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            window = window.type_as(input).to(input.device)
            self.window = window
            self.channel = channel
            
        # Ensure window is on correct device
        if self.window.device != input.device:
             self.window = self.window.to(input.device)

        # Return 1 - SSIM (because we want to minimize loss)
        return 1.0 - self._ssim(input, target, self.window, self.window_size, channel)


class HybridLoss(nn.Module):
    """
    Weighted combination of multiple loss functions.
    
    Designed to be instantiated via Hydra, allowing the user to mix-and-match
    losses without changing code.
    
    Example:
        Total = 1.0 * MSE + 0.1 * SSIM
    """

    def __init__(self, losses: Sequence[nn.Module], weights: Sequence[float]):
        """
        Args:
            losses (Sequence[nn.Module]): List of instantiated loss modules.
            weights (Sequence[float]): List of scalar weights for each loss.
        """
        super().__init__()
        assert len(losses) == len(weights), "Number of losses and weights must match."
        
        self.losses = nn.ModuleList(losses)
        self.weights = weights

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for loss_fn, w in zip(self.losses, self.weights):
            total_loss += w * loss_fn(input, target)
        return total_loss
    
class HuberLoss(nn.Module):
    """
    Huber Loss wrapper.

    Combines the benefits of MSE (L2) and MAE (L1). It is less sensitive to
    outliers than MSE, which is useful for noisy radar data.

    Formula:
        0.5 * x^2                  if |x| <= delta
        delta * (|x| - 0.5 * delta)  otherwise
    """
    def __init__(self, delta: float = 1.0):
        """
        Args:
            delta (float): Threshold where the loss changes from L2 to L1.
                           Defaults to 1.0.
        """
        super().__init__()
        self.loss_fn = nn.HuberLoss(delta=delta)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(input, target)


class CrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss wrapper.

    Used if the precipitation problem is framed as a classification task
    (e.g., binning precipitation values into classes: None, Light, Heavy).
    """
    def __init__(self, weights: Optional[List[float]] = None):
        """
        Args:
            weights (List[float], optional): Manual rescaling weight given to each
                                            class. Useful for imbalanced classes.
        """
        super().__init__()
        self.weights = torch.tensor(weights) if weights else None
        self.loss_fn = nn.CrossEntropyLoss(weight=self.weights)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: (B, C, H, W) Logits.
            target: (B, H, W) Class indices (LongTensor).
        """
        if self.weights is not None and self.weights.device != input.device:
            self.loss_fn.weight = self.weights.to(input.device)
        return self.loss_fn(input, target)


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for rain/no-rain classification.

    Focuses training on hard examples and down-weights easy negatives
    (e.g., vast areas of no rain).
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, threshold: float = 0.1):
        """
        Args:
            alpha (float): Weighting factor for the positive class (rain).
            gamma (float): Focusing parameter. Higher values focus more on hard examples.
            threshold (float): Threshold to binarize continuous targets (mm/h).
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.threshold = threshold

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Binarize Target based on precipitation threshold
        target_binary = (target > self.threshold).float()
        
        # Assume input are logits, so apply sigmoid
        probs = torch.sigmoid(input)
        
        # Standard Focal Loss Formula
        ce_loss = F.binary_cross_entropy_with_logits(input, target_binary, reduction="none")
        p_t = probs * target_binary + (1 - probs) * (1 - target_binary)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target_binary + (1 - self.alpha) * (1 - target_binary)
            loss = alpha_t * loss

        return loss.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss for regression tasks (converted to binary on-the-fly).

    Optimizes the overlap between predicted rain mask and actual rain mask.
    Crucial for correctly predicting the spatial extent (shape) of storms.
    """
    def __init__(self, threshold: float = 0.1, smooth: float = 1e-6):
        """
        Args:
            threshold (float): Precipitation threshold (mm/h) to define "rain".
            smooth (float): Smoothing factor to avoid division by zero.
        """
        super().__init__()
        self.threshold = threshold
        self.smooth = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Binarize inputs and targets
        # Note: Using sigmoid on input to simulate probability if it's raw regression output
        # For pure regression models, we might strictly cut at threshold.
        # Here we use soft approach for differentiability.
        input_soft = torch.sigmoid(input - self.threshold) 
        target_binary = (target > self.threshold).float()

        intersection = (input_soft * target_binary).sum()
        union = input_soft.sum() + target_binary.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class AdvancedTorrentialLoss(nn.Module):
    """
    Advanced Torrential (AT) Loss.

    A specialized weighted MSE that applies exponential penalties to 
    extreme rainfall events defined by multiple thresholds.
    """
    def __init__(self, thresholds: List[float] = [10.0, 30.0], weights: List[float] = [2.0, 5.0]):
        """
        Args:
            thresholds (List[float]): List of precipitation values (mm/h) defining tiers.
            weights (List[float]): Weights for each tier. Must match len(thresholds).
        """
        super().__init__()
        assert len(thresholds) == len(weights), "Thresholds and weights must have same length"
        self.thresholds = sorted(thresholds)
        self.weights = weights

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = (input - target) ** 2
        weight_map = torch.ones_like(target)

        # Apply weights progressively
        # If target > 10, weight becomes 2.0. If target > 30, weight becomes 5.0.
        for thresh, w in zip(self.thresholds, self.weights):
            mask = target >= thresh
            weight_map[mask] = w

        loss = mse * weight_map
        return loss.mean()


class SpectralLoss(nn.Module):
    """
    Fourier Amplitude and Correlation Loss (FACL).

    Calculates loss in the frequency domain (FFT). This helps the model
    capture global patterns and textures, reducing blurriness.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """
        Args:
            alpha (float): Weight for the Amplitude (Magnitude) term.
            beta (float): Weight for the Phase (Correlation) term.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Flatten batch and time dimensions for 2D FFT
        # Input: (B, T, C, H, W) -> (N, H, W) assuming C=1
        
        if input.dim() == 5:
            b, t, c, h, w = input.shape
            input = input.reshape(-1, h, w)
            target = target.reshape(-1, h, w)
        
        # FFT
        fft_pred = torch.fft.rfft2(input, norm='ortho')
        fft_target = torch.fft.rfft2(target, norm='ortho')

        # 1. Amplitude Loss (Magnitude)
        amp_pred = torch.abs(fft_pred)
        amp_target = torch.abs(fft_target)
        loss_amp = F.mse_loss(amp_pred, amp_target)

        # 2. Phase Loss (Correlation)
        # Using L2 distance in the complex plane as a proxy for phase alignment
        loss_phase = F.mse_loss(torch.view_as_real(fft_pred), torch.view_as_real(fft_target))

        return self.alpha * loss_amp + self.beta * loss_phase


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss (Feature Reconstruction Loss) using VGG16.

    Extracts features from a pre-trained VGG network and computes MSE 
    between feature maps. Forces the model to generate structurally realistic rain.
    """
    def __init__(self, layer_ids: List[int] = [3, 8, 15, 22]):
        """
        Args:
            layer_ids (List[int]): Indices of VGG features to use. 
                                   [3, 8, 15, 22] corresponds roughly to relu1_2, relu2_2...
        """
        super().__init__()
        try:
            # Load VGG16 pretrained on ImageNet
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.vgg_layers = vgg.features
            self.layer_ids = layer_ids
            
            # Freeze VGG parameters (we don't train VGG)
            for param in self.vgg_layers.parameters():
                param.requires_grad = False
            
            self.enabled = True
        except Exception as e:
            print(f"Warning: Could not load VGG weights (No Internet?). Perceptual loss disabled. Error: {e}")
            self.vgg_layers = None
            self.enabled = False

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.vgg_layers is None:
            return F.mse_loss(input, target) # Fallback

        # Handle 5D input
        if input.dim() == 5:
            b, t, c, h, w = input.shape
            input = input.reshape(b * t, c, h, w)
            target = target.reshape(b * t, c, h, w)

        # VGG expects 3 channels (RGB). Precipitation is usually 1 channel.
        # We repeat the channel 3 times.
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        loss = 0.0
        x = input
        y = target
        
        # Normalize input for VGG (ImageNet stats) roughly
        # Assuming input is precipitation, we scale it to look somewhat like an image
        # Or pass raw if already normalized. Here passing raw.
        
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            y = layer(y)
            if i in self.layer_ids:
                loss += F.mse_loss(x, y)
        
        return loss