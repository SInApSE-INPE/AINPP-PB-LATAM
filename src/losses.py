import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math

class WeightedMSELoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def forward(self, input, target, weights=None):
        loss = (input - target) ** 2
        w = weights if weights is not None else self.weights
        if w is not None:
             if isinstance(w, torch.Tensor) and w.device != loss.device:
                w = w.to(loss.device)
             loss *= w
        return loss.mean()

class WeightedMAELoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def forward(self, input, target, weights=None):
        loss = torch.abs(input - target)
        w = weights if weights is not None else self.weights
        if w is not None:
            if isinstance(w, torch.Tensor) and w.device != loss.device:
                w = w.to(loss.device)
            loss *= w
        return loss.mean()

class BalancedMSELoss(nn.Module):
    """
    Balanced MSE Loss as designed for imbalanced regression or precip.
    Typically involves weighting based on target magnitude or frequency.
    Here we implement a simple version that requires passing weights or pre-computed buckets.
    If no weights passed, acts as MSE.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input, target, weights=None):
        # If dynamic weighting based on target value is needed (like assigning higher weight to heavier rainfall):
        # Example: weight = 1 + alpha * target
        # For now, we assume weights are passed externally or just standard MSE if None.
        mse = (input - target) ** 2
        if weights is not None:
            return (mse * weights).mean()
        return mse.mean()

class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) Loss.
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

class MS_SSIMLoss(nn.Module):
    # Placeholder for Multi-Scale SSIM. A full implementation is verbose. 
    # For now, falls back to SSIM or a simplified multi-scale if requested.
    # In a real rigorous setting, pytorch-msssim package is recommended.
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.ssim = SSIMLoss(window_size, size_average)
        
    def forward(self, img1, img2):
        # Naive approximation: calculate SSIM on 2 scales
        loss1 = self.ssim(img1, img2)
        loss2 = self.ssim(F.avg_pool2d(img1, 2), F.avg_pool2d(img2, 2))
        return 0.5 * loss1 + 0.5 * loss2

class PerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[3, 8, 15, 22]): # relu1_2, relu2_2, relu3_3, relu4_3 approx
        super().__init__()
        try:
            vgg = models.vgg16(pretrained=True)
            self.vgg_layers = vgg.features
            self.layer_ids = layer_ids
            self.freeze()
        except Exception as e:
            print(f"Warning: Could not load VGG weights: {e}. Perceptual loss disabled/dummy.")
            self.vgg_layers = None

    def freeze(self):
        if self.vgg_layers:
            for param in self.vgg_layers.parameters():
                param.requires_grad = False

    def forward(self, input, target):
        if self.vgg_layers is None:
            return F.mse_loss(input, target)
            
        # VGG expects 3 channels. If 1 channel, repeat.
        if input.shape[1] == 1:
            input_vgg = input.repeat(1, 3, 1, 1)
            target_vgg = target.repeat(1, 3, 1, 1)
        else:
            input_vgg = input
            target_vgg = target

        loss = 0
        x = input_vgg
        y = target_vgg
        
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            y = layer(y)
            if i in self.layer_ids:
                loss += F.mse_loss(x, y)
        return loss

class FocalLoss(nn.Module):
    # Typically for classification, but can be adapted for regression ("Focal MSE") 
    # or used if the problem is cast as classification (e.g. quantization).
    # Here we assume standard classification (Cross Entropy replacement).
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Assumes inputs are logits, targets are indices
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CRPSLoss(nn.Module):
    # Continuous Ranked Probability Score.
    # For deterministic forecasts, CRPS reduces to MAE. 
    # For probabilistic forecasts (ensembles or distributions), it differs.
    # This implementation assumes Gaussian distribution output (mu, sigma).
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # Check if input has 2 channels (mu, sigma)
        if input.shape[1] == 2:
            mu = input[:, 0:1, :, :]
            sigma = torch.exp(input[:, 1:2, :, :]) # Ensure positive sigma
            
            # CRPS for Gaussian:
            # CRPS(N(mu, sig^2), y) = sig * [1/sqrt(pi) - 2*phi(z) - z*(2*Phi(z)-1)]
            # where z = (y - mu) / sig
            z = (target - mu) / sigma
            pdf = (1. / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * z ** 2)
            cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
            
            # Analytical CRPS for Gaussian
            loss = sigma * (z * (2 * cdf - 1) + 2 * pdf - 1/math.sqrt(math.pi))
            return loss.mean()
        else:
            # Fallback for deterministic: MAE
            return F.l1_loss(input, target)

class HybridLoss(nn.Module):
    def __init__(self, losses, weights):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights

    def forward(self, input, target, **kwargs):
        total_loss = 0
        for loss_fn, w in zip(self.losses, self.weights):
            total_loss += w * loss_fn(input, target, **kwargs)
        return total_loss

def get_loss(config):
    name = config.name.lower()
    
    if name == 'mse':
        return nn.MSELoss()
    elif name == 'mae':
        return nn.L1Loss()
    elif name == 'weighted_mse':
        return WeightedMSELoss()
    elif name == 'weighted_mae':
        return WeightedMAELoss()
    elif name == 'balanced_mse':
        return BalancedMSELoss()
    elif name == 'ssim':
        return SSIMLoss()
    elif name == 'ms_ssim':
        return MS_SSIMLoss()
    elif name == 'perceptual':
        return PerceptualLoss()
    elif name == 'focal':
        return FocalLoss(alpha=config.get('alpha', 0.25), gamma=config.get('gamma', 2.0))
    elif name == 'cross_entropy':
        weights = torch.tensor(config.get('weights')) if 'weights' in config else None
        return nn.CrossEntropyLoss(weight=weights)
    elif name == 'crps':
        return CRPSLoss()
    elif name == 'hybrid':
        # Example hybrid config: 
        # hybrid: [ {name: mse, weight: 1.0}, {name: ssim, weight: 0.1} ]
        # BUT simplest is to defined specific combos or parse a list.
        # For simplicity here, we assume a specific structure or just fail if not fully defined.
        # Let's implement a hardcoded common hybrid (MSE + SSIM) if params generic,
        # or robustly parse sub-configs if needed.
        # Given the prompt, let's allow a simple list of names/weights in config if possible.
        # But for now, let's just support MSE + SSIM as a default 'hybrid' or expect a structured config.
        # To avoid complexity in 'get_loss' parsing recursion, let's stick to a simple MSE+SSIM for 'hybrid'
        # unless more specific info is in config.
        
        # NOTE: For true flexibility, the recursive parsing of Hydra config is needed.
        # Let's assume the user configures it like:
        # loss:
        #   name: hybrid
        #   losses:
        #     - name: mse
        #     - name: ssim
        #   weights: [1.0, 0.1]
        
        sub_losses = []
        if 'losses' in config:
            for sub_conf in config.losses:
                sub_losses.append(get_loss(sub_conf))
            weights = config.get('weights', [1.0] * len(sub_losses))
            return HybridLoss(sub_losses, weights)
        else:
            # Default hybrid
            return HybridLoss([nn.MSELoss(), SSIMLoss()], [1.0, 1.0])
            
    else:
        raise ValueError(f"Unknown loss function: {name}")
