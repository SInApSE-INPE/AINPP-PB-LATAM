import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
from typing import List, Sequence
from pytorch_msssim import ssim


class SharpnessLoss(nn.Module):
    def __init__(self, w_mse=1.0, w_l1=1.0, w_ssim=1.0, threshold=0.5, high_weight=5.0):
        super().__init__()
        self.w_mse = w_mse
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.threshold = threshold
        self.high_weight = high_weight

    def forward(self, pred, target):
        # Weighted MSE (Focuses on sharp regions)
        mse_diff = (pred - target) ** 2
        weights = torch.where(target > self.threshold, self.high_weight, 1.0)
        loss_mse = (mse_diff * weights).mean()

        # L1 Loss (Helps overall sharpness)
        loss_l1 = F.l1_loss(pred, target)

        
        # SSIM Loss (Focuses on structure/texture)
        # SSIM returns value between 0 and 1 (1 is identical). We want to minimize 1 - SSIM.
        loss_ssim = 1 - ssim(pred, target, data_range=target.max() - target.min(), size_average=True)

        return (self.w_mse * loss_mse) + (self.w_l1 * loss_l1) + (self.w_ssim * loss_ssim)

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

# class SSIMLoss(nn.Module):
#     def __init__(self, window_size=11, size_average=True):
#         super(SSIMLoss, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.channel = 1
#         self.window = self.create_window(window_size, self.channel)

#     def gaussian(self, window_size, sigma):
#         gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
#         return gauss/gauss.sum()

#     def create_window(self, window_size, channel):
#         _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
#         _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#         window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
#         return window

#     def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
#         mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
#         mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
#         mu1_sq = mu1.pow(2)
#         mu2_sq = mu2.pow(2)
#         mu1_mu2 = mu1*mu2
#         sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
#         sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
#         sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
#         C1 = 0.01**2
#         C2 = 0.03**2
#         ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
#         if size_average:
#             return ssim_map.mean()
#         else:
#             return ssim_map.mean(1).mean(1).mean(1)

#     def forward(self, img1, img2, **kwargs):
#         (_, channel, _, _) = img1.size()
#         if channel == self.channel and self.window.data.type() == img1.data.type():
#             window = self.window
#         else:
#             window = self.create_window(self.window_size, channel)
#             if img1.is_cuda:
#                 window = window.cuda(img1.get_device())
#             window = window.type_as(img1)
#             self.window = window
#             self.channel = channel
#         return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

class SSIMLoss(nn.Module):
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

    def forward(self, img1, img2, **kwargs):
        # --- CORREÇÃO AQUI ---
        # Se a entrada for 5D (Batch, Time, Channel, H, W), achata Batch e Time.
        if img1.ndim == 5:
            b, t, c, h, w = img1.shape
            # Transforma em (Batch * Time, Channel, H, W)
            img1 = img1.reshape(b * t, c, h, w)
            img2 = img2.reshape(b * t, c, h, w)
        # ---------------------

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

class PerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[3, 8, 15, 22]): 
        super().__init__()
        try:
            # Note: requires internet or cached weights
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.vgg_layers = vgg.features
            self.layer_ids = layer_ids
            self.freeze()
        except Exception as e:
            print(f"Warning: Could not load VGG weights: {e}. Perceptual loss disabled.")
            self.vgg_layers = None

    def freeze(self):
        if self.vgg_layers:
            for param in self.vgg_layers.parameters():
                param.requires_grad = False

    def forward(self, input, target, **kwargs):
        if self.vgg_layers is None:
            return F.mse_loss(input, target)
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
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, **kwargs):
        # inputs: logits, targets: indices/probs
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        return focal_loss

class CRPSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, **kwargs):
        if input.shape[1] == 2:
            mu = input[:, 0:1, :, :]
            sigma = torch.exp(input[:, 1:2, :, :]) 
            z = (target - mu) / sigma
            pdf = (1. / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * z ** 2)
            cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
            loss = sigma * (z * (2 * cdf - 1) + 2 * pdf - 1/math.sqrt(math.pi))
            return loss.mean()
        else:
            return F.l1_loss(input, target)

# --- HYBRID LOSS REFATORADO PARA HYDRA ---

class HybridLoss(nn.Module):
    """
    Computa uma soma ponderada de múltiplas funções de perda.
    Ideal para ser instanciada via Hydra (recursivamente).
    """
    def __init__(self, losses: Sequence[nn.Module], weights: Sequence[float]):
        super().__init__()
        # nn.ModuleList é essencial para o PyTorch registrar os sub-módulos (ex: VGG do Perceptual)
        self.losses = nn.ModuleList(losses)
        self.weights = weights
        
        assert len(self.losses) == len(self.weights), "Número de losses e pesos deve ser igual"

    def forward(self, input, target, **kwargs):
        total_loss = 0
        for loss_fn, w in zip(self.losses, self.weights):
            # Passamos kwargs para garantir compatibilidade (ex: weights do WeightedMSE)
            total_loss += w * loss_fn(input, target, **kwargs)
        return total_loss