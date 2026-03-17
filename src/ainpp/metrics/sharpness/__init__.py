import numpy as np
from scipy import signal
import logging

logger = logging.getLogger(__name__)

class SharpnessMetrics:
    @staticmethod
    def compute(pred: np.ndarray, target: np.ndarray) -> dict:
        """
        Measures structural properties, blur effect, and sharpness of the spatial fields.
        Metrics: Total Variation (TV), Power Spectral Density (PSD) Slopes.
        """
        metrics = {}
        
        # Total Variation indicates how noisy or blurred an image is
        tv_pred = SharpnessMetrics._compute_total_variation(pred)
        tv_target = SharpnessMetrics._compute_total_variation(target)
        tv_ratio = tv_pred / (tv_target + 1e-6)
        
        metrics["Total_Variation_Ratio"] = float(tv_ratio)
        
        # Fast structural approximation (Gradient magnitude mean)
        grad_p_y, grad_p_x = np.gradient(pred.mean(axis=0) if pred.ndim == 3 else pred)
        grad_t_y, grad_t_x = np.gradient(target.mean(axis=0) if target.ndim == 3 else target)
        
        grad_p_mag = np.sqrt(grad_p_y**2 + grad_p_x**2).mean()
        grad_t_mag = np.sqrt(grad_t_y**2 + grad_t_x**2).mean()
        
        metrics["Gradient_Magnitude_Ratio"] = float(grad_p_mag / (grad_t_mag + 1e-6))
        
        return metrics
        
    @staticmethod
    def _compute_total_variation(img):
        # Assume batch dim or 2D
        if img.ndim == 3: # (B, H, W)
            dy = np.abs(img[:, 1:, :] - img[:, :-1, :]).mean()
            dx = np.abs(img[:, :, 1:] - img[:, :, :-1]).mean()
        else:
            dy = np.abs(img[1:, :] - img[:-1, :]).mean()
            dx = np.abs(img[:, 1:] - img[:, :-1]).mean()
        return dy + dx
