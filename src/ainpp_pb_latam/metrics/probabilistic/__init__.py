import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, auc, precision_recall_curve
import logging

logger = logging.getLogger(__name__)
EPS = 1e-6

class ProbabilisticMetrics:
    @staticmethod
    def compute(pred_probs: np.ndarray, target: np.ndarray, threshold: float) -> dict:
        """
        Calculates probabilistic metrics given a forecasted probability and a continuous target field.
        
        Args:
            pred_probs: numpy array [H, W] or [B, H, W] containing probabilities [0, 1].
            target: continuous true values [H, W] or [B, H, W].
            threshold: cutoff for the target continuous values.
        """
        # Binarize target
        target_bin = (target >= threshold).astype(int).flatten()
        pred_flat = pred_probs.flatten()
        
        # Ensure probs are strictly bounded
        pred_flat = np.clip(pred_flat, 0.0, 1.0)
        
        # Base checks
        if target_bin.sum() == 0 or target_bin.sum() == len(target_bin):
            return {"BrierScore": np.nan, "BSS": np.nan, "ROC_AUC": np.nan, "PR_AUC": np.nan}
            
        bs = brier_score_loss(target_bin, pred_flat)
        
        # Climatological baseline for BSS
        clim_prob = target_bin.mean()
        bs_ref = brier_score_loss(target_bin, np.full_like(pred_flat, clim_prob))
        bss = 1.0 - (bs / (bs_ref + EPS))
        
        try:
            roc_auc = roc_auc_score(target_bin, pred_flat)
        except ValueError:
            roc_auc = np.nan
            
        try:
            precision, recall, _ = precision_recall_curve(target_bin, pred_flat)
            pr_auc = auc(recall, precision)
        except ValueError:
            pr_auc = np.nan
            
        return {
            "BrierScore": float(bs),
            "BSS": float(bss),
            "ROC_AUC": float(roc_auc),
            "PR_AUC": float(pr_auc)
        }
