import numpy as np
import logging

logger = logging.getLogger(__name__)
EPS = 1e-6


class CategoricalMetrics:
    @staticmethod
    def compute(pred: np.ndarray, target: np.ndarray, threshold: float) -> dict:
        """
        Compute categorical metrics for a given threshold.
        Metrics: POD, FAR, CSI, ETS, HSS, Bias.
        Inputs: numpy arrays of shape (B, H, W) or (H, W).
        """
        pred_bin = (pred >= threshold).astype(int)
        target_bin = (target >= threshold).astype(int)

        hits = (pred_bin & target_bin).sum()
        misses = ((~pred_bin) & target_bin).sum()
        false_alarms = (pred_bin & (~target_bin)).sum()
        correct_negatives = ((~pred_bin) & (~target_bin)).sum()

        total = hits + misses + false_alarms + correct_negatives

        pod = hits / (hits + misses + EPS)
        far = false_alarms / (hits + false_alarms + EPS)
        csi = hits / (hits + misses + false_alarms + EPS)
        bias = (hits + false_alarms) / (hits + misses + EPS)

        # ETS
        hits_rand = (hits + misses) * (hits + false_alarms) / (total + EPS)
        ets = (hits - hits_rand) / (hits + misses + false_alarms - hits_rand + EPS)

        # HSS
        hss_num = 2 * (hits * correct_negatives - misses * false_alarms)
        hss_den = (hits + misses) * (misses + correct_negatives) + (hits + false_alarms) * (
            false_alarms + correct_negatives
        )
        hss = hss_num / (hss_den + EPS)

        return {
            "POD": float(pod),
            "FAR": float(far),
            "CSI": float(csi),
            "Bias": float(bias),
            "ETS": float(ets),
            "HSS": float(hss),
        }
