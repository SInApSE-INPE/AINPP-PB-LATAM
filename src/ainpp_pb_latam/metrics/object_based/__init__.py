import logging

import numpy as np
import scipy.ndimage as ndimage

logger = logging.getLogger(__name__)
EPS = 1e-6


class ObjectBasedMetrics:
    @staticmethod
    def compute(pred: np.ndarray, target: np.ndarray, threshold: float) -> dict:
        """
        Computes cellular and object-based metrics using Connected Components.
        """
        pred_bin = (pred >= threshold).astype(int)
        target_bin = (target >= threshold).astype(int)

        metrics = {
            "Object_POD": np.nan,
            "Object_FAR": np.nan,
            "Object_CSI": np.nan,
            "Centroid_Distance": np.nan,
            "Count_Bias": np.nan,
        }

        # Batch processing handler or single frame
        if pred_bin.ndim == 3:  # (B, H, W)
            # Aggregate across batch or compute mean
            # Simplified: process average across batch frames
            batch_pod, batch_far, batch_csi, batch_count_bias = [], [], [], []
            for i in range(pred_bin.shape[0]):
                m = ObjectBasedMetrics._compute_single_frame(pred_bin[i], target_bin[i])
                batch_pod.append(m["pod"])
                batch_far.append(m["far"])
                batch_csi.append(m["csi"])
                batch_count_bias.append(m["count_bias"])

            return {
                "Object_POD": float(np.nanmean(batch_pod)),
                "Object_FAR": float(np.nanmean(batch_far)),
                "Object_CSI": float(np.nanmean(batch_csi)),
                "Count_Bias": float(np.nanmean(batch_count_bias)),
            }
        else:
            m = ObjectBasedMetrics._compute_single_frame(pred_bin, target_bin)
            return {
                "Object_POD": float(m["pod"]),
                "Object_FAR": float(m["far"]),
                "Object_CSI": float(m["csi"]),
                "Count_Bias": float(m["count_bias"]),
            }

    @staticmethod
    def _compute_single_frame(pred_bin2d, target_bin2d):
        # Connected components
        pred_labeled, num_pred = ndimage.label(pred_bin2d)
        target_labeled, num_target = ndimage.label(target_bin2d)

        count_bias = (
            num_pred / (num_target + EPS)
            if num_target > 0
            else (np.nan if num_pred == 0 else np.inf)
        )

        hits = 0
        false_alarms = 0
        misses = 0

        # Simple IoU based matching could be implemented here.
        # Fallback to pixel overlap matching for brevity
        for p_idx in range(1, num_pred + 1):
            p_mask = pred_labeled == p_idx
            if np.any(p_mask & (target_bin2d == 1)):
                hits += 1
            else:
                false_alarms += 1

        for t_idx in range(1, num_target + 1):
            t_mask = target_labeled == t_idx
            if not np.any(t_mask & (pred_bin2d == 1)):
                misses += 1

        pod = hits / (hits + misses + EPS) if (hits + misses) > 0 else np.nan
        far = false_alarms / (hits + false_alarms + EPS) if (hits + false_alarms) > 0 else np.nan
        csi = (
            hits / (hits + misses + false_alarms + EPS)
            if (hits + misses + false_alarms) > 0
            else np.nan
        )

        return {"pod": pod, "far": far, "csi": csi, "count_bias": count_bias}
