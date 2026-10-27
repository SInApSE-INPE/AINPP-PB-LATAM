import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)
EPS = 1e-6


def contingency_table(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, int]:
    """
    Builds the 2x2 contingency table for the event ``value >= threshold``.

    Args:
        pred: Forecast field of any shape.
        target: Observed field with the same shape as ``pred``.
        threshold: Event threshold (mm/h).
        mask: Optional boolean array (broadcastable to ``pred``); only True pixels are counted.

    Returns:
        Dict with integer counts: hits, misses, false_alarms, correct_negatives.
    """
    # Boolean arrays: ``~`` is a logical NOT here (on int arrays it would be a bitwise NOT).
    pred_evt = np.asarray(pred) >= threshold
    obs_evt = np.asarray(target) >= threshold

    if mask is None:
        valid = np.ones(pred_evt.shape, dtype=bool)
    else:
        valid = np.broadcast_to(np.asarray(mask, dtype=bool), pred_evt.shape)

    return {
        "hits": int(np.count_nonzero(pred_evt & obs_evt & valid)),
        "misses": int(np.count_nonzero(~pred_evt & obs_evt & valid)),
        "false_alarms": int(np.count_nonzero(pred_evt & ~obs_evt & valid)),
        "correct_negatives": int(np.count_nonzero(~pred_evt & ~obs_evt & valid)),
    }


def scores_from_contingency(
    hits: float, misses: float, false_alarms: float, correct_negatives: float
) -> Dict[str, float]:
    """
    Computes categorical scores from contingency table counts.
    Metrics: POD, FAR, CSI, Bias, ETS, HSS.

    Scores whose denominator is zero (e.g. POD when no event was observed) are NaN,
    so that they are not silently reported as 0.
    """
    h, m, f, cn = (float(v) for v in (hits, misses, false_alarms, correct_negatives))
    total = h + m + f + cn

    def _ratio(num: float, den: float) -> float:
        return num / den if den > 0 else np.nan

    hits_rand = (h + m) * (h + f) / total if total > 0 else np.nan
    hss_den = (h + m) * (m + cn) + (h + f) * (f + cn)

    return {
        "POD": _ratio(h, h + m),
        "FAR": _ratio(f, h + f),
        "CSI": _ratio(h, h + m + f),
        "Bias": _ratio(h + f, h + m),
        "ETS": _ratio(h - hits_rand, h + m + f - hits_rand),
        "HSS": _ratio(2.0 * (h * cn - m * f), hss_den),
    }


class CategoricalMetrics:
    @staticmethod
    def compute(
        pred: np.ndarray,
        target: np.ndarray,
        threshold: float,
        mask: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Compute categorical metrics for a given threshold.
        Metrics: POD, FAR, CSI, ETS, HSS, Bias.
        Inputs: numpy arrays of shape (B, H, W) or (H, W).

        Note: for benchmark aggregation over a whole test set, accumulate the
        contingency tables (see ``contingency_table``) instead of averaging these scores.
        """
        table = contingency_table(pred, target, threshold, mask=mask)
        return {k: float(v) for k, v in scores_from_contingency(**table).items()}
