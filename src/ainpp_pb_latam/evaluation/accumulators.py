"""
Streaming accumulators for benchmark metrics.

Verification scores must be computed from statistics pooled over the whole test set
(e.g. one contingency table per lead time/threshold), not by averaging per-batch scores:
the mean of per-batch CSIs or RMSEs is not the CSI or RMSE of the test set and it
depends on the batch size and on how many events fall in each batch.
"""

from typing import Dict, Optional

import numpy as np
from scipy.ndimage import uniform_filter

from ainpp_pb_latam.metrics.categorical import contingency_table, scores_from_contingency


def _valid_mask(shape, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    return np.broadcast_to(np.asarray(mask, dtype=bool), shape)


class ContinuousAccumulator:
    """Pools sums over valid pixels to compute MAE, MSE, RMSE, ME, STDs and Pearson r."""

    def __init__(self) -> None:
        self.n = 0
        self.sum_p = 0.0
        self.sum_t = 0.0
        self.sum_pp = 0.0
        self.sum_tt = 0.0
        self.sum_pt = 0.0
        self.sum_abs_err = 0.0
        self.sum_sq_err = 0.0

    def update(self, pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None):
        valid = _valid_mask(pred.shape, mask)
        p = np.asarray(pred, dtype=np.float64)[valid]
        t = np.asarray(target, dtype=np.float64)[valid]
        self.n += p.size
        self.sum_p += p.sum()
        self.sum_t += t.sum()
        self.sum_pp += (p * p).sum()
        self.sum_tt += (t * t).sum()
        self.sum_pt += (p * t).sum()
        self.sum_abs_err += np.abs(p - t).sum()
        self.sum_sq_err += ((p - t) ** 2).sum()

    def compute(self) -> Dict[str, float]:
        if self.n == 0:
            keys = ["MAE", "RMSE", "MSE", "ME", "STD_obs", "STD_pred", "Correlation"]
            return {k: np.nan for k in keys}

        n = float(self.n)
        mean_p, mean_t = self.sum_p / n, self.sum_t / n
        mse = self.sum_sq_err / n
        var_p = max(self.sum_pp / n - mean_p**2, 0.0)
        var_t = max(self.sum_tt / n - mean_t**2, 0.0)
        cov = self.sum_pt / n - mean_p * mean_t
        std_p, std_t = np.sqrt(var_p), np.sqrt(var_t)
        corr = cov / (std_p * std_t) if std_p > 0 and std_t > 0 else np.nan

        return {
            "MAE": self.sum_abs_err / n,
            "RMSE": float(np.sqrt(mse)),
            "MSE": mse,
            "ME": mean_p - mean_t,
            "STD_obs": float(std_t),
            "STD_pred": float(std_p),
            "Correlation": float(corr),
        }


class ContingencyAccumulator:
    """Pools the 2x2 contingency table for one threshold."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.counts = {"hits": 0, "misses": 0, "false_alarms": 0, "correct_negatives": 0}

    def update(self, pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None):
        table = contingency_table(pred, target, self.threshold, mask=mask)
        for k, v in table.items():
            self.counts[k] += v

    def compute(self) -> Dict[str, float]:
        return {k: float(v) for k, v in scores_from_contingency(**self.counts).items()}


class FSSAccumulator:
    """
    Pooled Fractions Skill Score (Roberts & Lean, 2008) for one threshold and window size.

    FSS = 1 - sum((F - O)^2) / sum(F^2 + O^2), with the sums taken over every valid pixel
    of every frame, so the score is not biased by frames without events.
    Fractions are computed per frame over the last two dims with zero padding at the borders.
    """

    def __init__(self, threshold: float, window: int) -> None:
        self.threshold = threshold
        self.window = int(window)
        self.num = 0.0
        self.den = 0.0

    def update(self, pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None):
        valid = _valid_mask(pred.shape, mask)
        f_evt = ((pred >= self.threshold) & valid).astype(np.float32)
        o_evt = ((target >= self.threshold) & valid).astype(np.float32)

        size = [1] * f_evt.ndim
        size[-2] = size[-1] = self.window
        F = uniform_filter(f_evt, size=size, mode="constant", cval=0.0)
        O = uniform_filter(o_evt, size=size, mode="constant", cval=0.0)

        self.num += float(((F - O) ** 2)[valid].sum(dtype=np.float64))
        self.den += float((F**2 + O**2)[valid].sum(dtype=np.float64))

    def compute(self) -> float:
        return 1.0 - self.num / self.den if self.den > 0 else np.nan
