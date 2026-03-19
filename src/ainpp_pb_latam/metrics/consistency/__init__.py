import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp
import logging

logger = logging.getLogger(__name__)


class ConsistencyMetrics:
    @staticmethod
    def compute(pred: np.ndarray, target: np.ndarray) -> dict:
        """
        Measures the statistical consistency and distributional realism of the model
        outputs compared to the ground truth.
        Metrics: Wasserstein Distance, Kolmogorov-Smirnov Distance.
        """
        # Flattened spatial distribution matching
        p_flat = pred.flatten()
        t_flat = target.flatten()

        # 1. Earth Mover's Distance (Wasserstein 1D)
        # Using subsampling if matrix is too large for fast compute
        max_samples = 100000
        if len(p_flat) > max_samples:
            idx = np.random.choice(len(p_flat), max_samples, replace=False)
            p_samp = p_flat[idx]
            t_samp = t_flat[idx]
        else:
            p_samp, t_samp = p_flat, t_flat

        w_dist = wasserstein_distance(p_samp, t_samp)

        # 2. Kolmogorov-Smirnov static (Max CDF separation)
        ks_stat, _ = ks_2samp(p_samp, t_samp)

        return {
            "Wasserstein_Dist": float(w_dist),
            "KS_Distance": float(ks_stat),
            "Mean_Bias": float(p_flat.mean() - t_flat.mean()),
            "Std_Ratio": float(p_flat.std() / (t_flat.std() + 1e-6)),
        }
