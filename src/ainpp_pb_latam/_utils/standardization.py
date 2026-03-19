import os

import numpy as np


class LogZScoreStandardizer:
    def __init__(self, mean_log=None, std_log=None, params_dir=None, region=None):
        """
        Initialize the LogZScoreStandardizer.

        Args:
            mean_log (float or np.ndarray, optional): Mean of the log-transformed data.
            std_log (float or np.ndarray, optional): Std of the log-transformed data.
            params_dir (str, optional): Directory to load params from if mean/std not provided.
            region (str, optional): Region name to find the correct param files.
        """
        if mean_log is not None and std_log is not None:
            self.mean_log = mean_log
            self.std_log = std_log
        elif params_dir is not None and region is not None:
            self.load_params(params_dir, region)
        else:
            # Default to identity if nothing provided (or raise error depending on strictness)
            print("Warning: LogZScoreStandardizer initialized without parameters. Using Identity.")
            self.mean_log = 0.0
            self.std_log = 1.0

    def load_params(self, params_dir, region):
        """Load mean and std from .npy files."""
        try:
            mean_path = os.path.join(params_dir, f"gsmap_nrt+mvk_log_mean_{region}.npy")
            std_path = os.path.join(params_dir, f"gsmap_nrt+mvk_log_std_{region}.npy")

            self.mean_log = np.load(mean_path)
            self.std_log = np.load(std_path)
            print(f"Loaded standardization params from {params_dir}")
            print(f"Mean: {self.mean_log}, Std: {self.std_log}")
        except Exception as e:
            print(f"Error loading parameters: {e}. Using identity.")
            self.mean_log = 0.0
            self.std_log = 1.0

    def transform(self, x):
        """
        Apply Log-Zscore transformation.
        x_norm = (log1p(x) - mean) / std
        """
        x_log = np.log1p(x)
        x_norm = (x_log - self.mean_log) / self.std_log
        return x_norm

    def inverse_transform(self, x_norm):
        """
        Revert Log-Zscore transformation to mm/h.
        x_mmh = exp(x_norm * std + mean) - 1
        """
        # 1. Undo Z-Score
        # Ensure operations work with both scalar and tensors (numpy/torch)
        # converting to numpy if it's a tensor might be needed if we want pure numpy output,
        # but let's try to keep it compatible if passed types support broadcasting.
        if hasattr(x_norm, "cpu"):  # is tensor
            x_norm = x_norm.cpu().numpy()

        x_log = x_norm * self.std_log + self.mean_log

        # 2. Undo Log (exp(x) - 1)
        x_mmh = np.expm1(x_log)

        # 3. Ensure non-negativity
        return np.maximum(x_mmh, 0.0)
