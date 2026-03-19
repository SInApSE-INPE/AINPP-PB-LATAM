import numpy as np


class ContinuousMetrics:
    @staticmethod
    def compute(pred: np.ndarray, target: np.ndarray) -> dict:
        """
        Compute continuous metrics: MAE, RMSE, ME.
        Inputs: numpy arrays of shape (B, H, W) or (H, W).
        """
        mae = np.abs(pred - target).mean()
        mse = ((pred - target) ** 2).mean()
        rmse = np.sqrt(mse)
        me = (pred - target).mean()  # Mean Error (Bias)

        # for Taylor Diagram
        std_obs = target.std()
        std_pred = pred.std()

        # Pearson correlation
        if std_obs > 0 and std_pred > 0:
            corr = ((pred - pred.mean()) * (target - target.mean())).mean() / (std_obs * std_pred)
        else:
            corr = 0.0

        return {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "MSE": float(mse),
            "ME": float(me),
            "STD_obs": float(std_obs),
            "STD_pred": float(std_pred),
            "Correlation": float(corr),
        }
