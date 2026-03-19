import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, r2_score, roc_curve

EPS = 1e-6


class Metrics:
    @staticmethod
    def compute_continuous_metrics(pred, target):
        """
        Compute continuous metrics: MSE, RMSE, MAE, R2, Correlation.
        Inputs are expected to be flattened or have same shape.
        """
        # Ensure numpy
        if hasattr(pred, "cpu"):
            pred = pred.cpu().numpy()
        if hasattr(target, "cpu"):
            target = target.cpu().numpy()

        flat_pred = pred.flatten()
        flat_target = target.flatten()

        mse = mean_squared_error(flat_target, flat_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(flat_target, flat_pred)
        r2 = r2_score(flat_target, flat_pred)

        # Correlation
        if len(flat_pred) > 1:
            corr, _ = pearsonr(flat_target, flat_pred)
        else:
            corr = 0.0

        return {
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "R2": float(r2),
            "Correlation": float(corr),
        }

    @staticmethod
    def compute_categorical_metrics(pred, target, thresholds=[0.5, 2.0, 5.0]):
        """
        Compute categorical metrics for multiple thresholds.
        Metrics: POD, FAR, POFD, TS (CSI), ETS, HSS.
        """
        if hasattr(pred, "cpu"):
            pred = pred.cpu().numpy()
        if hasattr(target, "cpu"):
            target = target.cpu().numpy()

        results = {}

        for thresh in thresholds:
            pred_bin = (pred >= thresh).astype(int)
            target_bin = (target >= thresh).astype(int)

            # Contingency Table
            # TP: Pred=1, Target=1
            # FP: Pred=1, Target=0
            # FN: Pred=0, Target=1
            # TN: Pred=0, Target=0

            hits = (pred_bin & target_bin).sum()  # TP
            misses = ((~pred_bin) & target_bin).sum()  # FN
            false_alarms = (pred_bin & (~target_bin)).sum()  # FP
            correct_negatives = ((~pred_bin) & (~target_bin)).sum()  # TN

            total = hits + misses + false_alarms + correct_negatives

            # Metrics
            pod = hits / (hits + misses + EPS)
            far = false_alarms / (hits + false_alarms + EPS)
            pofd = false_alarms / (correct_negatives + false_alarms + EPS)
            ts = hits / (hits + misses + false_alarms + EPS)  # CSI

            # ETS
            hits_rand = (hits + misses) * (hits + false_alarms) / total
            ets = (hits - hits_rand) / (hits + misses + false_alarms - hits_rand + EPS)

            # HSS
            hss_num = 2 * (hits * correct_negatives - misses * false_alarms)
            hss_den = (hits + misses) * (misses + correct_negatives) + (hits + false_alarms) * (
                false_alarms + correct_negatives
            )
            hss = hss_num / (hss_den + EPS)

            # SR (Success Ratio) = 1 - FAR
            sr = 1.0 - far

            prefix = f"Thresh_{thresh}"
            results[f"{prefix}_POD"] = float(pod)
            results[f"{prefix}_FAR"] = float(far)
            results[f"{prefix}_POFD"] = float(pofd)
            results[f"{prefix}_TS"] = float(ts)
            results[f"{prefix}_ETS"] = float(ets)
            results[f"{prefix}_HSS"] = float(hss)
            results[f"{prefix}_SR"] = float(sr)

        return results

    @staticmethod
    def compute_probabilistic_metrics(pred_probs, target_bin, threshold=0.5):
        """
        Compute probabilistic metrics: Brier Score, ROC, BSS.
        pred_probs: Probability of exceeding threshold (0-1).
        target_bin: Binary target (0 or 1) based on threshold.
        """
        if hasattr(pred_probs, "cpu"):
            pred_probs = pred_probs.cpu().numpy()
        if hasattr(target_bin, "cpu"):
            target_bin = target_bin.cpu().numpy()

        flat_probs = pred_probs.flatten()
        flat_target = target_bin.flatten()

        # Brier Score
        bs = mean_squared_error(flat_target, flat_probs)

        # ROC
        fpr, tpr, _ = roc_curve(flat_target, flat_probs)
        roc_auc = auc(fpr, tpr)

        # BSS (Brier Skill Score) - Reference: Climatological probability (mean of target)
        clim_prob = flat_target.mean()
        bs_ref = mean_squared_error(flat_target, np.full_like(flat_probs, clim_prob))
        bss = 1 - (bs / (bs_ref + EPS))

        return {
            "BrierScore": float(bs),
            "BSS": float(bss),
            "AUC": float(roc_auc),
            "ROC_fpr": fpr,  # Arrays, careful when validiting/logging
            "ROC_tpr": tpr,
        }

    @staticmethod
    def compute_crps(pred_ensemble, target):
        """
        Compute CRPS for ensemble forecasts.
        pred_ensemble: (E, ...) E ensemble members
        target: (...) Observed values
        """
        if hasattr(pred_ensemble, "cpu"):
            pred_ensemble = pred_ensemble.cpu().numpy()
        if hasattr(target, "cpu"):
            target = target.cpu().numpy()

        # Using properscoring if available, else simplified implementation
        # Simplified CRPS for ensemble (Empirical CDF)
        # CRPS = Integral( (F(x) - H(x-y))^2 dx )
        # E_CRPS = E|X-y| - 0.5 * E|X-X'|

        # Assuming pred_ensemble shape: (Members, Batch, H, W)
        # Target shape: (Batch, H, W)

        # Flatten spatial dims to treat them as independent samples for avg CRPS
        # Or keep spatial structure. Usually we want average CRPS over domain.

        # Use E|X-y| - 0.5 * E|X-X'| formulation
        # 1. Mean Absolute Error of each member vs target -> Average over members
        mae_members = np.abs(pred_ensemble - target[None, ...]).mean(axis=0)

        # 2. Mean Absolute Difference between members
        # Simple estimator can be heavy: O(M^2). For small M (e.g. 10-50) it's fine.
        n_members = pred_ensemble.shape[0]
        if n_members > 1:
            diff_sum = 0.0
            for i in range(n_members):
                for j in range(n_members):
                    diff_sum += np.abs(pred_ensemble[i] - pred_ensemble[j])
            expected_diff_members = diff_sum / (n_members * n_members)

            crps = mae_members - 0.5 * expected_diff_members
        else:
            # Deterministic case -> CRPS = MAE
            crps = mae_members

        return float(crps.mean())
