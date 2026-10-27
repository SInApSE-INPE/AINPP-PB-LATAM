import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ainpp_pb_latam.aggregation import Aggregator
from ainpp_pb_latam.evaluation.accumulators import (
    ContingencyAccumulator,
    ContinuousAccumulator,
    FSSAccumulator,
)
from ainpp_pb_latam.metrics.categorical.spatial import km_to_window_cells
from ainpp_pb_latam.metrics.consistency import ConsistencyMetrics
from ainpp_pb_latam.metrics.object_based import ObjectBasedMetrics
from ainpp_pb_latam.metrics.probabilistic import ProbabilisticMetrics
from ainpp_pb_latam.metrics.sharpness import SharpnessMetrics

logger = logging.getLogger(__name__)


def model_name_from_config(config) -> str:
    """Derives a model name from ``model._target_`` (e.g. '...models.unet.forecaster.X' -> 'unet')."""
    target = config.get("model", {}).get("_target_", "") if config is not None else ""
    parts = target.split(".")
    if "models" in parts and parts.index("models") + 1 < len(parts):
        return parts[parts.index("models") + 1]
    return parts[-1] if target else "Unknown"


class Evaluator:
    """
    Orchestrates the benchmark evaluation over lead times and thresholds.

    Pooled metrics (aggregation='pooled') are computed from statistics accumulated over
    the whole test set: continuous scores, contingency-table scores and FSS.
    Frame-level diagnostics (object-based, sharpness, consistency) are computed per batch
    and reported as mean/std over batches (aggregation='batch_mean').

    Predictions must be in physical units (mm/h). If the model outputs standardized values,
    pass the ``standardizer`` used in training so they are inverted before scoring.
    """

    def __init__(
        self,
        model,
        test_loader,
        config,
        standardizer=None,
        device="cpu",
        model_name: Optional[str] = None,
    ):
        self.model = model
        self.loader = test_loader
        self.config = config
        self.standardizer = standardizer
        self.device = device

        # Protocol
        eval_cfg = config.get("evaluation", {})
        self.model_name = model_name or eval_cfg.get("model_name") or model_name_from_config(config)

        # Directories
        out_cfg = eval_cfg.get("output_dir", "outputs/evaluation")
        self.base_output_dir = Path(out_cfg)
        self.data_dir = self.base_output_dir / "data"
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.thresholds = list(eval_cfg.get("thresholds_mm_h", [0.1, 1.0, 5.0, 10.0]))
        self.time_step_min = int(eval_cfg.get("time_step_min", 60))
        self.fss_scales_km = list(eval_cfg.get("fss_scales_km", []))
        self.grid_res_deg = float(eval_cfg.get("grid_res_deg", 0.1))
        self.forecast_type = eval_cfg.get("forecast_type", "deterministic")
        self.max_batches = eval_cfg.get("max_batches", None)

        # What to compute
        self.compute_cat = eval_cfg.get("categorical", True)
        self.compute_cont = eval_cfg.get("continuous", True)
        self.compute_fss = eval_cfg.get("fss", True) and bool(self.fss_scales_km)
        self.compute_prob = eval_cfg.get("probabilistic", False)
        self.compute_obj = eval_cfg.get("object_based", True)
        self.compute_sharp = eval_cfg.get("sharpness", True)
        self.compute_consist = eval_cfg.get("consistency", True)

        if self.compute_prob and self.forecast_type != "probability":
            logger.warning(
                "Probabilistic metrics require forecast_type='probability' "
                "(model outputs exceedance probabilities); skipping them for a "
                "'%s' forecast.",
                self.forecast_type,
            )
            self.compute_prob = False

        self.fss_windows = {
            km: km_to_window_cells(km, res_deg=self.grid_res_deg) for km in self.fss_scales_km
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def lead_name(self, t: int) -> str:
        return f"+{(t + 1) * self.time_step_min}min"

    @staticmethod
    def _unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Accepts (x, y), (x, y, mask), (x, y, meta) or (x, y, mask, meta)."""
        if not isinstance(batch, (list, tuple)):
            return batch, batch, None
        data, target = batch[0], batch[1]
        mask = next((b for b in batch[2:] if isinstance(b, torch.Tensor)), None)
        return data, target, mask

    @staticmethod
    def _to_lead_major(arr: np.ndarray) -> np.ndarray:
        """(B, T, 1, H, W) -> (B, T, H, W)."""
        if arr.ndim == 5:
            if arr.shape[2] != 1:
                raise ValueError(f"Expected a single channel, got shape {arr.shape}")
            arr = arr[:, :, 0]
        return arr

    def _init_accumulators(self, n_leads: int) -> None:
        self.cont_acc = [ContinuousAccumulator() for _ in range(n_leads)]
        self.cat_acc = [
            {th: ContingencyAccumulator(th) for th in self.thresholds} for _ in range(n_leads)
        ]
        self.fss_acc = [
            {
                (th, km): FSSAccumulator(th, win)
                for th in self.thresholds
                for km, win in self.fss_windows.items()
            }
            for _ in range(n_leads)
        ]

    def _record(self, records: List[Dict], lead: str, thresh, name: str, value) -> None:
        records.append(
            {
                "model": self.model_name,
                "lead_time": lead,
                "threshold": thresh,
                "metric_name": name,
                "value": float(value),
            }
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def evaluate(self) -> pd.DataFrame:
        self.model.eval()
        self.model.to(self.device)

        batch_records: List[Dict] = []
        n_leads: Optional[int] = None

        logger.info("Starting benchmark evaluation for model %s", self.model_name)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.loader, desc="Evaluating")):
                if self.max_batches is not None and batch_idx >= self.max_batches:
                    break

                data, target, mask = self._unpack_batch(batch)
                output = self.model(data.to(self.device))

                output_np = output.detach().cpu().numpy()
                if self.standardizer is not None:
                    output_np = np.asarray(self.standardizer.inverse_transform(output_np))
                output_np = self._to_lead_major(output_np)
                target_np = self._to_lead_major(target.cpu().numpy())
                mask_np = None if mask is None else self._to_lead_major(mask.cpu().numpy())

                if output_np.shape != target_np.shape:
                    raise ValueError(
                        f"Prediction shape {output_np.shape} != target shape {target_np.shape}"
                    )

                if n_leads is None:
                    n_leads = output_np.shape[1]
                    self._init_accumulators(n_leads)

                for t in range(n_leads):
                    lead = self.lead_name(t)
                    pred_t = output_np[:, t]
                    target_t = target_np[:, t]
                    mask_t = None if mask_np is None else mask_np[:, t]

                    # Pooled metrics
                    if self.compute_cont:
                        self.cont_acc[t].update(pred_t, target_t, mask_t)
                    if self.compute_cat:
                        for acc in self.cat_acc[t].values():
                            acc.update(pred_t, target_t, mask_t)
                    if self.compute_fss:
                        for acc in self.fss_acc[t].values():
                            acc.update(pred_t, target_t, mask_t)

                    # Frame-level diagnostics (per batch)
                    if self.compute_sharp:
                        for name, val in SharpnessMetrics.compute(pred_t, target_t).items():
                            self._record(batch_records, lead, np.nan, name, val)
                    if self.compute_consist:
                        for name, val in ConsistencyMetrics.compute(pred_t, target_t).items():
                            self._record(batch_records, lead, np.nan, name, val)

                    for thresh in self.thresholds:
                        if self.compute_obj:
                            metrics = ObjectBasedMetrics.compute(pred_t, target_t, threshold=thresh)
                            for name, val in metrics.items():
                                self._record(batch_records, lead, thresh, name, val)
                        if self.compute_prob:
                            metrics = ProbabilisticMetrics.compute(
                                pred_t, target_t, threshold=thresh
                            )
                            for name, val in metrics.items():
                                self._record(batch_records, lead, thresh, name, val)

        if n_leads is None:
            logger.warning("Empty test loader; nothing evaluated.")
            return pd.DataFrame()

        pooled_records, counts = self._collect_pooled(n_leads)

        logger.info("Aggregating metrics and saving tidy table...")
        df_batch = Aggregator.summarize(Aggregator.construct_tidy_dataframe(batch_records))
        df_pooled = Aggregator.construct_tidy_dataframe(pooled_records)
        df_summary = Aggregator.combine(df_pooled, df_batch)
        Aggregator.save_results(df_summary, output_dir=str(self.base_output_dir))

        if counts:
            pd.DataFrame(counts).to_csv(self.data_dir / "contingency_tables.csv", index=False)

        return df_summary

    def _collect_pooled(self, n_leads: int) -> Tuple[List[Dict], List[Dict]]:
        records: List[Dict] = []
        counts: List[Dict] = []
        for t in range(n_leads):
            lead = self.lead_name(t)
            if self.compute_cont:
                for name, val in self.cont_acc[t].compute().items():
                    self._record(records, lead, np.nan, name, val)
            if self.compute_cat:
                for thresh, acc in self.cat_acc[t].items():
                    for name, val in acc.compute().items():
                        self._record(records, lead, thresh, name, val)
                    counts.append(
                        {"model": self.model_name, "lead_time": lead, "threshold": thresh}
                        | acc.counts
                    )
            if self.compute_fss:
                for (thresh, km), acc in self.fss_acc[t].items():
                    self._record(records, lead, thresh, f"FSS_{km:g}km", acc.compute())
        return records, counts
