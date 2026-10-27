from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from ainpp_pb_latam.evaluation.evaluator import Evaluator, model_name_from_config
from ainpp_pb_latam.models.persistence import PersistenceForecaster


def _config(tmp_path: Path, **overrides) -> OmegaConf:
    evaluation = {
        "output_dir": str(tmp_path / "eval"),
        "thresholds_mm_h": [1.0],
        "time_step_min": 60,
        "fss_scales_km": [11.1],
        "object_based": False,
        "sharpness": False,
        "consistency": False,
    }
    evaluation.update(overrides)
    return OmegaConf.create(
        {
            "model": {"_target_": "ainpp_pb_latam.models.persistence.forecaster.X"},
            "evaluation": evaluation,
        }
    )


def _perfect_persistence_batch(tout: int = 3):
    """Inputs whose last frame equals every target frame -> persistence is perfect."""
    last = torch.zeros(2, 1, 1, 8, 8)
    last[:, :, :, 2:5, 2:5] = 4.0
    x = torch.cat([torch.rand(2, 4, 1, 8, 8), last], dim=1)
    y = last.expand(-1, tout, -1, -1, -1).clone()
    return x, y


class TestPersistence:
    def test_repeats_last_frame(self) -> None:
        x = torch.rand(2, 5, 1, 4, 4)
        y = PersistenceForecaster(output_timesteps=3)(x)
        assert y.shape == (2, 3, 1, 4, 4)
        assert torch.equal(y[:, 2], x[:, -1])


class TestEvaluator:
    def _metric(self, df: pd.DataFrame, name: str, lead: str = "+60min") -> float:
        row = df[(df["metric_name"] == name) & (df["lead_time"] == lead)]
        assert len(row) == 1
        return float(row["mean"].iloc[0])

    def test_perfect_forecast_end_to_end(self, tmp_path: Path) -> None:
        x, y = _perfect_persistence_batch()
        evaluator = Evaluator(
            PersistenceForecaster(output_timesteps=3), [(x, y)], _config(tmp_path)
        )
        df = evaluator.evaluate()

        assert set(df["lead_time"]) == {"+60min", "+120min", "+180min"}
        assert self._metric(df, "RMSE") == pytest.approx(0.0)
        assert self._metric(df, "CSI") == pytest.approx(1.0)
        assert self._metric(df, "ETS") == pytest.approx(1.0)
        assert self._metric(df, "FSS_11.1km") == pytest.approx(1.0)
        assert (df["aggregation"] == "pooled").all()
        assert (tmp_path / "eval" / "evaluation_summary.csv").exists()
        counts = pd.read_csv(tmp_path / "eval" / "data" / "contingency_tables.csv")
        assert (counts[["hits", "misses", "false_alarms", "correct_negatives"]] >= 0).all().all()

    def test_mask_ignores_invalid_pixels(self, tmp_path: Path) -> None:
        x, y = _perfect_persistence_batch()
        y_corrupted = y.clone()
        y_corrupted[..., 0, :] = 99.0  # garbage in a row flagged invalid
        mask = torch.ones_like(y, dtype=torch.bool)
        mask[..., 0, :] = False

        evaluator = Evaluator(
            PersistenceForecaster(output_timesteps=3),
            [(x, y_corrupted, mask)],
            _config(tmp_path),
        )
        df = evaluator.evaluate()
        assert self._metric(df, "RMSE") == pytest.approx(0.0)
        assert self._metric(df, "CSI") == pytest.approx(1.0)

    def test_probabilistic_skipped_for_deterministic(self, tmp_path: Path) -> None:
        evaluator = Evaluator(PersistenceForecaster(), [], _config(tmp_path, probabilistic=True))
        assert evaluator.compute_prob is False

    def test_standardizer_is_inverted(self, tmp_path: Path) -> None:
        class _Log1p:
            def inverse_transform(self, arr):
                return np.expm1(arr)

        x, y = _perfect_persistence_batch()
        evaluator = Evaluator(
            PersistenceForecaster(output_timesteps=3),
            [(torch.log1p(x), y)],
            _config(tmp_path),
            standardizer=_Log1p(),
        )
        df = evaluator.evaluate()
        assert self._metric(df, "RMSE") == pytest.approx(0.0, abs=1e-5)

    def test_model_name_from_target(self) -> None:
        cfg = OmegaConf.create({"model": {"_target_": "ainpp_pb_latam.models.unet.forecaster.U"}})
        assert model_name_from_config(cfg) == "unet"


class TestOpticalFlow:
    def test_advects_moving_blob(self) -> None:
        pytest.importorskip("pysteps")
        from ainpp_pb_latam.models.optical_flow import OpticalFlowForecaster

        frames = torch.zeros(1, 4, 1, 64, 64)
        for t in range(4):  # blob moving 2 px/step to the east
            frames[0, t, 0, 20:30, 10 + 2 * t : 20 + 2 * t] = 5.0

        out = OpticalFlowForecaster(output_timesteps=2, motion_method="LK")(frames)
        assert out.shape == (1, 2, 1, 64, 64)
        assert torch.isfinite(out).all() and (out >= 0).all()
