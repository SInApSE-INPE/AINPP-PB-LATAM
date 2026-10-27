import numpy as np
import pandas as pd
import pytest

from ainpp_pb_latam.aggregation import Aggregator
from ainpp_pb_latam.evaluation.accumulators import (
    ContingencyAccumulator,
    ContinuousAccumulator,
    FSSAccumulator,
)
from ainpp_pb_latam.metrics.categorical import (
    CategoricalMetrics,
    contingency_table,
    scores_from_contingency,
)

# pred/obs events: hit, false alarm, miss, correct negative, false alarm
PRED = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
OBS = np.array([1.0, 0.0, 1.0, 0.0, 0.0])


class TestContingency:
    def test_counts_are_correct_and_non_negative(self) -> None:
        table = contingency_table(PRED, OBS, threshold=0.5)
        assert table == {"hits": 1, "misses": 1, "false_alarms": 2, "correct_negatives": 1}

    def test_counts_with_int_inputs(self) -> None:
        """Regression: ``~`` on int arrays is bitwise and produced negative counts."""
        table = contingency_table(PRED.astype(int), OBS.astype(int), threshold=1)
        assert table["correct_negatives"] == 1

    def test_mask_excludes_pixels(self) -> None:
        mask = np.array([True, True, True, True, False])
        table = contingency_table(PRED, OBS, threshold=0.5, mask=mask)
        assert table == {"hits": 1, "misses": 1, "false_alarms": 1, "correct_negatives": 1}

    def test_scores_match_hand_computation(self) -> None:
        scores = CategoricalMetrics.compute(PRED, OBS, threshold=0.5)
        assert scores["POD"] == pytest.approx(0.5)
        assert scores["FAR"] == pytest.approx(2 / 3)
        assert scores["CSI"] == pytest.approx(0.25)
        assert scores["Bias"] == pytest.approx(1.5)
        # hits_random = (1 + 1) * (1 + 2) / 5 = 1.2
        assert scores["ETS"] == pytest.approx((1 - 1.2) / (4 - 1.2))
        # HSS = 2 (1*1 - 1*2) / ((1+1)(1+1) + (1+2)(2+1))
        assert scores["HSS"] == pytest.approx(-2 / 13)

    def test_perfect_forecast(self) -> None:
        scores = CategoricalMetrics.compute(OBS, OBS, threshold=0.5)
        assert scores["CSI"] == pytest.approx(1.0)
        assert scores["ETS"] == pytest.approx(1.0)
        assert scores["HSS"] == pytest.approx(1.0)
        assert scores["FAR"] == pytest.approx(0.0)

    def test_undefined_scores_are_nan(self) -> None:
        scores = scores_from_contingency(0, 0, 0, 10)
        assert np.isnan(scores["POD"])
        assert np.isnan(scores["CSI"])


class TestPooledAccumulators:
    def test_pooled_contingency_differs_from_batch_mean(self) -> None:
        rng = np.random.default_rng(0)
        pred = rng.gamma(0.5, 2.0, size=(4, 16, 16))
        obs = rng.gamma(0.5, 2.0, size=(4, 16, 16))

        acc = ContingencyAccumulator(threshold=1.0)
        for b in range(4):
            acc.update(pred[b], obs[b])

        full = CategoricalMetrics.compute(pred, obs, threshold=1.0)
        assert acc.compute()["CSI"] == pytest.approx(full["CSI"])

    def test_continuous_matches_global_computation(self) -> None:
        rng = np.random.default_rng(1)
        pred = rng.random((3, 8, 8))
        obs = rng.random((3, 8, 8))
        mask = rng.random((3, 8, 8)) > 0.2

        acc = ContinuousAccumulator()
        for b in range(3):
            acc.update(pred[b], obs[b], mask[b])
        res = acc.compute()

        p, o = pred[mask], obs[mask]
        assert res["MAE"] == pytest.approx(np.abs(p - o).mean())
        assert res["RMSE"] == pytest.approx(np.sqrt(((p - o) ** 2).mean()))
        assert res["ME"] == pytest.approx((p - o).mean())
        assert res["Correlation"] == pytest.approx(np.corrcoef(p, o)[0, 1])

    def test_empty_continuous_is_nan(self) -> None:
        assert np.isnan(ContinuousAccumulator().compute()["RMSE"])


class TestFSS:
    def test_identical_fields_score_one(self) -> None:
        field = np.zeros((1, 20, 20))
        field[0, 5:8, 5:8] = 5.0
        acc = FSSAccumulator(threshold=1.0, window=3)
        acc.update(field, field)
        assert acc.compute() == pytest.approx(1.0)

    def test_displacement_penalized_at_grid_scale_only(self) -> None:
        obs = np.zeros((1, 21, 21))
        pred = np.zeros((1, 21, 21))
        obs[0, 10, 8] = 5.0
        pred[0, 10, 12] = 5.0

        small = FSSAccumulator(threshold=1.0, window=1)
        small.update(pred, obs)
        large = FSSAccumulator(threshold=1.0, window=15)
        large.update(pred, obs)

        assert small.compute() == pytest.approx(0.0)
        # 15x15 squares shifted by 4 px: FSS = 1 - (2 * 15 * 4) / (2 * 15 * 15)
        assert large.compute() == pytest.approx(1 - 120 / 450)

    def test_no_events_is_nan(self) -> None:
        acc = FSSAccumulator(threshold=1.0, window=3)
        acc.update(np.zeros((1, 5, 5)), np.zeros((1, 5, 5)))
        assert np.isnan(acc.compute())


class TestSkillScores:
    def test_skill_against_reference(self) -> None:
        df = pd.DataFrame(
            {
                "model": ["persistence", "unet", "persistence", "unet"],
                "lead_time": ["+60min"] * 4,
                "threshold": [np.nan, np.nan, 1.0, 1.0],
                "metric_name": ["MSE", "MSE", "CSI", "CSI"],
                "mean": [4.0, 3.0, 0.2, 0.6],
                "aggregation": ["pooled"] * 4,
            }
        )
        skill = Aggregator.skill_scores(df, reference="persistence")
        unet = skill[skill["model"] == "unet"].set_index("metric_name")["skill"]
        assert unet["MSE"] == pytest.approx(0.25)  # 1 - 3/4
        assert unet["CSI"] == pytest.approx(0.5)  # (0.6 - 0.2) / (1 - 0.2)

    def test_missing_reference_raises(self) -> None:
        df = pd.DataFrame(
            {
                "model": ["unet"],
                "lead_time": ["+60min"],
                "threshold": [np.nan],
                "metric_name": ["MSE"],
                "mean": [1.0],
            }
        )
        with pytest.raises(ValueError):
            Aggregator.skill_scores(df, reference="persistence")
