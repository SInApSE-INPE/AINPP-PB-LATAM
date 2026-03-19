from pathlib import Path

import pytest
import torch

from ainpp_pb_latam._utils.standardization import LogZScoreStandardizer
from ainpp_pb_latam.utils import EarlyStopping, build_optimizer


class _ToyModel(torch.nn.Module):
    """A minimal neural network model for testing utility functions."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestEarlyStopping:
    """Test suite for the EarlyStopping model regularizer."""

    def test_early_stopping_triggers_correctly(self, tmp_path: Path) -> None:
        """Test that EarlyStopping saves models on improvement and halts on stale training."""
        # Arrange
        model = _ToyModel()
        checkpoint = tmp_path / "best.pt"
        stopper = EarlyStopping(patience=2, delta=0.0, path=checkpoint, enabled=True)

        # Act & Assert
        # Baseline score
        stopper(1.0, model)
        assert not stopper.early_stop

        # Improvement saves checkpoint
        stopper(0.9, model)
        assert checkpoint.exists()
        assert not stopper.early_stop

        # Worse score (delay 1)
        stopper(1.0, model)
        assert not stopper.early_stop

        # Worse score (delay 2), should trigger early stopping
        stopper(1.1, model)
        assert stopper.early_stop


class TestOptimizerBuilder:
    """Test suite for the dynamic optimizer builder utility."""

    @pytest.mark.parametrize("lr", [0.005, 0.01, 1e-4])
    def test_build_optimizer_respects_learning_rate(self, lr: float) -> None:
        """Test that building an optimizer utilizes the configured learning rate."""
        # Arrange
        model = _ToyModel()
        cfg = {"lr": lr}

        # Act
        opt = build_optimizer(model.parameters(), cfg)

        # Assert
        assert opt.defaults["lr"] == pytest.approx(lr)


class TestStandardization:
    """Test suite for the data standardizers."""

    @pytest.mark.parametrize("values", [[0.0, 1.0, 2.0], [0.5, 3.14, 2.71]])
    def test_log_zscore_roundtrip(self, values: list[float]) -> None:
        """Test that LogZScoreStandardizer preserves values through forward/inverse transformation."""
        # Arrange
        std = LogZScoreStandardizer(mean_log=1.0, std_log=0.5)

        # Act
        transformed = std.transform(values)
        recovered = std.inverse_transform(transformed)

        # Assert
        assert recovered.shape == transformed.shape
        # Use np.testing or pytest.approx to check element-wise arrays easily
        for rec, val in zip(recovered, values):
            assert rec == pytest.approx(val, rel=1e-5, abs=1e-5)
