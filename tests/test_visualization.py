from pathlib import Path

import numpy as np
import pytest

from ainpp_pb_latam.visualization.plot_metrics import plot_reliability_diagram, plot_roc_curve


class TestPlotMetrics:
    """Test suite for the metric visualization plotting routines."""

    @pytest.mark.parametrize("size", [50, 100])
    def test_reliability_and_roc_outputs(self, tmp_path: Path, size: int) -> None:
        """Test that calibration and ROC visualization functions construct their expected image files."""
        # Arrange
        obs = np.random.randint(0, 2, size)
        probs = np.random.rand(size)

        # Act
        plot_reliability_diagram(obs, probs, tmp_path)
        plot_roc_curve(obs, probs, tmp_path)

        # Assert
        assert (
            tmp_path / "reliability_diagram.png"
        ).exists(), "Reliability plot file was not created"
        assert (tmp_path / "roc_curve.png").exists(), "ROC plot file was not created"
