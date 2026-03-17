import numpy as np

from ainpp_pb_latam.visualization.plot_metrics import plot_reliability_diagram, plot_roc_curve


def test_reliability_and_roc_outputs(tmp_path):
    obs = np.random.randint(0, 2, 50)
    probs = np.random.rand(50)

    plot_reliability_diagram(obs, probs, tmp_path)
    plot_roc_curve(obs, probs, tmp_path)

    assert (tmp_path / "reliability_diagram.png").exists()
    assert (tmp_path / "roc_curve.png").exists()
