import torch
import pytest

from ainpp.utils import EarlyStopping, build_optimizer
from ainpp._utils.standardization import LogZScoreStandardizer


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


def test_early_stopping_triggers(tmp_path):
    model = _ToyModel()
    checkpoint = tmp_path / "best.pt"
    stopper = EarlyStopping(patience=2, delta=0.0, path=checkpoint, enabled=True)

    stopper(1.0, model)  # baseline
    stopper(0.9, model)  # improvement saves checkpoint
    assert checkpoint.exists()

    stopper(1.0, model)  # worse (1)
    stopper(1.1, model)  # worse (2) -> should stop
    assert stopper.early_stop


def test_build_optimizer_respects_lr():
    model = _ToyModel()
    cfg = {"lr": 0.005}
    opt = build_optimizer(model.parameters(), cfg)
    assert opt.defaults["lr"] == cfg["lr"]


def test_log_zscore_roundtrip():
    std = LogZScoreStandardizer(mean_log=1.0, std_log=0.5)
    values = std.transform([0.0, 1.0, 2.0])
    recovered = std.inverse_transform(values)
    assert recovered.shape == values.shape
    assert recovered[0] == pytest.approx(0.0, rel=1e-6, abs=1e-6)
