import torch
import pytest

from ainpp_pb_latam.losses import HybridLoss, LogCoshLoss, SSIMLoss, WeightedMSELoss


def test_weighted_mse_increases_loss_when_target_high(tensor_pair):
    x, y = tensor_pair
    y_high = y + 5.0  # amplify targets to trigger weighting
    loss_plain = torch.nn.functional.mse_loss(x, y_high)
    loss_weighted = WeightedMSELoss(alpha=2.0, threshold=0.0)(x, y_high)
    assert loss_weighted.item() > loss_plain.item()


def test_logcosh_behaves_like_mse_for_small_errors(tensor_pair):
    x, _ = tensor_pair
    y = x + 1e-3
    loss = LogCoshLoss()(x, y)
    approx = torch.nn.functional.mse_loss(x, y)
    assert torch.isfinite(loss)
    # Allow slack: logcosh should be same order as MSE for small errors.
    assert loss.item() <= approx.item() * 2


def test_ssim_accepts_5d_and_returns_in_0_2_range():
    x = torch.rand(2, 3, 1, 8, 8)
    y = torch.rand(2, 3, 1, 8, 8)
    loss = SSIMLoss()(x, y)
    assert 0.0 <= loss.item() <= 2.0


def test_hybrid_loss_combines_losses(tensor_pair):
    x, y = tensor_pair
    x.requires_grad_(True)
    losses = [WeightedMSELoss(alpha=0.0), LogCoshLoss()]
    weights = [0.7, 0.3]
    hybrid = HybridLoss(losses=losses, weights=weights)
    value = hybrid(x, y)
    assert value.item() >= 0
    value.backward()
