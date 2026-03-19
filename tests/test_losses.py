import pytest
import torch
import torch.nn.functional as F

from ainpp_pb_latam.losses import HybridLoss, LogCoshLoss, SSIMLoss, WeightedMSELoss


class TestWeightedMSELoss:
    """Test suite for the WeightedMSELoss function."""

    @pytest.mark.parametrize("alpha, threshold", [(2.0, 0.0), (1.5, 2.0)])
    def test_weighted_mse_increases_loss_when_target_high(
        self, tensor_pair: tuple[torch.Tensor, torch.Tensor], alpha: float, threshold: float
    ) -> None:
        """Test that WeightedMSELoss produces higher loss than standard MSE for targets above threshold."""
        # Arrange
        x, y = tensor_pair
        y_high = y + 5.0  # amplify targets to guarantee they exceed the threshold

        # Act
        loss_plain = F.mse_loss(x, y_high)
        loss_weighted = WeightedMSELoss(alpha=alpha, threshold=threshold)(x, y_high)

        # Assert
        assert loss_weighted.item() > loss_plain.item()


class TestLogCoshLoss:
    """Test suite for the LogCoshLoss function."""

    def test_logcosh_behaves_like_mse_for_small_errors(
        self, tensor_pair: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test that Log-Cosh loss behaves similarly to MSE for small residuals."""
        # Arrange
        x, _ = tensor_pair
        y = x + 1e-3  # small error

        # Act
        loss = LogCoshLoss()(x, y)
        approx = F.mse_loss(x, y)

        # Assert
        assert torch.isfinite(loss)
        assert loss.item() <= approx.item() * 2  # Log-cosh approximation bound for small x


class TestSSIMLoss:
    """Test suite for the SSIMLoss function."""

    def test_ssim_accepts_5d_and_returns_in_valid_range(self) -> None:
        """Test that SSIM accepts 5D video tensors and returns a bounded metric between 0 and 2."""
        # Arrange
        x = torch.rand(2, 3, 1, 8, 8)
        y = torch.rand(2, 3, 1, 8, 8)

        # Act
        loss = SSIMLoss()(x, y)

        # Assert
        assert 0.0 <= loss.item() <= 2.0


class TestHybridLoss:
    """Test suite for the HybridLoss combined criterion."""

    def test_hybrid_loss_combines_losses_and_computes_gradients(
        self, tensor_pair: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test that HybridLoss effectively aggregates sub-losses and allows backpropagation."""
        # Arrange
        x, y = tensor_pair
        x.requires_grad_(True)

        losses = [WeightedMSELoss(alpha=0.0), LogCoshLoss()]
        weights = [0.7, 0.3]
        hybrid = HybridLoss(losses=losses, weights=weights)

        # Act
        value = hybrid(x, y)

        # Assert
        assert value.item() >= 0

        # Ensure differentiation works correctly without throwing exceptions
        value.backward()
        assert x.grad is not None
