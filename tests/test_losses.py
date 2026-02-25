import torch
import pytest
import shutil
from ainpp.losses import get_loss, WeightedMSELoss, WeightedMAELoss, BalancedMSELoss, SSIMLoss, MS_SSIMLoss, PerceptualLoss, FocalLoss, CRPSLoss, HybridLoss
from omegaconf import OmegaConf

@pytest.fixture
def dummy_data():
    # Batch size 2, 1 channel, 32x32
    input = torch.randn(2, 1, 32, 32, requires_grad=True)
    target = torch.randn(2, 1, 32, 32)
    return input, target

def test_mse_loss(dummy_data):
    input, target = dummy_data
    conf = OmegaConf.create({'name': 'mse'})
    loss_fn = get_loss(conf)
    loss = loss_fn(input, target)
    assert loss.item() >= 0
    loss.backward()

def test_mae_loss(dummy_data):
    input, target = dummy_data
    conf = OmegaConf.create({'name': 'mae'})
    loss_fn = get_loss(conf)
    loss = loss_fn(input, target)
    assert loss.item() >= 0
    loss.backward()

def test_weighted_mse_loss(dummy_data):
    input, target = dummy_data
    conf = OmegaConf.create({'name': 'weighted_mse'})
    loss_fn = get_loss(conf)
    # Test with external weights
    weights = torch.rand_like(input)
    loss = loss_fn(input, target, weights=weights)
    assert loss.item() >= 0
    loss.backward()

def test_balanced_mse_loss(dummy_data):
    input, target = dummy_data
    conf = OmegaConf.create({'name': 'balanced_mse'})
    loss_fn = get_loss(conf)
    # Test with external weights
    weights = torch.rand_like(input)
    loss = loss_fn(input, target, weights=weights)
    assert loss.item() >= 0
    loss.backward()

def test_ssim_loss(dummy_data):
    input, target = dummy_data
    # SSIM requires input to be in [0, 1] for meaningful results usually, but loss works on any range
    conf = OmegaConf.create({'name': 'ssim'})
    loss_fn = get_loss(conf)
    loss = loss_fn(input, target)
    assert loss.item() >= 0 # Loss is 1 - SSIM. SSIM is <= 1.
    loss.backward()

def test_ms_ssim_loss(dummy_data):
    input, target = dummy_data
    conf = OmegaConf.create({'name': 'ms_ssim'})
    loss_fn = get_loss(conf)
    loss = loss_fn(input, target)
    assert loss.item() >= 0
    loss.backward()

# Skip perceptual if VGG not available or slow internet in test env
def test_perceptual_loss(dummy_data):
    input, target = dummy_data
    conf = OmegaConf.create({'name': 'perceptual'})
    loss_fn = get_loss(conf)
    if loss_fn.vgg_layers is None:
        pytest.skip("VGG weights not available")
    loss = loss_fn(input, target)
    assert loss.item() >= 0
    loss.backward()

def test_focal_loss():
    # Focal loss expects logits and indices for classification
    # Batch 2, 4 classes
    input = torch.randn(2, 4, requires_grad=True)
    target = torch.tensor([0, 3])
    conf = OmegaConf.create({'name': 'focal', 'alpha': 0.25, 'gamma': 2.0})
    loss_fn = get_loss(conf)
    loss = loss_fn(input, target)
    assert loss.item() >= 0
    loss.backward()

def test_crps_loss():
    # CRPS expects 2 channels: predictions and uncertainty (log variance)
    # Target is single channel
    input = torch.randn(2, 2, 32, 32, requires_grad=True)
    target = torch.randn(2, 1, 32, 32)
    conf = OmegaConf.create({'name': 'crps'})
    loss_fn = get_loss(conf)
    loss = loss_fn(input, target)
    assert isinstance(loss.item(), float)
    loss.backward()

def test_hybrid_loss(dummy_data):
    input, target = dummy_data
    conf = OmegaConf.create({
        'name': 'hybrid',
        'losses': [{'name': 'mse'}, {'name': 'mae'}],
        'weights': [0.5, 0.5]
    })
    loss_fn = get_loss(conf)
    loss = loss_fn(input, target)
    assert loss.item() >= 0
    loss.backward()
