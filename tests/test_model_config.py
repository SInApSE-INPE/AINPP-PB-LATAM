import torch
import pytest
from torch import nn
from omegaconf import OmegaConf
from src.models.unet import UNet, get_activation, get_norm

@pytest.fixture
def default_config():
    return OmegaConf.create({
        "in_channels": 1,
        "out_channels": 1,
        "features": [16, 32], # smaller for fast test
        "bilinear": True,
        "activation": "relu",
        "normalization": "batch",
        "dropout": 0.0,
        "kernel_size": 3,
        "padding": 1,
        "pooling": "max"
    })

def test_unet_instantiation(default_config):
    model = UNet(default_config)
    assert isinstance(model, UNet)
    # Check output shape
    x = torch.randn(1, 1, 64, 64)
    y = model(x)
    assert y.shape == (1, 1, 64, 64)

def test_custom_activation(default_config):
    default_config.activation = "leaky_relu"
    model = UNet(default_config)
    # Inspect a layer to verify modification (simplified check)
    # In DoubleConv, we have Conv-Norm-Act. 
    # inc layer is DoubleConv
    # inc.double_conv[2] should be activation if norm is present
    assert isinstance(model.inc.double_conv[2], nn.LeakyReLU)

def test_normalization_none(default_config):
    default_config.normalization = "none"
    model = UNet(default_config)
    # If norm is none, identity is used or skipped. 
    # Helper get_norm returns Identity.
    assert isinstance(model.inc.double_conv[1], nn.Identity)

def test_dropout(default_config):
    default_config.dropout = 0.5
    model = UNet(default_config)
    # DoubleConv structure: Conv, Norm, Act, Dropout
    # Check if Dropout is present
    has_dropout = any(isinstance(m, nn.Dropout2d) for m in model.inc.double_conv)
    assert has_dropout

def test_depth_change(default_config):
    default_config.features = [16, 32, 64]
    model = UNet(default_config)
    # Should have 2 down layers (plus inc which acts as first stage logic in internal lists? no inc is separate)
    # In my dynamic logic:
    # features=[16, 32, 64]
    # inc: 1->16
    # downs: 16->32, 32->64 (2 layers)
    # bottleneck: 64->128
    # ups: 128->64, 64->32, 32->16 (3 layers)
    assert len(model.downs) == 2
    assert len(model.ups) == 3

def test_pooling_avg(default_config):
    default_config.pooling = 'avg'
    model = UNet(default_config)
    # Check first down layer
    assert isinstance(model.downs[0].maxpool_conv[0], nn.AvgPool2d)
