import unittest
import torch
from omegaconf import OmegaConf
from src.models.factory import get_model
from src.models.unet import UNet

class TestModels(unittest.TestCase):
    def test_unet_initialization(self):
        # Test direct init
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
        self.assertIsInstance(model, torch.nn.Module)
        
    def test_factory_get_model(self):
        conf = OmegaConf.create({
            "model": {
                "name": "unet",
                "in_channels": 4,
                "out_channels": 1,
                "bilinear": False
            }
        })
        model = get_model(conf)
        self.assertIsInstance(model, UNet)
        self.assertEqual(model.n_channels, 4)
        self.assertEqual(model.n_classes, 1)
        self.assertEqual(model.bilinear, False)

    def test_unet_forward_pass(self):
        model = UNet(n_channels=1, n_classes=1)
        # B, C, H, W
        x = torch.randn(1, 1, 64, 64) 
        output = model(x)
        self.assertEqual(output.shape, (1, 1, 64, 64))

if __name__ == '__main__':
    unittest.main()
