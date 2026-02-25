import unittest
import shutil
import os
import torch
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf
from ainpp.trainer import Trainer

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.abspath("tests/test_training_output")
        os.makedirs(self.test_dir, exist_ok=True)
        # Mock MLFlow to avoid creating run artifacts
        self.mlflow_patcher = patch('ainpp.trainer.mlflow')
        self.mock_mlflow = self.mlflow_patcher.start()
        
        # Force CPU
        self.cuda_patcher = patch('torch.cuda.is_available', return_value=False)
        self.cuda_patcher.start()

    def tearDown(self):
        self.cuda_patcher.stop()
        self.mlflow_patcher.stop()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def test_train_epoch(self):
        # Mock Model
        # Input (B, C, H, W) -> Output (B, C, H, W)
        model = MagicMock(spec=torch.nn.Module)
        model.parameters.return_value = [torch.randn(1, requires_grad=True)]
        model.return_value = torch.randn(2, 1, 10, 10, requires_grad=True)
        
        # Mock Loaders
        # Batch of (Data, Target)
        train_loader = [
            (torch.randn(2, 1, 10, 10), torch.randn(2, 1, 10, 10)),
            (torch.randn(2, 1, 10, 10), torch.randn(2, 1, 10, 10))
        ]
        val_loader = [
            (torch.randn(2, 1, 10, 10), torch.randn(2, 1, 10, 10))
        ]
        
        conf = OmegaConf.create({
            "training": {
                "distributed": False,
                "device": "cpu",
                "learning_rate": 0.001,
                "epochs": 1,
                "experiment_name": "test_exp",
                "loss": {"name": "mse"}
            },
            "model": {},
            "dataset": {}
        })
        
        trainer = Trainer(model, train_loader, val_loader, conf)
        
        # Test 1 Epoch Train
        loss = trainer.train_epoch(0)
        self.assertIsInstance(loss, float)
        self.assertTrue(model.train.called)
        
        # Test Validation
        val_loss = trainer.validate(0)
        self.assertIsInstance(val_loss, float)
        self.assertTrue(model.eval.called)

    def test_fit_loop(self):
        # Mock Model
        model = torch.nn.Linear(1, 1) # simple real model to allow backward
        
        # Mock Loaders
        train_loader = [
            (torch.randn(2, 1), torch.randn(2, 1))
        ]
        val_loader = [
            (torch.randn(2, 1), torch.randn(2, 1))
        ]
        
        conf = OmegaConf.create({
            "training": {
                "distributed": False,
                "device": "cpu",
                "learning_rate": 0.001,
                "epochs": 2,
                "experiment_name": "test_fit",
                "loss": {"name": "mse"}
            },
            "model": {},
            "dataset": {}
        })
        
        trainer = Trainer(model, train_loader, val_loader, conf)
        
        # Override output dir logic in trainer?
        # Trainer writes to "outputs/checkpoint...". 
        # We can't easily inject the path unless we change the trainer code or chdir.
        # Let's chdir temporarily or just allow it and cleanup 'outputs' if created in CWD.
        
        cwd = os.getcwd()
        try:
            os.chdir(self.test_dir)
            trainer.fit()
            
            self.assertTrue(os.path.exists("outputs/checkpoint_epoch_1.pth"))
            self.assertTrue(os.path.exists("outputs/checkpoint_epoch_2.pth"))
            
        finally:
            os.chdir(cwd)
            
        self.assertTrue(self.mock_mlflow.start_run.called)
        self.assertTrue(self.mock_mlflow.end_run.called)

if __name__ == '__main__':
    unittest.main()
