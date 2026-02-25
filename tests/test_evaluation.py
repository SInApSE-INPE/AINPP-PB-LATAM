import unittest
import os
import shutil
import json
import numpy as np
import torch
from unittest.mock import MagicMock
from omegaconf import OmegaConf
from ainpp.evaluation.metrics import Metrics
from ainpp.evaluation.evaluator import Evaluator
from ainpp.utils.standardization import LogZScoreStandardizer

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_eval_output"
        os.makedirs(self.test_dir, exist_ok=True)
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    # --- Metrics Tests ---
    def test_continuous_metrics(self):
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.0, 4.0])
        metrics = Metrics.compute_continuous_metrics(pred, target)
        self.assertAlmostEqual(metrics['MSE'], 1.0/3.0)
        self.assertAlmostEqual(metrics['MAE'], 1.0/3.0)

    def test_categorical_metrics(self):
        pred = np.array([0.5, 1.5, 2.5, 0.5])
        target = np.array([0.5, 0.5, 2.5, 3.5])
        thresholds = [1.0]
        metrics = Metrics.compute_categorical_metrics(pred, target, thresholds)
        self.assertAlmostEqual(metrics['Thresh_1.0_POD'], 0.5, places=5)

    def test_probabilistic_metrics(self):
        pred_probs = np.array([0.2, 0.8])
        target_bin = np.array([0, 1])
        metrics = Metrics.compute_probabilistic_metrics(pred_probs, target_bin)
        self.assertAlmostEqual(metrics['BrierScore'], 0.04)

    # --- Evaluator Tests ---
    def test_evaluator_flow(self):
        # Mocking components
        model = MagicMock()
        model.eval.return_value = None
        model.to.return_value = model
        # Output shape: B, C, H, W. Standardizer inverse takes tensor, returns tensor/np
        model.return_value = torch.randn(2, 1, 10, 10) 
        
        loader = [
            (torch.randn(2, 1, 10, 10), torch.randn(2, 1, 10, 10))
        ] # 1 Batch
        
        conf = OmegaConf.create({
            "evaluation": {
                "thresholds": [0.5],
                "max_batches": 1
            },
            "visualization": {
                "style": {},
                "maps": {},
                "animation": {}
            }
        })
        
        standardizer = MagicMock()
        standardizer.inverse_transform.side_effect = lambda x: x # Identity for mock
        
        evaluator = Evaluator(model, loader, conf, standardizer)
        
        # Override data dir for test
        evaluator.data_dir = os.path.join(self.test_dir, "data")
        evaluator.plots_dir = os.path.join(self.test_dir, "plots")
        os.makedirs(evaluator.data_dir, exist_ok=True)
        
        summary = evaluator.evaluate()
        
        # Check files
        self.assertTrue(os.path.exists(os.path.join(evaluator.data_dir, "metrics.json")))
        self.assertTrue(os.path.exists(os.path.join(evaluator.data_dir, "sample_0.npz")))
        self.assertTrue(os.path.exists(os.path.join(evaluator.plots_dir, "performance_diagram.png")))

if __name__ == '__main__':
    unittest.main()
