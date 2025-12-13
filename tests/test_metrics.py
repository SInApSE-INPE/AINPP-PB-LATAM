import unittest
import numpy as np
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.metrics import Metrics
from src.utils.standardization import LogZScoreStandardizer

class TestMetrics(unittest.TestCase):
    def test_continuous_metrics(self):
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.0, 4.0])
        
        # MSE = ((0)^2 + (0)^2 + (-1)^2)/3 = 0.333
        # MAE = (0 + 0 + 1)/3 = 0.333
        
        metrics = Metrics.compute_continuous_metrics(pred, target)
        self.assertAlmostEqual(metrics['MSE'], 1.0/3.0)
        self.assertAlmostEqual(metrics['MAE'], 1.0/3.0)
        self.assertAlmostEqual(metrics['R2'], 0.7857, places=3) 

    def test_categorical_metrics(self):
        # Threshold 1.0
        # Pred:   [0.5, 1.5, 2.5, 0.5] -> [0, 1, 1, 0]
        # Target: [0.5, 0.5, 2.5, 3.5] -> [0, 0, 1, 1]
        
        pred = np.array([0.5, 1.5, 2.5, 0.5])
        target = np.array([0.5, 0.5, 2.5, 3.5])
        thresholds = [1.0]
        
        # TP: idx 2 (1,1) -> 1
        # FP: idx 1 (1,0) -> 1
        # FN: idx 3 (0,1) -> 1
        # TN: idx 0 (0,0) -> 1
        
        # POD = TP/(TP+FN) = 1/2 = 0.5
        # FAR = FP/(TP+FP) = 1/2 = 0.5
        # TS = TP/(TP+FN+FP) = 1/3 = 0.333
        
        metrics = Metrics.compute_categorical_metrics(pred, target, thresholds)
        self.assertAlmostEqual(metrics['Thresh_1.0_POD'], 0.5, places=5)
        self.assertAlmostEqual(metrics['Thresh_1.0_FAR'], 0.5, places=5)
        self.assertAlmostEqual(metrics['Thresh_1.0_TS'], 1.0/3.0, places=5)

    def test_probabilistic_metrics(self):
        # Brier Score
        # Pred Probs: [0.2, 0.8]
        # Target:     [0, 1]
        # BS = ((0.2-0)^2 + (0.8-1)^2)/2 = (0.04 + 0.04)/2 = 0.04
        
        pred_probs = np.array([0.2, 0.8])
        target_bin = np.array([0, 1])
        
        metrics = Metrics.compute_probabilistic_metrics(pred_probs, target_bin)
        self.assertAlmostEqual(metrics['BrierScore'], 0.04)

class TestStandardization(unittest.TestCase):
    def test_log_zscore(self):
        mean_log = 1.0
        std_log = 0.5
        std = LogZScoreStandardizer(mean_log=mean_log, std_log=std_log)
        
        # Test Inverse
        # z = 0 -> x_log = 1.0 -> x = exp(1) - 1
        z = np.array([0.0])
        x = std.inverse_transform(z)
        expected = np.expm1(1.0)
        self.assertAlmostEqual(x[0], expected)
        
        # Test Forward
        z_rec = std.transform(x)
        self.assertAlmostEqual(z_rec[0], 0.0)

if __name__ == '__main__':
    unittest.main()
