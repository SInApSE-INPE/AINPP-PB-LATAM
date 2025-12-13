
import unittest
import os
import shutil
import json
import numpy as np
from src.visualization.runner import VisualizationRunner
from src.visualization.plot_metrics import plot_reliability_diagram, plot_roc_curve

class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_output_decoupled"
        self.input_dir = os.path.join(self.test_dir, "data")
        self.output_dir = os.path.join(self.test_dir, "plots")
        os.makedirs(self.input_dir, exist_ok=True)
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_runner_flow(self):
        # 1. Create Mock Data
        # Metrics
        summary = {
            "categorical": {
                "Thresh_0.5_POD": 0.8,
                "Thresh_0.5_SR": 0.7,
                "Thresh_0.5_FAR": 0.3
            }
        }
        with open(os.path.join(self.input_dir, "metrics.json"), 'w') as f:
            json.dump(summary, f)
            
        # Sample
        target = np.random.rand(5, 50, 50)
        prediction = np.random.rand(5, 50, 50)
        np.savez_compressed(
            os.path.join(self.input_dir, "sample_0.npz"), 
            target=target, 
            prediction=prediction
        )
        
        # 2. Run Runner
        config = {
            "style": {"context": "paper"},
            "maps": {"cmap": "viridis"},
            "animation": {"fps": 2}
        }
        runner = VisualizationRunner(config, self.input_dir, self.output_dir)
        runner.run()
        
        # 3. Verify Output
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "performance_diagram.png")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "sample_0_animation.gif")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "sample_0_comparison_last.png")))

    def test_plot_reliability_diagram(self):
        obs = np.random.randint(0, 2, 100)
        probs = np.random.rand(100)
        plot_reliability_diagram(obs, probs, self.input_dir) # Save to input dir for test
        self.assertTrue(os.path.exists(os.path.join(self.input_dir, "reliability_diagram.png")))

    def test_plot_roc_curve(self):
        obs = np.random.randint(0, 2, 100)
        probs = np.random.rand(100)
        plot_roc_curve(obs, probs, self.input_dir)
        self.assertTrue(os.path.exists(os.path.join(self.input_dir, "roc_curve.png")))

if __name__ == '__main__':
    unittest.main()
