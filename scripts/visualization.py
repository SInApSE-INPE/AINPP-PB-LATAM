
import sys
sys.path.insert(0, "/prj/ideeps/adriano.almeida/benchmark")
import os
import shutil
import json
import numpy as np
from ainpp.visualization.runner import VisualizationRunner
from ainpp.visualization.plot_metrics import plot_reliability_diagram, plot_roc_curve


test_dir = "./tests/test_output"
input_dir = os.path.join(test_dir, "data")
output_dir = os.path.join(test_dir, "plots")
os.makedirs(input_dir, exist_ok=True)

def test_runner_flow():
    # 1. Create Mock Data
    # Metrics
    summary = {
        "categorical": {
            "Thresh_0.5_POD": 0.8,
            "Thresh_0.5_SR": 0.7,
            "Thresh_0.5_FAR": 0.3
        }
    }
    with open(os.path.join(input_dir, "metrics.json"), 'w') as f:
        json.dump(summary, f)
        
    # Sample
    target = np.random.rand(5, 50, 50)
    prediction = np.random.rand(5, 50, 50)
    np.savez_compressed(
        os.path.join(input_dir, "sample_0.npz"), 
        target=target, 
        prediction=prediction
    )
    
    # 2. Run Runner
    config = {
        "style": {"context": "paper"},
        "maps": {"cmap": "viridis"},
        "animation": {"fps": 2}
    }
    runner = VisualizationRunner(config, input_dir, output_dir)
    runner.run()
    
    # 3. Verify Output
    assert os.path.exists(os.path.join(output_dir, "performance_diagram.png"))
    assert os.path.exists(os.path.join(output_dir, "sample_0_animation.gif"))
    assert os.path.exists(os.path.join(output_dir, "sample_0_comparison_last.png"))

def test_plot_reliability_diagram():
    obs = np.random.randint(0, 2, 100)
    probs = np.random.rand(100)
    plot_reliability_diagram(obs, probs, input_dir) # Save to input dir for test
    assert os.path.exists(os.path.join(input_dir, "reliability_diagram.png"))

def test_plot_roc_curve():
    obs = np.random.randint(0, 2, 100)
    probs = np.random.rand(100)
    plot_roc_curve(obs, probs, input_dir)
    assert os.path.exists(os.path.join(input_dir, "roc_curve.png"))

def main():
    test_runner_flow()
    test_plot_reliability_diagram()
    test_plot_roc_curve()


if __name__ == "__main__":
    main()
