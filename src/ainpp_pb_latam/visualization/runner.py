import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from .animations import create_animation
from .plot_maps import plot_comparison
from .plot_metrics import plot_performance_diagram, plot_reliability_diagram, plot_roc_curve
from .style import set_style


class VisualizationRunner:
    def __init__(self, config, input_dir, output_dir):
        """
        Args:
            config (DictConfig): Visualization configuration.
            input_dir (str): Directory containing metrics.json and *.npz samples.
            output_dir (str): Directory to save plots.
        """
        self.config = config
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        set_style(self.config.get("style", {}))

    def run(self):
        print(f"Starting visualization runner...")
        print(f"Input: {self.input_dir}")
        print(f"Output: {self.output_dir}")

        # 1. Metrics
        self.plot_metrics()

        # 2. Samples
        self.plot_samples()

    def plot_metrics(self):
        metrics_path = os.path.join(self.input_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            print(f"No metrics.json found at {metrics_path}")
            return

        with open(metrics_path, "r") as f:
            summary = json.load(f)

        # Performance Diagram
        # We need thresholds. If not in config, infer or skip?
        # The performance diagram function assumes thresholds are keys in the summary.
        # We can pass a dummy list or extract from keys?
        # Ideally, thresholds should be stored in metrics.json or config.
        # For now, let's look at stored keys in "categorical".

        thresholds = []
        if "categorical" in summary:
            for key in summary["categorical"].keys():
                if "Thresh_" in key and "_POD" in key:
                    # Parse "Thresh_0.5_POD"
                    try:
                        t = float(key.split("_")[1])
                        if t not in thresholds:
                            thresholds.append(t)
                    except:
                        pass
        thresholds.sort()

        if thresholds:
            plot_performance_diagram(summary, self.output_dir, thresholds)

    def plot_samples(self):
        sample_files = glob.glob(os.path.join(self.input_dir, "sample_*.npz"))
        if not sample_files:
            print("No sample .npz files found.")
            return

        maps_config = self.config.get("maps", {})
        anim_config = self.config.get("animation", {})

        for spath in sample_files:
            try:
                data = np.load(spath)
                target_seq = data["target"]
                pred_seq = data["prediction"]

                # Derive base name: sample_0.npz -> sample_0
                base_name = os.path.splitext(os.path.basename(spath))[0]

                # Animation
                create_animation(
                    target_seq,
                    pred_seq,
                    os.path.join(self.output_dir, f"{base_name}_animation.gif"),
                    fps=anim_config.get("fps", 5),
                    cmap=maps_config.get("cmap", "viridis"),
                )

                # Comparison (Last Step)
                plot_comparison(
                    target_seq[-1],
                    pred_seq[-1],
                    os.path.join(self.output_dir, f"{base_name}_comparison_last.png"),
                    title=f"{base_name} (Last Step)",
                    cmap=maps_config.get("cmap", "viridis"),
                    diff_cmap=maps_config.get("diff_cmap", "coolwarm"),
                )

            except Exception as e:
                print(f"Failed to process {spath}: {e}")
