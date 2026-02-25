

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from ainpp.evaluation.metrics import Metrics
from ainpp.visualization import VisualizationRunner

class Evaluator:
    def __init__(self, model, test_loader, config, standardizer, device='cpu'):
        self.model = model
        self.loader = test_loader
        self.config = config
        self.standardizer = standardizer
        self.device = device
        
        # Directories
        self.base_output_dir = os.path.join(os.getcwd(), "outputs", "evaluation")
        self.data_dir = os.path.join(self.base_output_dir, "data")
        self.plots_dir = os.path.join(self.base_output_dir, "plots")
        os.makedirs(self.data_dir, exist_ok=True)
        # Plots dir will be created by runner
        
        # Categorical thresholds (mm/h)
        self.thresholds = config.get("evaluation", {}).get("thresholds", [0.5, 2.0, 5.0])
        
        self.viz_config = config.get("visualization", {})

    def evaluate(self):
        self.model.eval()
        self.model.to(self.device)
        
        # Accumulators
        all_metrics = {
            "continuous": [],
            "categorical": [],
            "probabilistic": []
        }
        
        print(f"Starting evaluation on {len(self.loader)} batches...")
        
        # Limit batches for testing if configured
        max_batches = self.config.get("evaluation", {}).get("max_batches", None)
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(self.loader, desc="Evaluating")):
                if max_batches is not None and batch_idx >= max_batches:
                    print(f"Reached limit of {max_batches} batches. Stopping.")
                    break
                    
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Inference
                output = self.model(data)
                
                # Inverse Transform
                output_mmh = self.standardizer.inverse_transform(output)
                target_mmh = self.standardizer.inverse_transform(target)
                
                # Convert to numpy
                if hasattr(output_mmh, 'cpu'):
                    output_np = output_mmh.cpu().numpy()
                else:
                    output_np = output_mmh
                    
                if hasattr(target_mmh, 'cpu'):
                    target_np = target_mmh.cpu().numpy()
                else:
                    target_np = target_mmh

                # --- SAVE SAMPLES (First Batch Only) ---
                if batch_idx == 0:
                    self.save_samples(output_np, target_np, batch_idx)
                
                # 1. Continuous Metrics
                cont_metrics = Metrics.compute_continuous_metrics(output_np, target_np)
                all_metrics["continuous"].append(cont_metrics)
                
                # 2. Categorical Metrics
                cat_metrics = Metrics.compute_categorical_metrics(output_np, target_np, self.thresholds)
                all_metrics["categorical"].append(cat_metrics)
                
                # 3. Probabilistic Metrics
                prob_metrics = {}
                is_ensemble = output_np.ndim > 3 and output_np.shape[1] > 1
                
                if is_ensemble:
                    crps = Metrics.compute_crps(output_np.swapaxes(0, 1), target_np) 
                    prob_metrics["CRPS"] = crps
                else:
                    prob_metrics["CRPS"] = cont_metrics["MAE"]
                
                all_metrics["probabilistic"].append(prob_metrics)

        # Aggregate and Save
        summary = self.aggregate_metrics(all_metrics)
        self.save_results(summary)
        
        # Run Visualization
        self.run_visualization()
        
        return summary

    def save_samples(self, output, target, batch_idx):
        """Saves sample data to .npz for visualization."""
        print(f"Saving samples for batch {batch_idx}...")
        
        # Simplify input: take first sample in batch for now
        # If output is (B, T, H, W) -> (T, H, W)
        pred_seq = output[0] 
        target_seq = target[0]
        
        # Handle Ensemble: (M, T, H, W) -> mean -> (T, H, W) for viz saving
        # Saving full ensemble might be too heavy? Let's save mean for now.
        if pred_seq.ndim == 4: 
             pred_seq = np.mean(pred_seq, axis=0)

        path = os.path.join(self.data_dir, f"sample_{batch_idx}.npz")
        np.savez_compressed(path, target=target_seq, prediction=pred_seq)
        print(f"Saved sample to {path}")

    def aggregate_metrics(self, metrics_dict):
        summary = {}
        for category, list_of_dicts in metrics_dict.items():
            if not list_of_dicts: continue
            
            keys = list_of_dicts[0].keys()
            cat_summary = {}
            for k in keys:
                values = [d[k] for d in list_of_dicts if k in d and isinstance(d[k], (int, float))]
                if values:
                    cat_summary[k] = float(np.mean(values))
            
            summary[category] = cat_summary
        return summary

    def save_results(self, summary):
        path = os.path.join(self.data_dir, "metrics.json")
        with open(path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Metrics saved to {path}")

    def run_visualization(self):
        print("Running visualization pipeline...")
        runner = VisualizationRunner(
            config=self.viz_config,
            input_dir=self.data_dir,
            output_dir=self.plots_dir
        )
        runner.run()

