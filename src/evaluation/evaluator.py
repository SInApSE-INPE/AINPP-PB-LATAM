import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from .metrics import Metrics

class Evaluator:
    def __init__(self, model, test_loader, config, standardizer, device='cpu'):
        self.model = model
        self.loader = test_loader
        self.config = config
        self.standardizer = standardizer
        self.device = device
        self.output_dir = os.path.join(os.getcwd(), "outputs", "evaluation")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Categorical thresholds (mm/h)
        self.thresholds = config.get("evaluation", {}).get("thresholds", [0.5, 2.0, 5.0])

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
                # Assuming model returns (Batch, Time, H, W) or (Batch, C, H, W)
                output = self.model(data)
                
                # Inverse Transform
                # Target is likely standardized too? Usually dataset returns standardized.
                # If target is already mm/h, skip inverse. Assuming standardized based on dataset.py
                output_mmh = self.standardizer.inverse_transform(output)
                target_mmh = self.standardizer.inverse_transform(target)
                
                # Convert to numpy for metrics (Standardizer returns numpy usually)
                if hasattr(output_mmh, 'cpu'):
                    output_np = output_mmh.cpu().numpy()
                else:
                    output_np = output_mmh
                    
                if hasattr(target_mmh, 'cpu'):
                    target_np = target_mmh.cpu().numpy()
                else:
                    target_np = target_mmh
                
                # 1. Continuous Metrics
                cont_metrics = Metrics.compute_continuous_metrics(output_np, target_np)
                all_metrics["continuous"].append(cont_metrics)
                
                # 2. Categorical Metrics (per threshold)
                cat_metrics = Metrics.compute_categorical_metrics(output_np, target_np, self.thresholds)
                all_metrics["categorical"].append(cat_metrics)
                
                # 3. Probabilistic Metrics
                # Check if we have probabilities or ensemble
                # For now, if output has dimension > 1 for channels/members, treat as ensemble
                # If model is UNet with 1 channel, it's deterministic.
                # Assuming output shape (B, C, H, W). If C > 1, treating as ensemble members.
                prob_metrics = {}
                is_ensemble = output_np.ndim > 3 and output_np.shape[1] > 1
                
                if is_ensemble:
                    # CRPS
                    crps = Metrics.compute_crps(output_np.swapaxes(0, 1), target_np) # (M, B, ...)
                    prob_metrics["CRPS"] = crps
                else:
                    # Deterministic -> CRPS == MAE
                    # For ROC/Brier, we need probabilities. 
                    # If we don't have them, we can't really compute valid Brier/ROC without a prob model.
                    # We skip them or compute on binary?
                    # Let's skip ROC/Brier for deterministic to avoid confusion, or assume binary 0/1 (bad practice).
                    # User asked for them, so maybe they expect it.
                    prob_metrics["CRPS"] = cont_metrics["MAE"]

                # For Brier/ROC, we need P(y > thresh).
                # If ensemble: Mean of (members > thresh).
                # If deterministic: (pred > thresh).astype(float) -- degenerate.
                
                for thresh in self.thresholds:
                    if is_ensemble:
                        # Prob = Fraction of members > thresh
                        probs = (output_np > thresh).mean(axis=1) # Average over channel dim
                    else:
                        probs = (output_np > thresh).astype(float).squeeze(1) # Deterministic binary prob
                    
                    target_bin = (target_np > thresh).astype(int).squeeze(1) # Remove channel dim if 1
                    
                    pm = Metrics.compute_probabilistic_metrics(probs, target_bin, thresh)
                    # Prefix keys
                    for k, v in pm.items():
                        if not isinstance(v, (np.ndarray, list)): # Don't average arrays
                            prob_metrics[f"Thresh_{thresh}_{k}"] = v
                
                all_metrics["probabilistic"].append(prob_metrics)

        # Aggregate and Save
        summary = self.aggregate_metrics(all_metrics)
        self.save_results(summary)
        self.plot_diagrams(summary)
        
        return summary

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
        path = os.path.join(self.output_dir, "metrics.json")
        with open(path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Metrics saved to {path}")

    def plot_diagrams(self, summary):
        # Performance Diagram (Success Ratio vs POD)
        # We need aggregated POD and SR (1-FAR) for each threshold
        if "categorical" in summary:
            plt.figure(figsize=(8, 8))
            cat = summary["categorical"]
            pods = []
            srs = []
            labels = []
            
            for thresh in self.thresholds:
                pod_key = f"Thresh_{thresh}_POD"
                sr_key = f"Thresh_{thresh}_SR"
                if pod_key in cat and sr_key in cat:
                    pods.append(cat[pod_key])
                    srs.append(cat[sr_key])
                    labels.append(f"{thresh} mm/h")
            
            if pods:
                plt.scatter(srs, pods, c='red')
                for i, txt in enumerate(labels):
                    plt.annotate(txt, (srs[i], pods[i]))
                
                # Plot CSI curves
                x = np.linspace(0.01, 1, 100)
                for csi in [0.2, 0.4, 0.6, 0.8]:
                    # CSI = 1 / (1/SR + 1/POD - 1)
                    # y = 1 / (1/csi - 1/x + 1)
                    y = 1 / (1/csi - 1/x + 1)
                    plt.plot(x, y, '--', color='gray', linewidth=0.5)
                    plt.text(x[-1], y[-1], f"CSI={csi}", fontsize=8)

                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.xlabel("Success Ratio (1-FAR)")
                plt.ylabel("Probability of Detection (POD)")
                plt.title("Performance Diagram")
                plt.grid(True)
                plt.savefig(os.path.join(self.output_dir, "performance_diagram.png"))
                plt.close()
