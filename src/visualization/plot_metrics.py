import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import os

def plot_performance_diagram(summary, output_dir, thresholds):
    """
    Plots a Performance Diagram (Success Ratio vs POD) with CSI isolines.
    """
    if "categorical" not in summary:
        return

    plt.figure(figsize=(8, 8))
    cat = summary["categorical"]
    pods = []
    srs = []
    labels = []
    
    for thresh in thresholds:
        pod_key = f"Thresh_{thresh}_POD"
        sr_key = f"Thresh_{thresh}_SR" # Success Ratio = 1 - FAR
        
        # Check standard keys or compute from FAR
        if pod_key in cat:
            pod = cat[pod_key]
            if sr_key in cat:
                sr = cat[sr_key]
            elif f"Thresh_{thresh}_FAR" in cat:
                sr = 1 - cat[f"Thresh_{thresh}_FAR"]
            else:
                continue
                
            pods.append(pod)
            srs.append(sr)
            labels.append(f"{thresh} mm/h")
    
    if not pods:
        return

    # Plot CSI curves (background)
    x = np.linspace(0.01, 1, 100)
    for csi in [0.2, 0.4, 0.6, 0.8]:
        # CSI = 1 / (1/SR + 1/POD - 1)
        # y = 1 / (1/csi - 1/x + 1)
        y = 1 / (1/csi - 1/x + 1)
        # Filter valid y (0 to 1)
        mask = (y >= 0) & (y <= 1)
        plt.plot(x[mask], y[mask], '--', color='gray', linewidth=0.5)
        if mask.any():
             # Basic label placement
             plt.text(x[mask][-1], y[mask][-1], f"CSI={csi}", fontsize=8, color='gray')

    # Plot points
    plt.scatter(srs, pods, c='red', zorder=5)
    for i, txt in enumerate(labels):
        plt.annotate(txt, (srs[i], pods[i]), xytext=(5, 5), textcoords='offset points')

    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.xlabel("Success Ratio (1 - FAR)")
    plt.ylabel("Probability of Detection (POD)")
    plt.title("Performance Diagram")
    plt.grid(True)
    
    path = os.path.join(output_dir, "performance_diagram.png")
    plt.savefig(path)
    plt.close()
    print(f"Performance diagram saved to {path}")

def plot_reliability_diagram(obs, probs, output_dir, n_bins=10):
    """
    Plots a Reliability Diagram.
    
    Args:
        obs (np.array): Binary observations (0 or 1).
        probs (np.array): Forecast probabilities.
        output_dir (str): output directory.
        n_bins (int): Number of bins.
    """
    # Simple implementation or use sklearn.calibration.calibration_curve
    from sklearn.calibration import calibration_curve
    
    prob_true, prob_pred = calibration_curve(obs, probs, n_bins=n_bins)
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(prob_pred, prob_true, "s-", label="Model")
    
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True)
    
    path = os.path.join(output_dir, "reliability_diagram.png")
    plt.savefig(path)
    plt.close()
    print(f"Reliability diagram saved to {path}")

def plot_roc_curve(obs, probs, output_dir):
    """
    Plots ROC Curve.
    
    Args:
        obs (np.array): Binary observations.
        probs (np.array): Forecast probabilities.
    """
    fpr, tpr, _ = roc_curve(obs, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(path)
    plt.close()
    print(f"ROC curve saved to {path}")
