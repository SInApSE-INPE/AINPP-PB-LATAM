import matplotlib.pyplot as plt
import numpy as np
import os

def plot_comparison(target, prediction, output_path, title=None, cmap='viridis', diff_cmap='coolwarm'):
    """
    Plots a comparison of Target vs Prediction.
    
    Args:
        target (np.array): Target field (H, W).
        prediction (np.array): Prediction field (H, W).
        output_path (str): Path to save the plot.
        title (str): Title for the plot.
        cmap (str): Colormap for fields.
        diff_cmap (str): Colormap for difference.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Common vmin/vmax for consistent colorbar
    vmin = min(target.min(), prediction.min())
    vmax = max(target.max(), prediction.max())
    
    # 1. Target
    im0 = axes[0].imshow(target, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("Target")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 2. Prediction
    im1 = axes[1].imshow(prediction, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Error (Diff)
    error = prediction - target
    limit = max(abs(error.min()), abs(error.max()))
    im2 = axes[2].imshow(error, origin='lower', cmap=diff_cmap, vmin=-limit, vmax=limit)
    axes[2].set_title("Difference (Pred - Target)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    if title:
        plt.suptitle(title)
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
