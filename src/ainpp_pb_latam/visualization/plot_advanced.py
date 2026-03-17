import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def plot_barplots_and_rankings(df, output_dir, metric="CSI"):
    """
    Plots barplots comparing models and a ranking table.
    """
    if df.empty or 'model' not in df.columns or metric not in df['metric_name'].values:
        return
        
    df_metric = df[df['metric_name'] == metric].groupby('model')['mean'].mean().reset_index()
    df_metric = df_metric.sort_values('mean', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_metric, x='mean', y='model', palette='viridis')
    plt.title(f"Model Ranking by {metric} (Mean across all leads/thresholds)")
    plt.xlabel(f"Mean {metric}")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ranking_barplot_{metric.lower()}.png"))
    plt.close()

def plot_probability_histogram(probs, obs, output_dir):
    """
    Plots a probability histogram (Probabilistic Visualization).
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(probs[obs == 1], color='blue', label='Events (Obs=1)', kde=False, stat='density', alpha=0.5, bins=20)
    sns.histplot(probs[obs == 0], color='red', label='Non-Events (Obs=0)', kde=False, stat='density', alpha=0.5, bins=20)
    plt.title("Probability Histogram")
    plt.xlabel("Forecast Probability")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "probability_histogram.png"))
    plt.close()

def plot_pr_curve(obs, probs, output_dir):
    """
    Plots Precision-Recall curve.
    """
    from sklearn.metrics import precision_recall_curve, auc
    precision, recall, _ = precision_recall_curve(obs, probs)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall (POD)')
    plt.ylabel('Precision (Success Ratio)')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "pr_curve.png"))
    plt.close()

def plot_fss_by_scale(scales, fss_values, output_dir):
    """
    Plots FSS vs Scale (Spatial and multiscale visualization).
    """
    plt.figure(figsize=(8, 6))
    plt.plot(scales, fss_values, marker='o', linestyle='-', color='teal', label="FSS")
    plt.axhline(0.5, color='gray', linestyle='--', label="Useful Skill (FSS=0.5)")
    plt.title("Fraction Skill Score (FSS) by Spatial Scale")
    plt.xlabel("Scale (pixels/km)")
    plt.ylabel("FSS")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "fss_by_scale.png"))
    plt.close()

def plot_spatial_power_spectrum(target, pred, output_dir):
    """
    Plots radially averaged power spectrum.
    """
    def radial_profile(data, center):
        y, x = np.indices((data.shape))
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(int)
        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / np.maximum(nr, 1)
        return radialprofile
        
    f_t = np.fft.fft2(target)
    f_p = np.fft.fft2(pred)
    psd_t = np.abs(np.fft.fftshift(f_t))**2
    psd_p = np.abs(np.fft.fftshift(f_p))**2
    
    center = (target.shape[1]//2, target.shape[0]//2)
    rp_t = radial_profile(psd_t, center)
    rp_p = radial_profile(psd_p, center)
    
    plt.figure(figsize=(8, 6))
    plt.loglog(rp_t, label="Observed", color='blue')
    plt.loglog(rp_p, label="Forecast", color='orange')
    plt.title("Spatial Power Spectrum")
    plt.xlabel("Wavenumber k")
    plt.ylabel("Power Spectral Density")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "power_spectrum.png"))
    plt.close()

def plot_object_histograms(obj_areas_obs, obj_areas_pred, output_dir):
    """
    Plots object-based histogram distributions (Area).
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(obj_areas_obs, color='blue', label='Observed Objects', kde=True, stat='density', alpha=0.5, bins=15)
    sns.histplot(obj_areas_pred, color='red', label='Forecast Objects', kde=True, stat='density', alpha=0.5, bins=15)
    plt.title("Object Area Distribution")
    plt.xlabel("Area (pixels)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "object_area_histogram.png"))
    plt.close()

def plot_statistical_consistency(target, pred, output_dir):
    """
    Plots CDF and Exceedance Curves to check realism.
    """
    t_flat = np.sort(target.flatten())
    p_flat = np.sort(pred.flatten())
    
    # CDF
    plt.figure(figsize=(8, 6))
    plt.plot(t_flat, np.linspace(0, 1, len(t_flat)), color='blue', label="Observed")
    plt.plot(p_flat, np.linspace(0, 1, len(p_flat)), color='orange', label="Forecast")
    plt.title("Cumulative Distribution Function (CDF)")
    plt.xlabel("Precipitation Intensity (mm/h)")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cdf_consistency.png"))
    plt.close()
    
    # Exceedance Curve (1 - CDF)
    plt.figure(figsize=(8, 6))
    plt.semilogy(t_flat, 1.0 - np.linspace(0, 1, len(t_flat)), color='blue', label="Observed")
    plt.semilogy(p_flat, 1.0 - np.linspace(0, 1, len(p_flat)), color='orange', label="Forecast")
    plt.title("Exceedance Curve")
    plt.xlabel("Precipitation Intensity (mm/h)")
    plt.ylabel("Probability of Exceedance")
    plt.ylim(1e-5, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exceedance_curve.png"))
    plt.close()
