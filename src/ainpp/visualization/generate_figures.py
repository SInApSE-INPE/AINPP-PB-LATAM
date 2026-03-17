import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

def generate_benchmark_figures(df: pd.DataFrame, output_dir: str):
    """
    Generate evaluation figures based on the aggregated metrics DataFrame.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    if df.empty:
        logger.warning("Empty dataframe, skipping figure generation.")
        return
        
    logger.info(f"Generating benchmark figures in {output_dir}...")
    
    # Pre-process dataframe if needed
    # Ensure 'mean' column exists for plotting
    val_col = 'mean' if 'mean' in df.columns else 'value'
    
    # Extract unique properties
    models = df['model'].unique() if 'model' in df.columns else ['unknown']
    lead_times = df['lead_time'].unique() if 'lead_time' in df.columns else []
    
    # Sorting lead times based on number if string
    try:
        df['lt_num'] = df['lead_time'].str.extract('(\d+)').astype(float)
        df_sorted = df.sort_values(by='lt_num')
    except:
        df_sorted = df
    
    # Set seaborn style
    sns.set_theme(style="whitegrid")
    
    # 1. Curvas por lead time
    # Metrics to plot vs lead time (independent of threshold or choose a default)
    metrics_to_plot = ["RMSE", "MAE", "SSIM"]
    
    for metric in metrics_to_plot:
        data_metric = df_sorted[df_sorted['metric_name'] == metric]
        if not data_metric.empty:
            plt.figure(figsize=(8, 6))
            sns.lineplot(data=data_metric, x='lead_time', y=val_col, hue='model', marker='o')
            plt.title(f"{metric} vs Lead Time")
            plt.xlabel("Lead Time")
            plt.ylabel(metric)
            plt.tight_layout()
            plt.savefig(out_path / f"curve_{metric.lower()}_vs_leadtime.png")
            plt.close()
            
    # For categorical metrics that depend on threshold, plot CSI vs lead time for diff thresholds
    cat_metrics = ["CSI", "POD", "FAR", "FSS"]
    for metric in cat_metrics:
        data_metric = df_sorted[df_sorted['metric_name'] == metric]
        if not data_metric.empty and 'threshold' in data_metric.columns:
            # We can plot one line per threshold for the given model
            for model in data_metric['model'].unique():
                df_model = data_metric[data_metric['model'] == model]
                plt.figure(figsize=(8, 6))
                sns.lineplot(data=df_model, x='lead_time', y=val_col, hue='threshold', marker='o', palette='viridis')
                plt.title(f"{metric} vs Lead Time by Threshold\nModel: {model}")
                plt.xlabel("Lead Time")
                plt.ylabel(metric)
                plt.legend(title='Threshold (mm/h)')
                plt.tight_layout()
                plt.savefig(out_path / f"curve_{metric.lower()}_vs_leadtime_m-{model}.png")
                plt.close()
                
    # 2. Performance Diagram (Success Ratio vs POD)
    data_pod = df_sorted[df_sorted['metric_name'] == "POD"]
    data_far = df_sorted[df_sorted['metric_name'] == "FAR"]
    
    if not data_pod.empty and not data_far.empty and 'threshold' in data_pod.columns:
        plt.figure(figsize=(8, 8))
        
        # Plot CSI curves (background)
        x = np.linspace(0.01, 1, 100)
        for csi in [0.2, 0.4, 0.6, 0.8]:
            y = 1 / (1/csi - 1/x + 1)
            mask = (y >= 0) & (y <= 1)
            plt.plot(x[mask], y[mask], '--', color='gray', linewidth=0.5)
            if mask.any():
                 plt.text(x[mask][-1], y[mask][-1], f"CSI={csi}", fontsize=8, color='gray')
                 
        # Plot points grouped by model and threshold
        # We can aggregate over lead_times or plot individual ones.
        # Here we plot the mean over lead times for each threshold.
        pod_agg = data_pod.groupby(['model', 'threshold'])[val_col].mean().reset_index()
        far_agg = data_far.groupby(['model', 'threshold'])[val_col].mean().reset_index()
        
        merged_cat = pd.merge(pod_agg, far_agg, on=['model', 'threshold'], suffixes=('_POD', '_FAR'))
        merged_cat['SR'] = 1.0 - merged_cat[f'{val_col}_FAR']
        
        palette = sns.color_palette("husl", len(merged_cat['model'].unique()))
        model_colors = dict(zip(merged_cat['model'].unique(), palette))
        
        for _, row in merged_cat.iterrows():
            plt.scatter(row['SR'], row[f'{val_col}_POD'], color=model_colors[row['model']], s=50, zorder=5)
            plt.annotate(f"{row['threshold']}mm", (row['SR'], row[f'{val_col}_POD']), xytext=(5, 5), textcoords='offset points', fontsize=8)
            
        import matplotlib.patches as mpatches
        handles = [mpatches.Patch(color=c, label=m) for m, c in model_colors.items()]
        plt.legend(handles=handles, title="Models")
        
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)
        plt.xlabel("Success Ratio (1 - FAR)")
        plt.ylabel("Probability of Detection (POD)")
        plt.title("Performance Diagram")
        plt.grid(True)
        plt.savefig(out_path / "performance_diagram.png")
        plt.close()
        
    # 3. Taylor Diagram (if Correlation and STD are available)
    data_std_obs = df_sorted[df_sorted['metric_name'] == "STD_obs"]
    data_std_pred = df_sorted[df_sorted['metric_name'] == "STD_pred"]
    data_corr = df_sorted[df_sorted['metric_name'] == "Correlation"]
    
    if not data_corr.empty and not data_std_obs.empty and not data_std_pred.empty:
        # aggregate by model over all lead times
        std_obs_agg = data_std_obs.groupby('model')[val_col].mean().reset_index()
        std_pred_agg = data_std_pred.groupby('model')[val_col].mean().reset_index()
        corr_agg = data_corr.groupby('model')[val_col].mean().reset_index()
        
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, polar=True)
        
        obs_val = std_obs_agg[val_col].mean() # assume same for all models roughly
        
        # Outer ring
        angles = np.linspace(0, np.pi/2, 100)
        ax.plot(angles, [obs_val]*100, 'k--', label='Observation')
        
        palette = sns.color_palette("Set1", len(std_pred_agg['model'].unique()))
        
        for idx, model in enumerate(std_pred_agg['model'].unique()):
            m_std_pred = std_pred_agg[std_pred_agg['model'] == model][val_col].values[0]
            m_corr = corr_agg[corr_agg['model'] == model][val_col].values[0]
            
            theta = np.arccos(np.clip(m_corr, -1.0, 1.0))
            ax.plot(theta, m_std_pred, 'o', color=palette[idx], label=model, markersize=8)
            
        ax.set_thetamin(0)
        ax.set_thetamax(90)
        
        # Add RMSE contours
        rs, ts = np.meshgrid(np.linspace(0, obs_val*2, 100), np.linspace(0, np.pi/2, 100))
        rms = np.sqrt(obs_val**2 + rs**2 - 2*obs_val*rs*np.cos(ts))
        contours = ax.contour(ts, rs, rms, 5, colors='g', alpha=0.3)
        ax.clabel(contours, inline=True, fontsize=8)
        
        ax.set_title("Taylor Diagram", pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        # Remove empty or non-needed grid lines
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(out_path / "taylor_diagram.png")
        plt.close()

    # 4. Painéis Resumo (Heatmaps)
    # Heatmap of CSI: rows=lead_time, cols=threshold
    data_csi = df_sorted[df_sorted['metric_name'] == "CSI"]
    if not data_csi.empty and 'threshold' in data_csi.columns:
        for model in data_csi['model'].unique():
            df_model = data_csi[data_csi['model'] == model]
            pivot_csi = df_model.pivot_table(index='lead_time', columns='threshold', values=val_col)
            
            # Reorder rows by lt_num if possible
            if 'lt_num' in df_model.columns:
                lt_order = df_model[['lead_time', 'lt_num']].drop_duplicates().sort_values('lt_num')['lead_time']
                pivot_csi = pivot_csi.reindex(lt_order)
                
            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot_csi, annot=True, cmap='YlGnBu', fmt=".3f")
            plt.title(f"CSI Heatmap (Lead Time vs Threshold)\nModel: {model}")
            plt.ylabel("Lead Time")
            plt.xlabel("Threshold (mm/h)")
            plt.tight_layout()
            plt.savefig(out_path / f"heatmap_csi_m-{model}.png")
            plt.close()

    logger.info("Figure generation completed!")

