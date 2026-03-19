import os
import sys
from pathlib import Path

# Add src to path to allow direct execution
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from ainpp_pb_latam.visualization.generate_figures import generate_benchmark_figures
from ainpp_pb_latam.visualization.plot_advanced import (
    plot_barplots_and_rankings,
    plot_fss_by_scale,
    plot_object_histograms,
    plot_pr_curve,
    plot_probability_histogram,
    plot_spatial_power_spectrum,
    plot_statistical_consistency,
)
from ainpp_pb_latam.visualization.plot_maps import plot_comparison
from ainpp_pb_latam.visualization.plot_metrics import (
    plot_performance_diagram,
    plot_reliability_diagram,
    plot_roc_curve,
)


def main():
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "dummy_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_str = str(output_dir)
    print(f"Generating all dummy figures in: {out_str}")

    # 1. Dummy DataFrame for 'generate_benchmark_figures' and 'plot_barplots_and_rankings'
    # Need metrics: RMSE, MAE, SSIM, CSI, POD, FAR, FSS, STD_obs, STD_pred, Correlation
    records = []
    models = ["unet", "gan", "afno"]
    lead_times = ["T+1", "T+2", "T+3"]
    thresholds = [0.1, 1.0, 5.0]

    for m in models:
        for lt in lead_times:
            # Continuous metrics (no threshold)
            records.append(
                {
                    "model": m,
                    "lead_time": lt,
                    "threshold": np.nan,
                    "metric_name": "RMSE",
                    "mean": np.random.uniform(5, 10),
                }
            )
            records.append(
                {
                    "model": m,
                    "lead_time": lt,
                    "threshold": np.nan,
                    "metric_name": "MAE",
                    "mean": np.random.uniform(2, 5),
                }
            )
            records.append(
                {
                    "model": m,
                    "lead_time": lt,
                    "threshold": np.nan,
                    "metric_name": "SSIM",
                    "mean": np.random.uniform(0.6, 0.9),
                }
            )
            records.append(
                {
                    "model": m,
                    "lead_time": lt,
                    "threshold": np.nan,
                    "metric_name": "STD_obs",
                    "mean": 10.0,
                }
            )
            records.append(
                {
                    "model": m,
                    "lead_time": lt,
                    "threshold": np.nan,
                    "metric_name": "STD_pred",
                    "mean": np.random.uniform(7, 12),
                }
            )
            records.append(
                {
                    "model": m,
                    "lead_time": lt,
                    "threshold": np.nan,
                    "metric_name": "Correlation",
                    "mean": np.random.uniform(0.5, 0.95),
                }
            )

            # Categorical/Threshold dependent
            for t in thresholds:
                records.append(
                    {
                        "model": m,
                        "lead_time": lt,
                        "threshold": t,
                        "metric_name": "CSI",
                        "mean": np.random.uniform(0.1, 0.5),
                    }
                )
                records.append(
                    {
                        "model": m,
                        "lead_time": lt,
                        "threshold": t,
                        "metric_name": "POD",
                        "mean": np.random.uniform(0.3, 0.8),
                    }
                )
                records.append(
                    {
                        "model": m,
                        "lead_time": lt,
                        "threshold": t,
                        "metric_name": "FAR",
                        "mean": np.random.uniform(0.1, 0.6),
                    }
                )
                records.append(
                    {
                        "model": m,
                        "lead_time": lt,
                        "threshold": t,
                        "metric_name": "FSS",
                        "mean": np.random.uniform(0.3, 0.8),
                    }
                )

    df_dummy = pd.DataFrame(records)

    # Gerar figuras base (Curvas lead time, Taylor, Heatmaps, Performance Diagram global)
    generate_benchmark_figures(df_dummy, out_str)

    # Comparison between models: Rankings/Barplots
    plot_barplots_and_rankings(df_dummy, out_str, metric="RMSE")
    plot_barplots_and_rankings(df_dummy, out_str, metric="CSI")

    # 2. Dummy Data for Probabilistic, ROC, PR, Spatial and Object Metrics
    obs_1d = np.random.randint(0, 2, 1000)
    probs_1d = np.random.rand(1000)

    # Probabilistic Visuals
    plot_roc_curve(obs_1d, probs_1d, out_str)
    plot_pr_curve(obs_1d, probs_1d, out_str)
    plot_reliability_diagram(obs_1d, probs_1d, out_str)
    plot_probability_histogram(probs_1d, obs_1d, out_str)

    # 3. Spatial and Consistency Data (Continuous maps)
    B, H, W = 1, 256, 256
    target_map = np.random.lognormal(
        mean=0, sigma=1, size=(H, W)
    )  # precipitation like distribution
    pred_map = target_map * 0.8 + np.random.normal(0, 0.5, (H, W))
    pred_map = np.clip(pred_map, 0, None)

    # Map Comparison (Sharpness / Error map)
    plot_comparison(
        target_map,
        pred_map,
        Path(out_str) / "sharpness_comparison.png",
        title="Spatial Error & Blur check",
    )

    # Spatial Spectrum
    plot_spatial_power_spectrum(target_map, pred_map, out_str)

    # Statistical Consistency (CDF, Exceedance)
    plot_statistical_consistency(target_map, pred_map, out_str)

    # 4. Multiscale FSS
    scales = [1, 3, 5, 9, 15, 31, 63]
    fss_dummy = [np.clip(0.2 + (s / 63) * 0.6 + np.random.normal(0, 0.05), 0, 1) for s in scales]
    plot_fss_by_scale(scales, fss_dummy, out_str)

    # 5. Object-based
    obj_areas_obs = np.random.gamma(shape=2, scale=100, size=50)  # Gamma distribution for areas
    obj_areas_pred = np.random.gamma(shape=1.8, scale=120, size=45)
    plot_object_histograms(obj_areas_obs, obj_areas_pred, out_str)

    print(
        "\n[SUCCESS] Todas as figuras e categorias listadas na SKILL.md foram geradas com sucesso!"
    )


if __name__ == "__main__":
    main()
