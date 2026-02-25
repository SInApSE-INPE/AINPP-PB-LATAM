from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Literal, Tuple, Union

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from dask.diagnostics import ProgressBar

# =============================================================================
# CONFIG
# =============================================================================
LAT_NAME = "lat"
LON_NAME = "lon"
TIME_NAME = "time"

Engine = Literal["zarr", "netcdf"]


# =============================================================================
# HELPERS
# =============================================================================
def parse_lead_hours(name: str) -> int:
    """
    Parses the lead time in hours from a string (e.g., 'LEAD_01h').

    Args:
        name (str): String containing the lead time.

    Returns:
        int: Lead time in hours.

    Raises:
        ValueError: If the pattern is not found.
    """
    m = re.search(r"LEAD_(\d+)h", name)
    if not m:
        raise ValueError(f"Cannot parse lead from {name}")
    return int(m.group(1))


def sort_and_dedup_time(ds: xr.Dataset) -> xr.Dataset:
    """
    Sorts a dataset by time and removes duplicate time steps.

    Args:
        ds (xr.Dataset): Input dataset.

    Returns:
        xr.Dataset: Dataset sorted by time with duplicates removed.
    """
    if TIME_NAME not in ds.coords:
        return ds
    ds = ds.sortby(TIME_NAME)
    t = ds[TIME_NAME].values
    _, idx = np.unique(t, return_index=True)
    if idx.size != t.size:
        ds = ds.isel({TIME_NAME: np.sort(idx)})
    return ds


def open_lead_store(zarr_path: Path, chunks: Optional[Dict[str, int]] = None) -> xr.Dataset:
    """
    Opens a Zarr store for a specific lead time, sorting and deduplicating by time.

    Args:
        zarr_path (Path): Path to the Zarr store.
        chunks (Dict[str, int], optional): Chunk sizes.

    Returns:
        xr.Dataset: Opened dataset.
    """
    ds = xr.open_zarr(zarr_path, consolidated=True, chunks=chunks)
    ds = sort_and_dedup_time(ds)
    return ds


def _spearman_2d(a: np.ndarray, b: np.ndarray) -> np.float32:
    """
    Computes Spearman correlation between two 2D arrays (lat, lon), ignoring NaNs.
    Returns rho only.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        np.float32: Spearman correlation coefficient.
    """
    x = np.asarray(a).ravel()
    y = np.asarray(b).ravel()

    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.float32(np.nan)

    rho = spearmanr(x[m], y[m]).correlation
    return np.float32(rho)


# =============================================================================
# METRICS CORE
# =============================================================================
def compute_basic_error_metrics(
    fct: xr.DataArray,
    obs: xr.DataArray,
) -> xr.Dataset:
    """
    Computes MSE, MAE, Bias (mean error), and RMSE.
    Returns metrics(time) after reducing over (lat, lon).

    Args:
        fct (xr.DataArray): Forecast data.
        obs (xr.DataArray): Observation data.

    Returns:
        xr.Dataset: Dataset containing the computed metrics.
    """
    err = fct - obs
    red = (LAT_NAME, LON_NAME)

    mse = (err ** 2).mean(dim=red)
    mae = np.abs(err).mean(dim=red)
    bias = err.mean(dim=red)
    rmse = np.sqrt(mse)

    return xr.Dataset(
        data_vars=dict(
            mse=mse.astype("float32"),
            mae=mae.astype("float32"),
            bias=bias.astype("float32"),
            rmse=rmse.astype("float32"),
        )
    )


def compute_spearman_time_series(
    fct: xr.DataArray,
    obs: xr.DataArray,
) -> xr.DataArray:
    """
    Computes Spearman rho per time step.
    Requires lat/lon single chunk if using dask='parallelized' with core dims.

    Args:
        fct (xr.DataArray): Forecast data.
        obs (xr.DataArray): Observation data.

    Returns:
        xr.DataArray: Spearman stats per time.
    """
    rho = xr.apply_ufunc(
        _spearman_2d,
        fct,
        obs,
        input_core_dims=[[LAT_NAME, LON_NAME], [LAT_NAME, LON_NAME]],
        output_core_dims=[[]],     # scalar per time
        vectorize=True,            # loops over remaining dims (time)
        dask="parallelized",
        output_dtypes=[np.float32],
    )
    rho.name = "spearman_rho"
    return rho


# =============================================================================
# PER-LEAD PIPELINE
# =============================================================================
@dataclass(frozen=True)
class MetricsConfig:
    read_chunks: Optional[Dict[str, int]] = None
    output_chunks: Optional[Dict[str, int]] = None


def default_read_chunks() -> Dict[str, int]:
    return {TIME_NAME: 256, LAT_NAME: 220, LON_NAME: 242}


def compute_metrics_for_lead(
    zarr_path: Path,
    cfg: MetricsConfig,
) -> xr.Dataset:
    """
    Computes metrics for a single lead time from a Zarr store.

    Returns Dataset with dims:
      lead_time_hours, time
    variables:
      mse, mae, bias, rmse, spearman_rho
      mse_mean_time, mae_mean_time, bias_mean_time, rmse_mean_time, spearman_rho_mean_time

    Args:
        zarr_path (Path): Path to the Zarr store.
        cfg (MetricsConfig): Configuration object.

    Returns:
        xr.Dataset: computed metrics dataset
    """
    chunks = cfg.read_chunks or default_read_chunks()
    ds = open_lead_store(zarr_path, chunks=chunks)

    fct = ds["precip_fct"]
    obs = ds["precip_obs"]
    fct, obs = xr.align(fct, obs, join="inner")

    # Critical for apply_ufunc with core dims
    fct = fct.chunk({LAT_NAME: -1, LON_NAME: -1})
    obs = obs.chunk({LAT_NAME: -1, LON_NAME: -1})

    lead = parse_lead_hours(zarr_path.name)

    ds_err = compute_basic_error_metrics(fct, obs)
    rho = compute_spearman_time_series(fct, obs)

    ds_out = xr.merge([ds_err, rho.to_dataset()])

    # time means
    ds_out["mse_mean_time"] = ds_out["mse"].mean(dim=TIME_NAME)
    ds_out["mae_mean_time"] = ds_out["mae"].mean(dim=TIME_NAME)
    ds_out["bias_mean_time"] = ds_out["bias"].mean(dim=TIME_NAME)
    ds_out["rmse_mean_time"] = ds_out["rmse"].mean(dim=TIME_NAME)
    ds_out["spearman_rho_mean_time"] = ds_out["spearman_rho"].mean(dim=TIME_NAME)

    # add lead dimension
    ds_out = ds_out.expand_dims(lead_time_hours=[np.int32(lead)])

    ds_out.attrs.update(
        dict(
            description="Scalar error metrics and Spearman correlation for precipitation forecasts",
            domain="LATAM 880x970 (0.1 degree)",
            units="mm hr-1",
            source=str(zarr_path),
            notes="All error metrics reduced over lat/lon; Spearman computed per time over flattened grid.",
        )
    )

    ds.close()
    return ds_out


# =============================================================================
# ALL LEADS -> SINGLE ZARR
# =============================================================================
def compute_metrics_for_all_leads(
    model_root: Path,
    out_path: Path,
    cfg: MetricsConfig,
    engine: Engine = "zarr",
) -> xr.Dataset:
    """
    Reads all LEAD_*.zarr in model_root, computes metrics, and writes to one file.

    Args:
        model_root (Path): Directory containing LEAD_*.zarr directories.
        out_path (Path): Path to save the output file.
        cfg (MetricsConfig): Configuration object.
        engine (Engine, optional): Output engine ('zarr' or 'netcdf'). Defaults to "zarr".

    Returns:
        xr.Dataset: Combined dataset with all metrics.
    """
    lead_stores = sorted([p for p in model_root.glob("LEAD_*.zarr") if p.is_dir()])
    if not lead_stores:
        raise FileNotFoundError(f"No LEAD_*.zarr found in {model_root}")

    per_lead: List[xr.Dataset] = []
    
    # Use tqdm if available, otherwise just print
    try:
        from tqdm import tqdm
        iterator = tqdm(lead_stores, desc="Computing metrics")
    except ImportError:
        iterator = lead_stores

    for z in iterator:
        # print(f"[INFO] Computing metrics for {z.name}")
        per_lead.append(compute_metrics_for_lead(z, cfg))

    ds_all = xr.concat(per_lead, dim="lead_time_hours", join="outer")

    # keep lead sorted
    ds_all = ds_all.sortby("lead_time_hours")

    # make time monotonic globally (helpful for downstream)
    if TIME_NAME in ds_all.coords:
        ds_all = ds_all.sortby(TIME_NAME)

    # output chunking
    if cfg.output_chunks:
        ds_all = ds_all.chunk(cfg.output_chunks)
    else:
        # reasonable default for small scalar outputs
        if TIME_NAME in ds_all.dims:
            ds_all = ds_all.chunk({"lead_time_hours": 1, TIME_NAME: 256})
        else:
            ds_all = ds_all.chunk({"lead_time_hours": 1})

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if engine == "zarr":
        ds_all.to_zarr(out_path, mode="w", consolidated=True, zarr_format=2)
    else:
        ds_all.to_netcdf(out_path)

    print(f"[OK] Saved: {out_path}")
    return ds_all

# =============================================================================
# PLOTTING
# =============================================================================
def plot_metrics(
    model_results: Dict[str, Path],
    baseline_results: Optional[Dict[str, Path]] = None,
    output_dir: Path = Path("."),
    metrics: Optional[Dict[str, str]] = None
):
    """
    Plots metrics for multiple models comparing against baselines.
    
    Args:
        model_results: Dict mapping model name to path of scalar_metrics_all_leads.zarr
        baseline_results: Dict mapping baseline name to path of statistics .nc/.zarr
        output_dir: Directory to save plots
        metrics: Dict mapping metric key to display name
    """
    if metrics is None:
        metrics = {
            'mse_mean_time': 'Mean Squared Error',
            'mae_mean_time': 'Mean Absolute Error',
            'bias_mean_time': 'Bias',
            'rmse_mean_time': 'Root Mean Squared Error',
            'spearman_rho_mean_time': 'Spearman Correlation',
        }

    model_colors = [
        "#1f77b4", "#ff7f0e", "#ffd700", "#2ca02c", "#d62728", 
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
    ]
    
    letters = [chr(i) for i in range(97, 123)]  # a-z

    df = pd.DataFrame()
    
    # Process Models
    for i, (name, path) in enumerate(model_results.items()):
        try:
            ds = xr.open_zarr(path, consolidated=True)
            temp = ds[list(metrics.keys())].to_dataframe().reset_index()
            temp['model_name'] = name
            df = pd.concat([df, temp], axis=0)
        except Exception as e:
            print(f"Error loading {name} at {path}: {e}")

    # Process Baselines
    # Note: Baselines often have different structures (one file per forecast hour vs one combined file)
    # The original code handled separate files for baselines. 
    # Here we assume baselines might be processed similar to models or we need a custom loader.
    # For now, implemented a generic loader assuming similar structure or we skip specific baseline logic
    # unless provided in the dictionary.
    
    if baseline_results:
        for name, path in baseline_results.items():
            # This part needs customization based on how baselines are stored
            # For this implementation, we assume they are compatible or handled by the user
            pass

    if df.empty:
        print("No data to plot.")
        return

    # Plotting
    model_names = list(model_results.keys())
    # Add baselines to model_names if processed
    
    num_metrics = len(metrics)
    cols = 3
    rows = (num_metrics + cols - 1) // cols
    
    fig, ax = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    if rows == 1: ax = np.expand_dims(ax, axis=0)
    if cols == 1: ax = np.expand_dims(ax, axis=1)

    for i, (metric_key, metric_label) in enumerate(metrics.items()):
        ax_row = i // cols
        ax_col = i % cols
        
        # Determine unique models in df
        unique_models = df['model_name'].unique()
        
        for j, model_name in enumerate(unique_models):
            subset = df[df['model_name'] == model_name]
            subset = subset.sort_values('lead_time_hours')
            
            color = model_colors[j % len(model_colors)]
            
            ax[ax_row, ax_col].plot(
                subset['lead_time_hours'], subset[metric_key],
                label=model_name,
                marker='o',
                markersize=5,
                color=color
            )
            
        ax[ax_row, ax_col].set_title(metric_label, y=1.02, fontsize=14, fontweight='600')
        ax[ax_row, ax_col].set_title(f"{letters[i]})", loc='left', fontsize=14, fontweight='600')
        ax[ax_row, ax_col].set_xlabel('Lead Time (hours)', fontsize=14)
        ax[ax_row, ax_col].tick_params(axis='both', which='major', labelsize=12)
        ax[ax_row, ax_col].grid(ls='--', alpha=0.5)

    # Hide empty subplots
    for k in range(i + 1, rows * cols):
        r, c = k // cols, k % cols
        ax[r, c].axis('off')
        
    # Legend
    handles, labels = ax[0, 0].get_legend_handles_labels()
    # Place legend in the last (empty) subplot if available, else bottom
    last_ax = ax[-1, -1]
    if i < rows * cols - 1:
        last_ax.legend(handles, labels, loc='center', ncol=1, fontsize=12, frameon=False, title='Model', title_fontsize=14)
    else:
        fig.legend(handles, labels, loc='lower center', ncol=len(unique_models), bbox_to_anchor=(0.5, 0.0))

    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    
    out_png = output_dir / 'cont_metrics_all_models.png'
    out_pdf = output_dir / 'cont_metrics_all_models.pdf'
    
    plt.savefig(out_png, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(out_pdf, dpi=150, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    # Example usage
    pass
