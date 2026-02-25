"""
Categorical metrics (Contingency Table) for precipitation nowcasting.

This module implements categorical metrics calculation using xarray and dask for efficient
parallel processing of large datasets (Zarr/NetCDF).

Metrics included:
- POD (Probability of Detection)
- FAR (False Alarm Ratio)
- CSI (Critical Success Index)
- ETS (Equitable Threat Score)
- HSS (Heidke Skill Score)
- Bias (Frequency Bias)
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Literal

import numpy as np
import xarray as xr
from numcodecs import Blosc


# =============================================================================
# Helpers
# =============================================================================
TIME_NAME = "time"
LAT_NAME = "lat"
LON_NAME = "lon"

def parse_lead_hours_from_name(name: str) -> int:
    """
    Parse lead hours from a directory name like 'LEAD_006h.zarr' or 'LEAD_006h'.
    """
    m = re.search(r"LEAD_(\d+)h", name)
    if not m:
        raise ValueError(f"Could not parse lead hours from: {name}")
    return int(m.group(1))


def safe_rmtree(path: Path) -> None:
    """Safely remove a directory tree if it exists."""
    if path.exists():
        shutil.rmtree(path)


def make_zarr_v2_compressor() -> Blosc:
    """
    Returns a Zarr v2 compatible compressor (numcodecs).
    """
    return Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)


def open_lead_store(zarr_path: Path, chunks: Dict[str, int]) -> xr.Dataset:
    """
    Opens a Zarr store, sorting and deduplicating by time.
    """
    ds = xr.open_zarr(zarr_path, consolidated=True, chunks=chunks)
    if TIME_NAME in ds.coords:
        ds = ds.sortby(TIME_NAME)
        # Deduplicate time (defensive)
        t = ds[TIME_NAME].values
        _, idx = np.unique(t, return_index=True)
        if idx.size != t.size:
            ds = ds.isel({TIME_NAME: np.sort(idx)})
    return ds


# =============================================================================
# Contingency Table Core
# =============================================================================
def compute_contingency_hits(
    fct: xr.DataArray,
    obs: xr.DataArray,
    threshold: float,
) -> xr.Dataset:
    """
    Computes Hits, Misses, False Alarms, Correct Negatives for a given threshold.
    Returns counts per time step (summed over lat/lon).

    Args:
        fct (xr.DataArray): Forecast data.
        obs (xr.DataArray): Observation data.
        threshold (float): Threshold value.

    Returns:
        xr.Dataset: Dataset with variables hits, misses, fa, cn.
    """
    # Binarize
    f_bool = fct >= threshold
    o_bool = obs >= threshold

    # Contingency elements
    hits = (f_bool & o_bool)
    misses = (~f_bool & o_bool)
    fa = (f_bool & ~o_bool)
    cn = (~f_bool & ~o_bool)

    # Sum over spatial dims
    red = (LAT_NAME, LON_NAME)
    
    return xr.Dataset(
        data_vars=dict(
            hits=hits.sum(dim=red).astype("int32"),
            misses=misses.sum(dim=red).astype("int32"),
            fa=fa.sum(dim=red).astype("int32"),
            cn=cn.sum(dim=red).astype("int32"),
        )
    )


def compute_categorical_stats(ct: xr.Dataset, eps: float = 1e-6) -> xr.Dataset:
    """
    Computes POD, FAR, CSI, ETS, HSS, Bias from contingency table counts.
    Expected vars in ct: hits, misses, fa, cn.

    Args:
        ct (xr.Dataset): Contingency table counts.
        eps (float, optional): Epsilon for stability. Defaults to 1e-6.

    Returns:
        xr.Dataset: Categorical statistics.
    """
    hits = ct["hits"]
    misses = ct["misses"]
    fa = ct["fa"]
    cn = ct["cn"]
    
    total = hits + misses + fa + cn
    
    # POD = Hits / (Hits + Misses)
    pod = hits / (hits + misses + eps)
    
    # FAR = FA / (Hits + FA)
    far = fa / (hits + fa + eps)
    
    # CSI = Hits / (Hits + Misses + FA)
    csi = hits / (hits + misses + fa + eps)
    
    # Bias = (Hits + FA) / (Hits + Misses)
    bias = (hits + fa) / (hits + misses + eps)
    
    # ETS
    # Hits_rand = (Hits + Misses) * (Hits + FA) / Total
    hits_rand = (hits + misses) * (hits + fa) / (total + eps)
    ets = (hits - hits_rand) / (hits + misses + fa - hits_rand + eps)
    
    # HSS
    # Num = 2 * (Hits * CN - Misses * FA)
    # Denom = (Hits + Misses)*(Misses + CN) + (Hits + FA)*(FA + CN)
    hss_num = 2 * (hits * cn - misses * fa)
    hss_den = (hits + misses) * (misses + cn) + (hits + fa) * (fa + cn)
    hss = hss_num / (hss_den + eps)
    
    return xr.Dataset(
        data_vars=dict(
            pod=pod.astype("float32"),
            far=far.astype("float32"),
            csi=csi.astype("float32"),
            bias=bias.astype("float32"),
            ets=ets.astype("float32"),
            hss=hss.astype("float32"),
        )
    )


# =============================================================================
# Configuration
# =============================================================================
@dataclass(frozen=True)
class CategoricalConfig:
    thresholds: Sequence[float] = (0.1, 1.0, 5.0, 10.0)
    # input read chunks
    read_chunks: Optional[Dict[str, int]] = None
    # output chunking
    out_chunks: Optional[Dict[str, int]] = None


def default_chunks() -> Dict[str, int]:
    return {TIME_NAME: 256, LAT_NAME: 220, LON_NAME: 242}


# =============================================================================
# Per-lead computation
# =============================================================================
def compute_metrics_for_lead(zarr_path: Path, cfg: CategoricalConfig) -> xr.Dataset:
    chunks = cfg.read_chunks or default_chunks()
    ds = open_lead_store(zarr_path, chunks=chunks)

    fct = ds["precip_fct"]
    obs = ds["precip_obs"]
    fct, obs = xr.align(fct, obs, join="inner")

    lead_hours = parse_lead_hours_from_name(zarr_path.name)
    thr_list = list(cfg.thresholds)

    per_thr: List[xr.Dataset] = []

    for thr in thr_list:
        # Compute counts
        ct = compute_contingency_hits(fct, obs, threshold=thr)
        
        # Compute stats per time
        stats = compute_categorical_stats(ct)
        
        # Merge counts and stats
        combined = xr.merge([ct, stats])
        
        # Add threshold dim
        combined = combined.assign_coords(threshold=np.float32(thr)).expand_dims("threshold")
        per_thr.append(combined)

    ds_out = xr.concat(per_thr, dim="threshold")
    
    # Add lead dimension
    ds_out = ds_out.assign_coords(lead_time_hours=np.int32(lead_hours)).expand_dims("lead_time_hours")
    
    # Compute mean over time
    vars_to_mean = list(ds_out.data_vars)
    for v in vars_to_mean:
        ds_out[f"{v}_mean_time"] = ds_out[v].mean(dim=TIME_NAME).astype("float32")
    
    # Defensive sort
    if TIME_NAME in ds_out.coords:
        ds_out = ds_out.sortby(TIME_NAME)

    ds_out.attrs.update(
        dict(
            description="Categorical metrics (POD, FAR, CSI, etc.) for precipitation",
            lead_store=str(zarr_path),
        )
    )
    
    ds.close()
    return ds_out


# =============================================================================
# All-leads driver + Zarr v2 save
# =============================================================================
def save_dataset_zarr_v2(ds: xr.Dataset, out_path: Path) -> None:
    """
    Save dataset to Zarr v2 to avoid Zarr v3 codec errors.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    safe_rmtree(out_path)

    compressor = make_zarr_v2_compressor()
    encoding = {}
    for v in ds.data_vars:
        # Use appropriate compression/dtype
        dtype = "float32" if ds[v].dtype.kind == 'f' else "int32"
        encoding[v] = {
            "compressor": compressor,
            "dtype": dtype,
        }

    ds.to_zarr(
        out_path,
        mode="w",
        consolidated=True,
        encoding=encoding,
        zarr_format=2,
    )


def compute_metrics_for_all_leads(
    model_root: Path,
    out_path: Path,
    cfg: CategoricalConfig,
) -> xr.Dataset:
    """
    Compute Categorical metrics for all leads.
    """
    model_root = Path(model_root)
    lead_stores = sorted([p for p in model_root.glob("LEAD_*.zarr") if p.is_dir()])
    if not lead_stores:
        raise FileNotFoundError(f"No LEAD_*.zarr found in {model_root}")

    results: List[xr.Dataset] = []
    
    try:
        from tqdm import tqdm
        iterator = tqdm(lead_stores, desc="Computing Cat Metrics")
    except ImportError:
        iterator = lead_stores

    for z in iterator:
        results.append(compute_metrics_for_lead(z, cfg))

    ds_all = xr.concat(results, dim="lead_time_hours")
    ds_all = ds_all.sortby("lead_time_hours")

    # Output chunking
    out_chunks = cfg.out_chunks or {
        "lead_time_hours": 1,
        "threshold": 1,
        TIME_NAME: 256
    }
    ds_all = ds_all.chunk(out_chunks)

    save_dataset_zarr_v2(ds_all, out_path)
    print(f"[OK] Saved: {out_path}")
    
    return ds_all


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    pass
