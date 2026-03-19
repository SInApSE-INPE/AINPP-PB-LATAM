"""
Fractions Skill Score (FSS) for precipitation nowcasting.

This module implements FSS metrics calculation using xarray and dask for efficient
parallel processing of large datasets (Zarr/NetCDF).

Key components:
- FSS calculation using neighborhood fractions (box filter).
- Support for multiple thresholds and spatial scales.
- Zarr v2 output compatibility (to avoid codec issues).
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from numcodecs import Blosc
from scipy.ndimage import uniform_filter


# =============================================================================
# Helpers
# =============================================================================
def km_to_window_cells(
    scale_km: float,
    res_deg: float = 0.1,
    km_per_deg_lat: float = 111.0,
    force_odd: bool = True,
) -> int:
    """
    Convert a physical scale in km to a square window size in grid cells
    for a regular lat/lon grid with resolution res_deg.
    For 0.1°, 1 cell ~ 11.1 km in latitude.

    Args:
        scale_km (float): Scale in kilometers.
        res_deg (float, optional): Grid resolution in degrees. Defaults to 0.1.
        km_per_deg_lat (float, optional): Km per degree latitude. Defaults to 111.0.
        force_odd (bool, optional): Force window size to be odd. Defaults to True.

    Returns:
        int: Window size in cells.
    """
    cell_km = res_deg * km_per_deg_lat
    r = int(np.ceil(scale_km / cell_km))  # radius in cells
    if force_odd:
        return max(1, 2 * r + 1)
    return max(1, r)


def parse_lead_hours_from_name(name: str) -> int:
    """
    Parse lead hours from a directory name like 'LEAD_006h.zarr' or 'LEAD_006h'.

    Args:
        name (str): Directory or file name.

    Returns:
        int: Lead time in hours.

    Raises:
        ValueError: If patterns is not found.
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
    if "time" in ds.coords:
        ds = ds.sortby("time")
        # Deduplicate time (defensive)
        t = ds["time"].values
        _, idx = np.unique(t, return_index=True)
        if idx.size != t.size:
            ds = ds.isel(time=np.sort(idx))
    return ds


# =============================================================================
# FSS core (2D box filter on last two dims)
# =============================================================================
def _fraction_field_last2d_numpy(x: np.ndarray, win: int) -> np.ndarray:
    """
    Compute neighborhood fractions with a box filter (mean) over the last two dims.
    x shape: (..., lat, lon) as float32 (0/1 for binary event mask).
    """
    size = [1] * x.ndim
    size[-2] = win
    size[-1] = win
    return uniform_filter(x.astype(np.float32), size=size, mode="constant", cval=0.0)


def _safe_div(num: xr.DataArray, den: xr.DataArray, eps: float = 1e-12) -> xr.DataArray:
    """Safe division avoiding division by zero."""
    return xr.where(den > 0, num / (den + eps), 0.0)


def fss_from_fractions(F: xr.DataArray, O: xr.DataArray, eps: float = 1e-12) -> xr.DataArray:
    """
    Compute Fractions Skill Score (FSS) from fraction fields.
    Roberts & Lean (2008):
      FSS = 1 - MSE(F, O) / (mean(F^2) + mean(O^2))
    Here MSE and means are spatial means over (lat, lon), returning per-time values.

    Args:
        F (xr.DataArray): Forecast fraction field.
        O (xr.DataArray): Observation fraction field.
        eps (float, optional): Epsilon for stability. Defaults to 1e-12.

    Returns:
        xr.DataArray: FSS values (per time).
    """
    diff2 = (F - O) ** 2
    mse = diff2.mean(dim=("lat", "lon"))

    denom = (F**2).mean(dim=("lat", "lon")) + (O**2).mean(dim=("lat", "lon"))
    return (1.0 - _safe_div(mse, denom, eps=eps)).astype("float32")


def build_fraction_fields(
    fct: xr.DataArray,
    obs: xr.DataArray,
    threshold: float,
    win: int,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Build neighborhood fraction fields F and O (0..1) from exceedance events (>= threshold).
    Uses xr.apply_ufunc for dask parallelization over time chunks.

    IMPORTANT: To satisfy gufunc core-dim constraints with dask='parallelized',
    lat/lon must be single-chunk. Ensure caller does:
      fct = fct.chunk({"lat": -1, "lon": -1})
      obs = obs.chunk({"lat": -1, "lon": -1})
    """
    f_evt = (fct >= threshold).astype("float32")  # 0/1
    o_evt = (obs >= threshold).astype("float32")

    F = xr.apply_ufunc(
        _fraction_field_last2d_numpy,
        f_evt,
        kwargs={"win": win},
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["lat", "lon"]],
        dask="parallelized",
        output_dtypes=[np.float32],
    )
    O = xr.apply_ufunc(
        _fraction_field_last2d_numpy,
        o_evt,
        kwargs={"win": win},
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["lat", "lon"]],
        dask="parallelized",
        output_dtypes=[np.float32],
    )
    return F, O


# =============================================================================
# Configuration
# =============================================================================
@dataclass(frozen=True)
class FSSConfig:
    thresholds: Sequence[float] = (0.1, 1.0, 5.0, 10.0)
    scales_km: Sequence[float] = (11.1, 22.2, 55.5, 111.0, 222.0, 333.0)
    res_deg: float = 0.1
    # input read chunks (tune to your filesystem / scheduler)
    chunks: Optional[Dict[str, int]] = None
    # output chunking (small dims chunked as 1; time chunked for fast slicing)
    out_time_chunk: int = 48


def default_chunks() -> Dict[str, int]:
    return {"time": 48, "lat": 220, "lon": 242}


# =============================================================================
# Per-lead computation
# =============================================================================
def compute_fss_for_lead(zarr_path: Path, cfg: FSSConfig) -> xr.Dataset:
    """
    Computes FSS for a single lead time zarr store.
    """
    chunks = cfg.chunks or default_chunks()
    ds = open_lead_store(zarr_path, chunks=chunks)

    fct = ds["precip_fct"]
    obs = ds["precip_obs"]
    fct, obs = xr.align(fct, obs, join="inner")

    # Required for dask parallelized gufunc with core dims (lat/lon)
    fct = fct.chunk({"lat": -1, "lon": -1})
    obs = obs.chunk({"lat": -1, "lon": -1})

    lead_hours = parse_lead_hours_from_name(zarr_path.name)

    thr_list = list(cfg.thresholds)
    sc_list = list(cfg.scales_km)

    # We will build an output tensor: (threshold, scale_km, time)
    # by accumulating DataArrays and concatenating safely.
    per_thr: List[xr.Dataset] = []

    for thr in thr_list:
        per_scale: List[xr.DataArray] = []
        for skm in sc_list:
            win = km_to_window_cells(skm, res_deg=cfg.res_deg, force_odd=True)

            F, O = build_fraction_fields(fct=fct, obs=obs, threshold=thr, win=win)
            fss_t = fss_from_fractions(F, O)  # (time,)

            # annotate scale
            fss_t = fss_t.assign_coords(scale_km=np.float32(skm)).expand_dims("scale_km")
            per_scale.append(fss_t)

        fss_thr = xr.concat(per_scale, dim="scale_km")  # (scale_km, time)
        fss_thr = fss_thr.assign_coords(threshold=np.float32(thr)).expand_dims("threshold")
        per_thr.append(fss_thr.to_dataset(name="fss"))

    ds_out = xr.concat(per_thr, dim="threshold")  # (threshold, scale_km, time)

    # Add lead dimension
    ds_out = ds_out.assign_coords(lead_time_hours=np.int32(lead_hours)).expand_dims(
        "lead_time_hours"
    )

    # Add time-mean summary
    ds_out["fss_mean_time"] = ds_out["fss"].mean(dim="time").astype("float32")

    # Defensive: ensure time monotonic within this lead
    if "time" in ds_out.coords:
        ds_out = ds_out.sortby("time")

    # Keep metadata light to avoid merge conflicts across leads
    ds_out.attrs.update(
        dict(
            description="Fractions Skill Score (FSS) for precipitation threshold exceedance",
            lead_store=str(zarr_path),
            res_deg=float(cfg.res_deg),
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

    # Remove existing store to avoid mixing v2/v3 metadata (e.g., zarr.json)
    safe_rmtree(out_path)

    compressor = make_zarr_v2_compressor()

    encoding = {}
    for v in ds.data_vars:
        encoding[v] = {
            "compressor": compressor,  # v2
            "dtype": "float32",
        }

    ds.to_zarr(
        out_path,
        mode="w",
        consolidated=True,
        encoding=encoding,
        zarr_format=2,  # CRITICAL: force v2
    )


def compute_fss_for_all_leads(
    model_root: Path,
    out_path: Path,
    cfg: FSSConfig,
) -> xr.Dataset:
    """
    Compute FSS for each LEAD_*.zarr and save one aggregated Zarr (v2).
    """
    model_root = Path(model_root)
    lead_stores = sorted([p for p in model_root.glob("LEAD_*.zarr") if p.is_dir()])
    if not lead_stores:
        raise FileNotFoundError(f"No LEAD_*.zarr stores found in: {model_root}")

    results: List[xr.Dataset] = []

    # Use tqdm if available
    try:
        from tqdm import tqdm

        iterator = tqdm(lead_stores, desc="Computing FSS")
    except ImportError:
        iterator = lead_stores

    for z in iterator:
        # print(f"[INFO] Computing FSS for {z.name}")
        ds_fss = compute_fss_for_lead(z, cfg)
        results.append(ds_fss)

    # IMPORTANT: avoid combine_by_coords (monotonic global time constraint).
    # We concatenate along lead_time_hours, keeping each lead's time coordinate intact.
    ds_all = xr.concat(results, dim="lead_time_hours")

    # Sort lead_time_hours for convenience
    ds_all = ds_all.sortby("lead_time_hours")

    # Chunk output for efficient downstream queries
    ds_all = ds_all.chunk(
        {
            "lead_time_hours": 1,
            "threshold": 1,
            "scale_km": 1,
            "time": int(cfg.out_time_chunk),
        }
    )

    # Save as Zarr v2
    save_dataset_zarr_v2(ds_all, out_path)
    print(f"[OK] Saved: {out_path}")

    return ds_all
