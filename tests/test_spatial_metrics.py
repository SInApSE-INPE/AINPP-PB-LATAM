
import pytest
import numpy as np
import xarray as xr
import pandas as pd
from src.evaluation.spatial_metrics import (
    km_to_window_cells,
    fss_from_fractions,
    build_fraction_fields,
    compute_fss_for_lead,
    FSSConfig
)

def test_km_to_window_cells():
    # 0.1 deg ~ 11.1 km
    # 11.1 km -> 1 cell -> window 1 (if force_odd=False) or 3 (if force_odd=True?)
    # The function says: r = ceil(scale / cell_km). if scale=11.1, cell=11.1, r=1.
    # if force_odd: max(1, 2*r+1) -> 2*1+1 = 3.
    # Wait, r is "radius in cells"? The function says "r = int(np.ceil(scale_km / cell_km))".
    # Implementation:
    # cell_km = 0.1 * 111 = 11.1
    # r = ceil(scale / 11.1)
    # force_odd: 2*r + 1.
    
    # Scale 11.1 km: r = 1. window = 3.
    assert km_to_window_cells(11.1, res_deg=0.1, force_odd=True) == 3
    
    # Scale 1 km: r = 1 (ceil(1/11.1)). window = 3.
    assert km_to_window_cells(1.0, res_deg=0.1, force_odd=True) == 3
    
    # Scale 0 km: r=0? ceil(0) = 0. 2*0+1 = 1.
    assert km_to_window_cells(0.0, res_deg=0.1, force_odd=True) == 1


def test_fss_perfect_match():
    # If F and O are identical, FSS should be 1.
    lats = np.linspace(-10, 10, 10)
    lons = np.linspace(-10, 10, 10)
    
    data = np.random.rand(10, 10)
    F = xr.DataArray(data, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))
    O = F.copy()
    
    fss = fss_from_fractions(F, O)
    assert np.isclose(fss, 1.0)


def test_fss_total_mismatch():
    # F=1 everywhere, O=0 everywhere: FSS = 0?
    # MSE = 1. Mean(F^2) = 1. Mean(O^2) = 0.
    # FSS = 1 - 1 / (1 + 0) = 0.
    lats = np.linspace(-10, 10, 10)
    lons = np.linspace(-10, 10, 10)
    
    F = xr.DataArray(np.ones((10, 10)), coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))
    O = xr.DataArray(np.zeros((10, 10)), coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))
    
    fss = fss_from_fractions(F, O)
    assert np.isclose(fss, 0.0)


def test_build_fraction_fields():
    # Simple check on shapes and range
    times = pd.date_range("2023-01-01", periods=2)
    lats = np.arange(10)
    lons = np.arange(10)
    
    fct = xr.DataArray(np.random.rand(2, 10, 10), coords={"time": times, "lat": lats, "lon": lons}, dims=("time", "lat", "lon")).chunk({"time": 1, "lat": -1, "lon": -1})
    obs = xr.DataArray(np.random.rand(2, 10, 10), coords={"time": times, "lat": lats, "lon": lons}, dims=("time", "lat", "lon")).chunk({"time": 1, "lat": -1, "lon": -1})
    
    F, O = build_fraction_fields(fct, obs, threshold=0.5, win=3)
    
    assert F.shape == fct.shape
    assert O.shape == obs.shape
    assert F.max() <= 1.0
    assert F.min() >= 0.0
