
import pytest
import numpy as np
import xarray as xr
import pandas as pd
from ainpp.evaluation.categorical_metrics import (
    compute_contingency_hits,
    compute_categorical_stats,
    CategoricalConfig,
    compute_metrics_for_lead
)

def test_compute_contingency_hits():
    # 2x2 grid, 2 time steps
    # Time 0:
    # F: [[0.5, 1.5], [0.5, 1.5]]
    # O: [[1.5, 1.5], [0.5, 0.5]]
    # Threshold 1.0
    # F_bool: [[0, 1], [0, 1]]
    # O_bool: [[1, 1], [0, 0]]
    # Hits: (1,1)->1. Misses: (1,0)->0, (0,1)->0?
    # F=1, O=1 -> Hit. (0,1).
    # F=0, O=1 -> Miss. (0,0).
    # F=1, O=0 -> FA.   (1,1).
    # F=0, O=0 -> CN.   (1,0).
    
    # Time 1: All zeros -> CN=4.
    
    times = pd.date_range("2023-01-01", periods=2)
    lats = np.arange(2)
    lons = np.arange(2)
    
    fct_data = np.array([
        [[0.5, 1.5], [0.5, 1.5]],
        [[0.0, 0.0], [0.0, 0.0]]
    ])
    obs_data = np.array([
        [[1.5, 1.5], [0.5, 0.5]],
        [[0.0, 0.0], [0.0, 0.0]]
    ])
    
    fct = xr.DataArray(fct_data, coords={"time": times, "lat": lats, "lon": lons}, dims=("time", "lat", "lon"))
    obs = xr.DataArray(obs_data, coords={"time": times, "lat": lats, "lon": lons}, dims=("time", "lat", "lon"))
    
    ct = compute_contingency_hits(fct, obs, threshold=1.0)
    
    # Time 0
    assert ct["hits"].isel(time=0) == 1
    assert ct["misses"].isel(time=0) == 1
    assert ct["fa"].isel(time=0) == 1
    assert ct["cn"].isel(time=0) == 1
    
    # Time 1
    assert ct["hits"].isel(time=1) == 0
    assert ct["cn"].isel(time=1) == 4


def test_compute_categorical_stats():
    # Manual CT
    ct = xr.Dataset(
        data_vars=dict(
            hits=xr.DataArray([10, 0], dims="time"),
            misses=xr.DataArray([0, 0], dims="time"),
            fa=xr.DataArray([0, 0], dims="time"),
            cn=xr.DataArray([10, 10], dims="time"),
        )
    )
    
    stats = compute_categorical_stats(ct)
    
    # Time 0: Perfect forecast
    assert np.isclose(stats["pod"].isel(time=0), 1.0)
    assert np.isclose(stats["far"].isel(time=0), 0.0)
    assert np.isclose(stats["csi"].isel(time=0), 1.0)
    assert np.isclose(stats["bias"].isel(time=0), 1.0)
    
    # Time 1: All zeros (empty event) -> POD undefined (0/0), code handles specific ways? 
    # With eps=1e-6:
    # pod = 0 / (0 + 0 + eps) = 0.
    # far = 0 / (0 + 0 + eps) = 0.
    # csi = 0.
    assert np.isclose(stats["pod"].isel(time=1), 0.0)
