import pandas as pd
import pytest
import numpy as np
import xarray as xr
from src.evaluation.continuous_metrics import compute_basic_error_metrics, compute_spearman_time_series


@pytest.fixture
def dummy_data():
    """Creates dummy forecast and observation data."""
    times = pd.date_range("2023-01-01", periods=5, freq="H")
    lats = np.linspace(-10, 10, 10)
    lons = np.linspace(-10, 10, 10)
    
    # Random data
    np.random.seed(42)
    fct_data = np.random.rand(len(times), len(lats), len(lons))
    obs_data = np.random.rand(len(times), len(lats), len(lons))
    
    fct = xr.DataArray(
        fct_data,
        coords={"time": times, "lat": lats, "lon": lons},
        dims=("time", "lat", "lon"),
        name="precip_fct"
    )
    obs = xr.DataArray(
        obs_data,
        coords={"time": times, "lat": lats, "lon": lons},
        dims=("time", "lat", "lon"),
        name="precip_obs"
    )
    return fct, obs


def test_compute_basic_error_metrics(dummy_data):
    fct, obs = dummy_data
    ds_metrics = compute_basic_error_metrics(fct, obs)
    
    assert "mse" in ds_metrics
    assert "mae" in ds_metrics
    assert "rmse" in ds_metrics
    assert "bias" in ds_metrics
    
    # Check dimensions (time only, lat/lon reduced)
    assert ds_metrics["mse"].dims == ("time",)
    assert ds_metrics["mae"].dims == ("time",)
    
    # Check values (simple manual check)
    err = fct - obs
    expected_mse = (err**2).mean(dim=("lat", "lon"))
    np.testing.assert_allclose(ds_metrics["mse"].values, expected_mse.values, rtol=1e-5)


def test_compute_spearman_time_series(dummy_data):
    fct, obs = dummy_data
    rho = compute_spearman_time_series(fct, obs)
    
    assert rho.dims == ("time",)
    assert rho.size == 5
    
    # Check values are between -1 and 1 (or NaN)
    assert np.all((rho >= -1) | (rho <= 1) | np.isnan(rho))
