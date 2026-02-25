import logging
import gzip
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import zarr
from numcodecs import Blosc
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """
    Modular pipeline for GSMaP data preprocessing.
    Handles data loading, temporal alignment, normalization (Log-Zscore), and Zarr storage.
    """

    def __init__(self, config: DictConfig):
        """
        Initializes the pipeline with a Hydra configuration.
        """
        self.config = config
        self.region_name = config.preprocessing.region.name
        self.lat_dim = config.preprocessing.region.lat_dim
        self.lon_dim = config.preprocessing.region.lon_dim
        self.lat_range = config.preprocessing.region.lat_range
        self.lon_range = config.preprocessing.region.lon_range
        
        self.input_base = Path(config.preprocessing.paths.input_base)
        self.output_zarr = Path(config.preprocessing.paths.output_zarr)
        self.params_dir = Path(config.preprocessing.paths.params_dir)
        
        self.mvk_suffixes = [
            "0000.1.dat.gz", 
            "0000.0.dat.gz", 
            "1000.0.dat.gz"
        ]

    def _find_file(self, base_dir: Path, timestamp: pd.Timestamp, product: str) -> Optional[Path]:
        """Finds the GSMaP file path for a specific timestamp and product."""
        date_str = timestamp.strftime('%Y%m%d')
        time_str = timestamp.strftime('%H%M')
        year_str, month_str, day_str = timestamp.strftime('%Y %m %d').split()

        if product == "mvk":
            base_filename = f"gsmap_mvk.{date_str}.{time_str}.v8"
            for suffix in self.mvk_suffixes:
                potential_path = base_dir / year_str / month_str / day_str / f"{base_filename}.{suffix}"
                if potential_path.exists():
                    return potential_path
        else:
            potential_path = base_dir / year_str / month_str / day_str / f"gsmap_nrt.{date_str}.{time_str}.dat.gz"
            if potential_path.exists():
                return potential_path
        return None

    def _read_data(self, file_path: Path) -> np.ndarray:
        """Reads a GSMaP file into a numpy array."""
        try:
            with gzip.open(file_path, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.float32).reshape((self.lat_dim, self.lon_dim))
                return np.nan_to_num(data, nan=0.0)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return np.zeros((self.lat_dim, self.lon_dim), dtype=np.float32)

    def build_dataset(self, product: str, time_range: pd.DatetimeIndex) -> xr.DataArray:
        """Builds a lazy Dask-backed xarray DataArray for a given product."""
        logger.info(f"Building lazy dataset for: {product.upper()}")
        
        base_dir = self.input_base / f"gsmap_{product}-{self.region_name}"
        lat_coords = np.linspace(self.lat_range[0], self.lat_range[1], self.lat_dim)
        lon_coords = np.linspace(self.lon_range[0], self.lon_range[1], self.lon_dim)

        delayed_reader = dask.delayed(self._read_data)
        lazy_chunks = []

        for ts in time_range:
            filepath = self._find_file(base_dir, ts, product)
            if filepath is None:
                chunk = da.zeros((self.lat_dim, self.lon_dim), dtype=np.float32)
            else:
                chunk = da.from_delayed(
                    delayed_reader(filepath),
                    shape=(self.lat_dim, self.lon_dim),
                    dtype=np.float32
                )
            lazy_chunks.append(chunk)

        dask_array = da.stack(lazy_chunks, axis=0)
        return xr.DataArray(
            dask_array,
            dims=("time", "lat", "lon"),
            coords={"time": time_range, "lat": lat_coords, "lon": lon_coords},
            name=f"gsmap_{product}"
        )

    def run(self):
        """Executes the full preprocessing pipeline."""
        logger.info(f"Starting pipeline for region: {self.region_name}")
        
        train_years = list(self.config.preprocessing.years.train)
        val_years = list(self.config.preprocessing.years.validation)
        test_years = list(self.config.preprocessing.years.test)
        all_years = sorted(train_years + val_years + test_years)
        
        time_range = pd.date_range(
            start=f"{all_years[0]}-01-01 00:00",
            end=f"{all_years[-1]}-12-31 23:00",
            freq='h'
        )

        # 1. Build Datasets
        ds_mvk = self.build_dataset("mvk", time_range)
        ds_nrt = self.build_dataset("nrt", time_range)
        datasets = {"mvk": ds_mvk, "nrt": ds_nrt}

        # 2. Statistics Calculation (Train only)
        logger.info("Calculating statistics in Log domain...")
        train_slice = slice(f"{train_years[0]}-01-01", f"{train_years[-1]}-12-31")
        mvk_train = ds_mvk.sel(time=train_slice)
        mvk_train_log = np.log1p(mvk_train)

        with ProgressBar():
            mean_log = mvk_train_log.mean(dim=("time", "lat", "lon")).compute()
            std_log = mvk_train_log.std(dim=("time", "lat", "lon")).compute()

        if std_log == 0:
            std_log = 1.0
        
        logger.info(f"Stats -> Mean: {float(mean_log):.4f}, Std: {float(std_log):.4f}")

        # 3. Apply Normalization
        normalized = {}
        for k, da_in in datasets.items():
            da_log = np.log1p(da_in)
            normalized[k] = (da_log - mean_log) / std_log

        # 4. Save to Zarr
        self.output_zarr.parent.mkdir(parents=True, exist_ok=True)
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        encoding = {f"gsmap_{key}": {"compressor": compressor} for key in normalized.keys()}

        groups = {
            "train": train_slice,
            "validation": slice(f"{val_years[0]}-01-01", f"{val_years[-1]}-12-31"),
            "test": slice(f"{test_years[0]}-01-01", f"{test_years[-1]}-12-31"),
        }

        chunk_encoding = {
            'time': self.config.preprocessing.chunks.time, 
            'lat': self.config.preprocessing.chunks.lat, 
            'lon': self.config.preprocessing.chunks.lon
        }

        for group_name, time_slice in groups.items():
            logger.info(f"Saving group '{group_name}'...")
            subset = {f"gsmap_{key}": da.sel(time=time_slice) for key, da in normalized.items()}
            ds_to_save = xr.Dataset(subset).chunk(chunk_encoding)
            
            with ProgressBar():
                ds_to_save.to_zarr(
                    self.output_zarr,
                    mode='a' if group_name != "train" else 'w',
                    group=group_name,
                    encoding=encoding,
                    consolidated=True,
                    zarr_version=2
                )

        # 5. Save Parameters
        self.params_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.params_dir / f"mean_log_{self.region_name}.npy", mean_log.values)
        np.save(self.params_dir / f"std_log_{self.region_name}.npy", std_log.values)
        
        logger.info("Preprocessing complete.")
