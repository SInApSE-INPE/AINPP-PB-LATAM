import numpy as np
import xarray as xr
import pandas as pd
import gzip
from pathlib import Path
import dask
from dask.diagnostics import ProgressBar
import dask.array as da
import zarr
from numcodecs import Blosc
import os

class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.region = config.region.name
        self.input_base = Path(config.paths.input_base)
        self.output_base = Path(config.paths.output_base)
        self.params_base = Path(config.paths.params_base)
        
        # Dimensions from config
        self.lat_dim = config.region.dims[0]
        self.lon_dim = config.region.dims[1]
        self.lat_range = config.region.lat_range
        self.lon_range = config.region.lon_range
        
        # Suffix options for MVK
        self.mvk_suffixes = [
            "0000.1.dat.gz", 
            "0000.0.dat.gz", 
            "1000.0.dat.gz"
        ]

    def find_gsmap_file(self, base_dir, timestamp, product):
        """Finds the GSMaP file path for a specific timestamp."""
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
            # NRT
            potential_path = base_dir / year_str / month_str / day_str / f"gsmap_nrt.{date_str}.{time_str}.dat.gz"
            if potential_path.exists():
                return potential_path
        return None

    def read_gsmap_data(self, file_path):
        """Reads a GSMaP file and returns a 2D numpy array (lat, lon)."""
        try:
            with gzip.open(file_path, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.float32).reshape((self.lat_dim, self.lon_dim))
                return np.nan_to_num(data, nan=0.0)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return np.zeros((self.lat_dim, self.lon_dim), dtype=np.float32)

    def run(self):
        print(f"Starting GSMaP preprocessing for region: {self.region}")
        
        # Combine years
        train_years = self.config.years.train
        val_years = self.config.years.val
        test_years = self.config.years.test
        all_years = sorted(train_years + val_years + test_years)
        
        print(f"Processing years: {all_years[0]} to {all_years[-1]}")
        full_time_range = pd.date_range(
            start=f"{all_years[0]}-01-01 00:00",
            end=f"{all_years[-1]}-12-31 23:00",
            freq='h'
        )

        lat_coords = np.linspace(self.lat_range[0], self.lat_range[1], self.lat_dim)
        lon_coords = np.linspace(self.lon_range[0], self.lon_range[1], self.lon_dim)

        input_dirs = {
            "mvk": self.input_base / f"gsmap_mvk-{self.region}",
            "nrt": self.input_base / f"gsmap_nrt-{self.region}",
        }
        
        output_zarr = self.output_base / f"gsmap_nrt+mvk_log_zscore_{self.region}.zarr"
        output_zarr.parent.mkdir(parents=True, exist_ok=True)

        datasets = {}
        timestamps_present = {}

        # Build datasets lazily
        for product, base_dir in input_dirs.items():
            print(f"\n🔹 Building dataset for: {product.upper()}")
            # Serialize the read function to avoid self pickle issues if any, 
            # though dask usually handles methods fine if object is picklable.
            # Use cleaner function or partial if needed. 
            # Dask delayed allows calling methods.
            
            delayed_reader = dask.delayed(self.read_gsmap_data)
            lazy_chunks, timestamps_ok = [], []

            for ts in full_time_range:
                filepath = self.find_gsmap_file(base_dir, ts, product)
                if filepath is None:
                    chunk = da.zeros((self.lat_dim, self.lon_dim), dtype=np.float32)
                else:
                    timestamps_ok.append(ts)
                    chunk = da.from_delayed(
                        delayed_reader(filepath),
                        shape=(self.lat_dim, self.lon_dim),
                        dtype=np.float32
                    )
                lazy_chunks.append(chunk)

            timestamps_present[product] = timestamps_ok
            dask_array = da.stack(lazy_chunks, axis=0)
            datasets[product] = xr.DataArray(
                dask_array,
                dims=("time", "lat", "lon"),
                coords={"time": full_time_range, "lat": lat_coords, "lon": lon_coords},
                name=f"gsmap_{product}"
            )

        # Temporal check
        print("\n🕒 Checking temporal alignment...")
        mvk_times = set(timestamps_present["mvk"])
        nrt_times = set(timestamps_present["nrt"])
        missing_in_mvk = sorted(list(nrt_times - mvk_times))
        missing_in_nrt = sorted(list(mvk_times - nrt_times))

        if missing_in_mvk or missing_in_nrt:
            print("❌ Inconsistency detected!")
            print(f"NRT only: {len(missing_in_mvk)}")
            print(f"MVK only: {len(missing_in_nrt)}")
            # exit(1) # Raise error instead
            raise RuntimeError("Temporal inconsistency detected.")
        else:
            print("✅ Timestamps match.")

        # Calc stats
        print("\nCalculating statistics in LOG domain (based on MVK - Train)...")
        train_start = f'{train_years[0]}-01-01'
        train_end = f'{train_years[-1]}-12-31'
        
        mvk_train = datasets["mvk"].sel(time=slice(train_start, train_end))
        mvk_train_log = np.log1p(mvk_train)

        with ProgressBar():
            print("  ↳ Calculating Mean/Std...")
            mean_log = mvk_train_log.mean(dim=("time", "lat", "lon")).compute()
            std_log = mvk_train_log.std(dim=("time", "lat", "lon")).compute()

        if std_log.values == 0:
            print("⚠️ Warning: Std is 0. Adjusting to 1.0.")
            std_log.values = 1.0

        print(f"Stats -> Mean: {mean_log.values:.4f}, Std: {std_log.values:.4f}")

        # Normalize
        print("\nApplying (Log1p -> Z-score)...")
        normalized = {}
        for k, da_in in datasets.items():
            da_log = np.log1p(da_in)
            normalized[k] = (da_log - mean_log) / std_log

        # Save
        compressor = Blosc(cname="zstd", clevel=self.config.processing.compression_level, shuffle=Blosc.BITSHUFFLE)
        encoding = {f"gsmap_{key}": {"compressor": compressor} for key in normalized.keys()}

        groups = {
            "train": slice(f'{train_years[0]}-01-01', f'{train_years[-1]}-12-31'),
            "validation": slice(f'{val_years[0]}-01-01', f'{val_years[0]}-12-31'),
            "test": slice(f'{test_years[0]}-01-01', f'{test_years[0]}-12-31'),
        }

        chunk_encoding = {
            'time': self.config.processing.chunk_time, 
            'lat': self.config.region.chunks.lat, 
            'lon': self.config.region.chunks.lon
        }

        for group_name, time_slice in groups.items():
            print(f"\n💾 Saving group '{group_name}' to {output_zarr}...")
            subset = {f"gsmap_{key}": da.sel(time=time_slice) for key, da in normalized.items()}
            ds_to_save = xr.Dataset(subset).chunk(chunk_encoding)
            
            with ProgressBar():
                ds_to_save.to_zarr(
                    output_zarr,
                    mode='a' if group_name != "train" else 'w', # Overwrite on first group (train), append others?
                    # Be careful: 'w' overwrites the whole store. We iterate.
                    # Usually saving separate groups into same root.
                    # 'w' wipes root. 'a' appends/modifies.
                    # First call 'w' if we want clean slate?
                    # The groups here are Zarr groups?
                    # original code: group=group_name.
                    # If we use mode='w' for first group 'train', it creates the store.
                    # Subsequent groups use mode='a'.
                    
                    group=group_name,
                    encoding=encoding,
                    consolidated=True,
                    zarr_version=2
                )

        # Save params
        self.params_base.mkdir(parents=True, exist_ok=True)
        np.save(self.params_base / f"gsmap_nrt+mvk_log_mean_{self.region}.npy", mean_log.values)
        np.save(self.params_base / f"gsmap_nrt+mvk_log_std_{self.region}.npy", std_log.values)

        print("\n✅ Process completed successfully!")
