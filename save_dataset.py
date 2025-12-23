# def inverse_transform(tensor_norm, mean_log, std_log):
#     # Reverts Log-Zscore normalization to mm/h.
#     # Formula: x_mmh = exp(x_norm * std + mean) - 1
#     # 1. Undo Z-Score
#     x_log = tensor_norm * std_log + mean_log
    
#     # 2. Undo Log (exp(x) - 1)
#     x_mmh = np.expm1(x_log)
    
#     # 3. Ensure non-negativity (numerical correction)
#     return np.maximum(x_mmh, 0.0)

print("Starting GSMaP data preprocessing (MVK + NRT) with Log-Normalization...")
import xarray as xr
import numpy as np
import pandas as pd
import gzip
from pathlib import Path
import dask
from dask.diagnostics import ProgressBar
import dask.array as da
import zarr
from numcodecs import Blosc

# ============================================================
# MAIN SETTINGS
# ============================================================

# region = "ainpp-latin-america"
# region = "ainpp-south-america"
region = "ainpp-amazon-basin"

INPUT_BASE_DIRS = {
    "mvk": Path(f"/prj/ideeps/adriano.almeida/data/ainpp/regions/gsmap_mvk-{region}"),
    "nrt": Path(f"/prj/ideeps/adriano.almeida/data/ainpp/regions/gsmap_nrt-{region}"),
}

# CHANGE: Changed filename to indicate log_zscore
OUTPUT_ZARR_STORE = Path(f"/prj/ideeps/adriano.almeida/data/ainpp/legacy/gsmap_nrt+mvk_log_zscore_{region}.zarr")
OUTPUT_ZARR_STORE.parent.mkdir(parents=True, exist_ok=True)

TRAIN_YEARS = [2018, 2019, 2020, 2021, 2022]
VALIDATION_YEARS = [2023]
TEST_YEARS = [2024]

# latin america
# LAT_DIM, LON_DIM = 880, 970
# LAT_MIN, LAT_MAX = -55, 33
# LON_MIN, LON_MAX = -120, -23

# CHUNK_TIME = 360 # 360 hours = 15 days
# CHUNK_LAT = 440 # 440 latitudes = 440/880 = 50%
# CHUNK_LON =  485 # 485 longitudes = 485/970 = 50%

# south america
# LAT_DIM, LON_DIM = 680, 500
# LAT_MIN, LAT_MAX = -55, 13
# LON_MIN, LON_MAX = -83, -33

# CHUNK_TIME = 360 # 360 hours = 15 days
# CHUNK_LAT = 340 # 340 latitudes = 340/680 = 50%
# CHUNK_LON = 250 # 250 longitudes = 250/500 = 50%

# amazon basin settings
LAT_DIM, LON_DIM = 300, 360
LAT_MIN, LAT_MAX = -21, 9
LON_MIN, LON_MAX = -80, -44

CHUNK_TIME = 360 # 360 hours = 15 days
CHUNK_LAT = 150 # 150 latitudes = 150/300 = 50%
CHUNK_LON = 180 # 180 longitudes = 180/360 = 50%



MVK_SUFFIX_OPTIONS = [
    "0000.1.dat.gz", # 2018, 2019, 2020, 2021
    "0000.0.dat.gz", # 2021, 2022, 2023 
    "1000.0.dat.gz"  # 2023, 2024
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_gsmap_file(base_dir, timestamp, product):
    """Finds the GSMaP file path for a specific timestamp."""
    date_str = timestamp.strftime('%Y%m%d')
    time_str = timestamp.strftime('%H%M')
    year_str, month_str, day_str = timestamp.strftime('%Y %m %d').split()

    if product == "mvk":
        base_filename = f"gsmap_mvk.{date_str}.{time_str}.v8"
        for suffix in MVK_SUFFIX_OPTIONS:
            potential_path = base_dir / year_str / month_str / day_str / f"{base_filename}.{suffix}"
            if potential_path.exists():
                return potential_path
    else:
        # NRT -> no extra suffix
        potential_path = base_dir / year_str / month_str / day_str / f"gsmap_nrt.{date_str}.{time_str}.dat.gz"
        if potential_path.exists():
            return potential_path
    # print(f"Warning: Missing file: {potential_path}")
    return None

def read_gsmap_data(file_path):
    """Reads a GSMaP file and returns a 2D numpy array (lat, lon)."""
    try:
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.float32).reshape((LAT_DIM, LON_DIM))
            # Ensures no raw NaNs from the file
            return np.nan_to_num(data, nan=0.0)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return np.zeros((LAT_DIM, LON_DIM), dtype=np.float32)

# ============================================================
# DATASET CONSTRUCTION
# ============================================================

if __name__ == "__main__":
    all_years = sorted(TRAIN_YEARS + VALIDATION_YEARS + TEST_YEARS)
    print(f"Processing years: {all_years[0]} to {all_years[-1]}")
    full_time_range = pd.date_range(
        start=f"{all_years[0]}-01-01 00:00",
        end=f"{all_years[-1]}-12-31 23:00",
        freq='h'
    )

    lat_coords = np.linspace(LAT_MIN, LAT_MAX, LAT_DIM)
    lon_coords = np.linspace(LON_MIN, LON_MAX, LON_DIM)

    datasets = {}
    timestamps_present = {}

    for product, base_dir in INPUT_BASE_DIRS.items():
        print(f"\n🔹 Building dataset for: {product.upper()}")
        delayed_reader = dask.delayed(read_gsmap_data)
        lazy_chunks, timestamps_ok = [], []

        for ts in full_time_range:
            filepath = find_gsmap_file(base_dir, ts, product)
            if filepath is None:
                # If file doesn't exist, fill with zeros.
                # Note: log1p(0) = 0, so this is safe for subsequent transformation.
                chunk = da.zeros((LAT_DIM, LON_DIM), dtype=np.float32)
            else:
                timestamps_ok.append(ts)
                chunk = da.from_delayed(
                    delayed_reader(filepath),
                    shape=(LAT_DIM, LON_DIM),
                    dtype=np.float32
                )
            lazy_chunks.append(chunk)

        timestamps_present[product] = timestamps_ok
        dask_array = da.stack(lazy_chunks, axis=0)
        var_name = f"gsmap_{product}"
        datasets[product] = xr.DataArray(
            dask_array,
            dims=("time", "lat", "lon"),
            coords={"time": full_time_range, "lat": lat_coords, "lon": lon_coords},
            name=var_name
        )

    # ============================================================
    # TEMPORAL ALIGNMENT CHECK
    # ============================================================

    print("\n🕒 Checking temporal alignment between MVK and NRT...")
    mvk_times = set(timestamps_present["mvk"])
    nrt_times = set(timestamps_present["nrt"])
    missing_in_mvk = sorted(list(nrt_times - mvk_times))
    missing_in_nrt = sorted(list(mvk_times - nrt_times))

    if missing_in_mvk or missing_in_nrt:
        print("❌ Inconsistency detected between timestamps!")
        print(f"→ Files present in NRT but missing in MVK: {len(missing_in_mvk)}")
        print(f"→ Files present in MVK but missing in NRT: {len(missing_in_nrt)}")
        print("Aborting execution to avoid Zarr misalignment.")
        exit(1)
    else:
        print("✅ Timestamps match exactly between MVK and NRT.")

    # ============================================================
    # STATISTICS CALCULATION (LOG-TRANSFORMED)
    # ============================================================

    print("\nCalculating statistics in LOG domain (based on MVK - Train)...")
    
    # Selects only training data to prevent data leakage
    mvk_train = datasets["mvk"].sel(time=slice(f'{TRAIN_YEARS[0]}-01-01', f'{TRAIN_YEARS[-1]}-12-31'))
    
    # [IMPORTANT] Applies Log1p (log(x+1)) lazily
    mvk_train_log = np.log1p(mvk_train)

    with ProgressBar():
        print("  ↳ Calculating Mean (Log)...")
        mean_log = mvk_train_log.mean(dim=("time", "lat", "lon")).compute()
        print("  ↳ Calculating Standard Deviation (Log)...")
        std_log = mvk_train_log.std(dim=("time", "lat", "lon")).compute()

    if std_log.values == 0:
        print("⚠️ Warning: Standard deviation is 0. Adjusting to 1.0.")
        std_log.values = 1.0

    print(f"Log-Transformed Statistics -> Mean: {mean_log.values:.4f}, Std: {std_log.values:.4f}")

    # ============================================================
    # TRANSFORMATION AND NORMALIZATION APPLICATION
    # ============================================================

    print("\nApplying (Log1p -> Z-score) to datasets...")
    normalized = {}
    
    for k, da_in in datasets.items():
        # 1. Logarithmic Transformation
        da_log = np.log1p(da_in)
        # 2. Z-Score Normalization using log statistics
        normalized[k] = (da_log - mean_log) / std_log

    # ============================================================
    # SAVING TO ZARR
    # ============================================================

    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    encoding = {f"gsmap_{key}": {"compressor": compressor} for key in normalized.keys()}

    groups = {
        "train": slice(f'{TRAIN_YEARS[0]}-01-01', f'{TRAIN_YEARS[-1]}-12-31'),
        "validation": slice(f'{VALIDATION_YEARS[0]}-01-01', f'{VALIDATION_YEARS[0]}-12-31'),
        "test": slice(f'{TEST_YEARS[0]}-01-01', f'{TEST_YEARS[0]}-12-31'),
    }

    # [IMPORTANT] Optimized chunking for Deep Learning
    # Lat/Lon cover the whole image (avoids spatial fragmentation)
    # Time covers 48h (good balance for loading temporal sequences)
    chunk_encoding = {'time': CHUNK_TIME, 'lat': CHUNK_LAT, 'lon': CHUNK_LON}

    for group_name, time_slice in groups.items():
        print(f"\n💾 Saving group '{group_name}' to {OUTPUT_ZARR_STORE}...")
        subset = {f"gsmap_{key}": da.sel(time=time_slice) for key, da in normalized.items()}
        ds_to_save = xr.Dataset(subset)
        
        # Applies chunking
        ds_to_save = ds_to_save.chunk(chunk_encoding)
        
        with ProgressBar():
            ds_to_save.to_zarr(
                OUTPUT_ZARR_STORE,
                mode='a' if group_name != "train" else 'w',
                group=group_name,
                encoding=encoding,
                consolidated=True,
                zarr_version=2
            )

    # ============================================================
    # SAVE NORMALIZATION PARAMETERS
    # ============================================================

    PARAMS_DIR = Path("/prj/ideeps/adriano.almeida/data/ainpp/legacy/model_params")
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Saving with explicit names to avoid future confusion
    np.save(PARAMS_DIR / f"gsmap_nrt+mvk_log_mean_{region}.npy", mean_log.values)
    np.save(PARAMS_DIR / f"gsmap_nrt+mvk_log_std_{region}.npy", std_log.values)

    print("\n✅ Process completed successfully!")
    print(f"Dataset saved at: {OUTPUT_ZARR_STORE}")
    print(f"LOG Parameters saved at: {PARAMS_DIR}")