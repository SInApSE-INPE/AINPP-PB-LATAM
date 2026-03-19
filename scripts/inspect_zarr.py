import zarr
import pandas as pd
import sys
import numpy as np


def inspect(path):
    print(f"Inspecting: {path}")
    try:
        store = zarr.open(path, mode="r")
        print("Type:", type(store))
        if isinstance(store, zarr.Group):
            print("Keys:", list(store.keys()))

            if "time" in store:
                t = store["time"]
                print("Time shape:", t.shape)
                print("Time dtype:", t.dtype)
                print("Time attrs:", dict(t.attrs))
                try:
                    # Attempt to convert to datetime
                    times = pd.to_datetime(t[:])
                    print("Date Range:", times.min(), "to", times.max())
                    print("Sample values:", times[:5])
                except Exception as e:
                    print(f"Error converting time to datetime: {e}")
                    print("Raw time values (first 5):", t[:5])
            else:
                print("'time' key not found in group.")
        elif isinstance(store, zarr.core.Array):
            print("Is Array. Shape:", store.shape)
        else:
            print("Unknown Zarr object.")

    except Exception as e:
        print(f"Failed to open/read Zarr: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect(sys.argv[1])
    else:
        print("Usage: python inspect_zarr.py <path>")
