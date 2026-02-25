import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch
import zarr


@pytest.fixture(autouse=True)
def _set_seed():
    """Keep tests deterministic."""
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    yield


@pytest.fixture
def tensor_pair() -> Tuple[torch.Tensor, torch.Tensor]:
    """Small input/target tensors for loss tests."""
    x = torch.rand(2, 1, 8, 8)
    y = torch.rand(2, 1, 8, 8)
    return x, y


@pytest.fixture
def small_zarr(tmp_path: Path) -> Path:
    """Create a minimal GSMaP-style Zarr with train/validation/test groups."""
    time = np.arange(24)
    lat = np.linspace(-1.0, 1.0, 4)
    lon = np.linspace(-1.0, 1.0, 4)
    data_in = np.random.rand(len(time), len(lat), len(lon)).astype(np.float32)
    data_out = np.random.rand(len(time), len(lat), len(lon)).astype(np.float32)

    zarr_path = tmp_path / "gsmap_small.zarr"
    from zarr.storage import LocalStore

    store = LocalStore(str(zarr_path))
    for group in ("train", "validation", "test"):
        g = zarr.group(store=store, path=group, overwrite=True)
        g.create_dataset("gsmap_nrt", data=data_in, chunks=data_in.shape, overwrite=True)
        g.create_dataset("gsmap_mvk", data=data_out, chunks=data_out.shape, overwrite=True)

    return zarr_path
