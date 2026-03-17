import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch
import zarr
from zarr.storage import DirectoryStore


@pytest.fixture(autouse=True)
def _set_seed() -> None:
    """
    Keep tests deterministic by setting seeds for random, numpy, and torch.
    This fixture runs automatically before every test.
    """
    seed: int = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    yield


@pytest.fixture
def tensor_pair() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Provide small input/target tensors for loss and metric tests.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A pair of random tensors of shape (2, 1, 8, 8).
    """
    x = torch.rand(2, 1, 8, 8)
    y = torch.rand(2, 1, 8, 8)
    return x, y


@pytest.fixture
def small_zarr(tmp_path: Path) -> Path:
    """
    Create a minimal GSMaP-style Zarr file with train/validation/test groups
    for dataset and dataloader testing.
    
    Args:
        tmp_path (Path): Pytest fixture providing a temporary directory.
        
    Returns:
        Path: Path to the generated minimal Zarr store.
    """
    time = np.arange(24)
    lat = np.linspace(-1.0, 1.0, 4)
    lon = np.linspace(-1.0, 1.0, 4)
    
    data_in = np.random.rand(len(time), len(lat), len(lon)).astype(np.float32)
    data_out = np.random.rand(len(time), len(lat), len(lon)).astype(np.float32)

    zarr_path = tmp_path / "gsmap_small.zarr"
    store = DirectoryStore(str(zarr_path))

    
    for group in ("train", "validation", "test"):
        g = zarr.group(store=store, path=group, overwrite=True)
        g.create_dataset("gsmap_nrt", data=data_in, chunks=data_in.shape, overwrite=True)
        g.create_dataset("gsmap_mvk", data=data_out, chunks=data_out.shape, overwrite=True)

    return zarr_path
