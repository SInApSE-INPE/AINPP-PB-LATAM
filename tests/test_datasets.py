from pathlib import Path

import pytest
import torch

from ainpp_pb_latam.datasets import AINPPPBLATAMDataset


class TestAINPPPBLATAMDataset:
    """Test suite for the AINPP-PB-LATAM dataset handling using Zarr inputs."""

    def test_dataset_minimal_zarr(self, small_zarr: Path) -> None:
        """Test that the dataset can load a Zarr store, retrieve patches, and maintain correct shapes."""
        # Arrange
        ds = AINPPPBLATAMDataset(
            zarr_path=str(small_zarr),
            group="train",
            input_timesteps=2,
            output_timesteps=1,
            patch_height=4,
            patch_width=4,
            patch_stride_h=4,
            patch_stride_w=4,
            steps_per_epoch=None,
            seed=0,
        )

        # Act
        length = len(ds)
        x, y = ds[0]

        # Assert
        assert length > 0
        assert x.shape == (2, 1, 4, 4)
        assert y.shape == (1, 1, 4, 4)
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_dataset_stride_iteration_deterministic(self, small_zarr: Path) -> None:
        """Test that iterating with stride parameters successfully fetches distinct temporal/spatial patches."""
        # Arrange
        ds = AINPPPBLATAMDataset(
            zarr_path=str(small_zarr),
            group="validation",
            input_timesteps=2,
            output_timesteps=1,
            patch_height=2,
            patch_width=2,
            patch_stride_h=2,
            patch_stride_w=2,
            steps_per_epoch=None,
            seed=0,
        )

        # Act
        first = ds[0][0]
        second = ds[1][0]

        # Assert
        # Ensure that sequential indexing fetches physically different spatial/temporal slices
        assert not torch.equal(first, second)

    def test_valid_mask_counts_each_pixel_once_and_masks_nan(self, tmp_path: Path) -> None:
        """Overlapping edge-snapped patches must not double count pixels; NaN targets are masked."""
        import numpy as np
        import xarray as xr

        target = np.ones((6, 4, 4), dtype=np.float32)
        target[3, 0, 0] = np.nan  # first target step of the t0=0 sample
        dims = ("time", "lat", "lon")
        ds = xr.Dataset(
            {"gsmap_nrt": (dims, np.ones_like(target)), "gsmap_mvk": (dims, target)},
            coords={"time": np.arange(6), "lat": np.arange(4.0), "lon": np.arange(4.0)},
        )
        path = tmp_path / "nan.zarr"
        ds.to_zarr(path, group="test", consolidated=True)

        data = AINPPPBLATAMDataset(
            zarr_path=str(path),
            group="test",
            input_timesteps=3,
            output_timesteps=1,
            stride=3,
            patch_height=3,
            patch_width=3,
            return_valid_mask=True,
        )
        # 4x4 domain, 3x3 patches -> origins (0,0), (0,1), (1,0), (1,1) overlap
        assert len(data.patch_origins) == 4

        coverage = np.zeros((4, 4), dtype=int)
        for k, (i0, j0) in enumerate(data.patch_origins):
            x, y, mask = data[k]
            assert mask.dtype == torch.bool and mask.shape == y.shape
            assert not torch.isnan(y).any()
            coverage[i0 : i0 + 3, j0 : j0 + 3] += mask[0, 0].numpy().astype(int)

        expected = np.ones((4, 4), dtype=int)
        expected[0, 0] = 0  # NaN target
        np.testing.assert_array_equal(coverage, expected)
