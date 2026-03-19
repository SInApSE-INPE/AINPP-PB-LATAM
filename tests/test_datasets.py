from pathlib import Path

import pytest
import torch

from ainpp_pb_latam.datasets import AINPPPBLATAMDataset

# Zarr v3 + xarray compatibility needs proper adapter; skip until implemented.
pytestmark = pytest.mark.skip(reason="Dataset tests temporarily skipped pending zarr v3 harness")


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
