"""
Dataset module for GSMaP Precipitation Data stored in Zarr format.
Supports rectangular patching, deterministic/random sampling, and Hydra integration.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Auxiliary Types
PatchOrigin = Tuple[int, int]
TensorPair = Tuple[torch.Tensor, torch.Tensor]


class AINPPPBLATAMDataset(Dataset[TensorPair]):
    """
    Spatiotemporal Dataset for GSMaP Zarr data with flexible rectangular patching.

    This dataset handles:
    1.  Lazy loading from Zarr stores (using Xarray).
    2.  Temporal slicing (Past -> Future).
    3.  Spatial slicing (Rectangular Patches).
    4.  Deterministic iteration (for validation/testing).
    5.  Random sampling (for training with `steps_per_epoch`).

    Attributes:
        ds (xr.Dataset): The opened xarray Dataset.
        valid_t0 (List[int]): List of valid start indices for time sequences.
        patch_origins (List[PatchOrigin]): List of (lat, lon) start indices for spatial patches.
    """

    def __init__(
        self,
        zarr_path: str,
        group: str = "train",
        input_timesteps: int = 12,
        output_timesteps: int = 6,
        stride: Optional[int] = None,
        # Rectangular Patch Parameters
        patch_height: Optional[int] = 320,
        patch_width: Optional[int] = 320,
        patch_stride_h: Optional[int] = None,
        patch_stride_w: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        seed: int = 42,
        consolidated: bool = True,
        input_var: str = "gsmap_nrt",
        target_var: str = "gsmap_mvk",
        dtype: str = "float32",
        return_metadata: bool = False,
        return_valid_mask: bool = False,
    ) -> None:
        """
        Initializes the dataset.

        Args:
            zarr_path (str): Path to the Zarr store.
            group (str): Zarr group/split to open (e.g., 'train', 'validation').
            input_timesteps (int): Number of history frames (Tin).
            output_timesteps (int): Number of forecast frames (Tout).
            stride (Optional[int]): Temporal stride between samples.
                Defaults to output_timesteps for train, sequence_length for eval.
            patch_height (Optional[int]): Height of spatial patch. Defaults to 320.
            patch_width (Optional[int]): Width of spatial patch. Defaults to 320.
            patch_stride_h (Optional[int]): Vertical stride for patch generation.
                If None, uses patch_height (no overlap).
            patch_stride_w (Optional[int]): Horizontal stride for patch generation.
                If None, uses patch_width (no overlap).
            steps_per_epoch (Optional[int]): If set, enables Random Sampling mode and
                ``len(dataset)`` equals this value, i.e. it is the number of SAMPLES (not
                batches) per epoch. If None, uses Deterministic Iteration.
            seed (int): Random seed for sampling reproducibility. In Random Sampling mode
                sample ``idx`` of epoch ``e`` is drawn from an RNG seeded with
                ``(seed, e, idx)``: independent of DataLoader workers, different every
                epoch (call ``set_epoch``) and reproducible.
            consolidated (bool): Whether to use consolidated metadata for Zarr.
            input_var (str): Variable name for input data.
            target_var (str): Variable name for target data.
            dtype (str): Target PyTorch data type (e.g., 'float32', 'float16').
            return_metadata (bool): If True, __getitem__ also returns a metadata dict.
            return_valid_mask (bool): If True, __getitem__ also returns a boolean mask
                shaped like ``y`` that is True only where the target is not NaN and the
                pixel is "owned" by this patch (overlapping patches never count the same
                domain pixel twice). Intended for evaluation in deterministic mode.

        Returns order of __getitem__: ``(x, y[, mask][, meta])``.
        """
        super().__init__()

        # 1. Basic Configuration
        self.zarr_path = Path(zarr_path)
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Path not found: {self.zarr_path}")

        self.group = group
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.steps_per_epoch = steps_per_epoch
        self.consolidated = consolidated
        self.input_var = input_var
        self.target_var = target_var
        self.return_metadata = return_metadata
        self.return_valid_mask = return_valid_mask

        # Type conversion
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        # A single shared RNG would be copied into every DataLoader worker (and re-copied
        # every epoch), producing duplicated and repeated samples. Draws are instead derived
        # per (seed, epoch, idx) in __getitem__.
        self.seed = int(seed)
        self.epoch = 0

        # Patch Config (Stored to resolve after opening Zarr)
        self._cfg_patch_h = patch_height
        self._cfg_patch_w = patch_width
        self._cfg_stride_h = patch_stride_h
        self._cfg_stride_w = patch_stride_w

        # Temporal Resolution
        self.sequence_length = self.input_timesteps + self.output_timesteps
        self.stride_t = self._resolve_temporal_stride(stride)

        # 2. Open Dataset
        logger.info(f"Opening Zarr: {self.zarr_path} (Group: {self.group})")
        self.ds = xr.open_zarr(
            self.zarr_path,
            group=self.group,
            consolidated=self.consolidated,
        )

        # Validation & NaN Handling
        if self.input_var not in self.ds.variables:
            raise KeyError(f"Input variable '{self.input_var}' not found in dataset.")

        # Lazy loading. Input NaNs -> 0.0; target NaNs are kept until __getitem__ so that
        # missing observations can be masked out of the evaluation instead of counted as "no rain".
        self.input_da = self.ds[self.input_var].fillna(0.0)
        self.target_da = self.ds[self.target_var]

        # 3. Domain Dimensions
        self.H = int(self.ds.sizes["lat"])
        self.W = int(self.ds.sizes["lon"])
        self.total_timesteps = int(self.ds.sizes["time"])

        # 4. Resolve Rectangular Patches (Now that H and W are known)
        self.patch_h, self.patch_w = self._resolve_patch_shape(self.H, self.W)
        self.stride_h, self.stride_w = self._resolve_patch_stride(self.patch_h, self.patch_w)

        # 5. Initialize Indices
        self._init_indices()

        logger.info(
            f"Dataset Ready: {len(self)} samples | "
            f"Patch: {self.patch_h}x{self.patch_w} | "
            f"Grid Stride: {self.stride_h}x{self.stride_w}"
        )

    def _resolve_temporal_stride(self, explicit_stride: Optional[int]) -> int:
        """Determines the temporal stride based on split type if not provided."""
        if explicit_stride is not None:
            return int(explicit_stride)
        if self.group == "train":
            return int(self.output_timesteps)
        return int(self.sequence_length)

    def _resolve_patch_shape(self, H: int, W: int) -> Tuple[int, int]:
        """Resolves patch dimensions, defaulting to full image if None."""
        ph = self._cfg_patch_h if self._cfg_patch_h is not None else H
        pw = self._cfg_patch_w if self._cfg_patch_w is not None else W

        if ph <= 0 or pw <= 0:
            raise ValueError("Patch dimensions must be > 0.")
        if ph > H or pw > W:
            raise ValueError(f"Patch ({ph}x{pw}) exceeds domain size ({H}x{W}).")
        return int(ph), int(pw)

    def _resolve_patch_stride(self, ph: int, pw: int) -> Tuple[int, int]:
        """Resolves patch strides, defaulting to patch size (no overlap) if None."""
        sh = self._cfg_stride_h if self._cfg_stride_h is not None else ph
        sw = self._cfg_stride_w if self._cfg_stride_w is not None else pw
        return int(sh), int(sw)

    def _init_indices(self):
        """Initializes valid temporal and spatial indices."""
        # Temporal
        max_t0 = self.total_timesteps - self.sequence_length
        if max_t0 < 0:
            raise ValueError(
                f"Insufficient timesteps: {self.total_timesteps} < {self.sequence_length}"
            )
        self.valid_t0 = list(range(0, max_t0 + 1, self.stride_t))

        # Spatial (Deterministic Grid)
        self.patch_origins = self._build_patch_grid(
            height=self.H,
            width=self.W,
            patch_h=self.patch_h,
            patch_w=self.patch_w,
            stride_h=self.stride_h,
            stride_w=self.stride_w,
        )
        self.ownership_masks = self._build_ownership_masks() if self.return_valid_mask else None

    def _build_ownership_masks(self) -> List[np.ndarray]:
        """
        Assigns every domain pixel to the first patch that covers it.

        The edge-snapped patch grid overlaps near the borders; without this, pixels in
        the overlap would be counted twice by the evaluation.
        """
        owner = np.full((self.H, self.W), -1, dtype=np.int64)
        for k, (i0, j0) in enumerate(self.patch_origins):
            window = owner[i0 : i0 + self.patch_h, j0 : j0 + self.patch_w]
            window[window == -1] = k
        return [
            owner[i0 : i0 + self.patch_h, j0 : j0 + self.patch_w] == k
            for k, (i0, j0) in enumerate(self.patch_origins)
        ]

    @staticmethod
    def _build_patch_grid(height, width, patch_h, patch_w, stride_h, stride_w) -> List[PatchOrigin]:
        """Builds a list of (lat, lon) start coordinates covering the domain."""
        max_i = height - patch_h
        max_j = width - patch_w

        i_list = list(range(0, max(max_i + 1, 1), stride_h))
        j_list = list(range(0, max(max_j + 1, 1), stride_w))

        # Ensure edge coverage (last patch snaps to the border)
        if i_list[-1] != max_i:
            i_list.append(max_i)
        if j_list[-1] != max_j:
            j_list.append(max_j)

        return [(i, j) for i in i_list for j in j_list]

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch used to seed Random Sampling, so each epoch draws new samples.

        Must be called in the main process before iterating the DataLoader (workers are
        re-created per epoch and receive a copy of this value). Datasets that never call
        it (e.g. validation) keep a fixed, reproducible random subset.
        """
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.steps_per_epoch is not None:
            return int(self.steps_per_epoch)
        return len(self.valid_t0) * len(self.patch_origins)

    def __getitem__(self, idx: int) -> Tuple[Union[torch.Tensor, Dict], ...]:
        # Sampling Logic
        if self.steps_per_epoch is not None:
            # Random Sampling (Training Mode)
            rng = np.random.default_rng([self.seed, self.epoch, int(idx)])
            t0 = self.valid_t0[int(rng.integers(0, len(self.valid_t0)))]
            patch_idx = int(rng.integers(0, len(self.patch_origins)))
        else:
            # Deterministic Iteration (Validation Mode)
            n_patches = len(self.patch_origins)
            t0 = self.valid_t0[idx // n_patches]
            patch_idx = idx % n_patches
        i0, j0 = self.patch_origins[patch_idx]

        # Slicing Dimensions
        ph, pw = self.patch_h, self.patch_w
        t_in_end = t0 + self.input_timesteps
        t_out_end = t_in_end + self.output_timesteps

        i1 = i0 + ph
        j1 = j0 + pw

        # Xarray Slicing -> Numpy
        x_np = self.input_da.isel(
            time=slice(t0, t_in_end), lat=slice(i0, i1), lon=slice(j0, j1)
        ).to_numpy()

        y_np = self.target_da.isel(
            time=slice(t_in_end, t_out_end), lat=slice(i0, i1), lon=slice(j0, j1)
        ).to_numpy()
        y_valid = np.isfinite(y_np)
        y_np = np.where(y_valid, y_np, 0.0)

        # Convert to Tensor (Add channel dim: T, C, H, W)
        x = torch.from_numpy(x_np[:, None, :, :]).to(dtype=self.dtype)
        y = torch.from_numpy(y_np[:, None, :, :]).to(dtype=self.dtype)

        out: List[Union[torch.Tensor, Dict]] = [x, y]
        if self.return_valid_mask:
            assert self.ownership_masks is not None
            mask_np = y_valid & self.ownership_masks[patch_idx][None, :, :]
            out.append(torch.from_numpy(mask_np[:, None, :, :]))
        if self.return_metadata:
            out.append({"t0": t0, "i0": i0, "j0": j0, "ph": ph, "pw": pw})

        return tuple(out)
