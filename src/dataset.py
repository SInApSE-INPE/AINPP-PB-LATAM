from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


Split = str  # expected: "train" | "validation" | "test"
PatchOrigin = Tuple[int, int] # (i0, j0)
TensorPair = Tuple[torch.Tensor, torch.Tensor] # (input, target)


@dataclass(frozen=True)
class DatasetConfig:
    """
    Configuration for dataset access and sampling.

    Attributes
    ----------
    zarr_path:
        Path to the Zarr store.
    group:
        Zarr group name (split), typically one of: {"train", "validation", "test"}.
    input_timesteps:
        Number of past timesteps used as model input.
    output_timesteps:
        Number of future timesteps used as target.
    stride:
        Temporal stride between consecutive samples. If None, it is inferred:
        - train: output_timesteps
        - validation/test: input_timesteps + output_timesteps
    patch_size:
        Spatial patch size (square patch).
    patch_stride:
        Spatial stride for patch grid (used for deterministic patching).
        For overlap, set patch_stride < patch_size (e.g., 50% overlap: patch_stride = patch_size // 2).
    steps_per_epoch:
        If provided, dataset uses random sampling and the length becomes steps_per_epoch.
    seed:
        Seed for random sampling.
    consolidated:
        Whether to open Zarr with consolidated metadata.
    input_var:
        Variable name for input (NRT).
    target_var:
        Variable name for target (MVK).
    """

    zarr_path: Union[str, Path]
    group: Split = "train"
    input_timesteps: int = 12
    output_timesteps: int = 6
    stride: Optional[int] = None
    patch_size: int = 320
    patch_stride: int = 320
    steps_per_epoch: Optional[int] = None
    seed: int = 42
    consolidated: bool = True
    input_var: str = "gsmap_nrt"
    target_var: str = "gsmap_mvk"


class AINPPPBLATAMDataset(Dataset[TensorPair]):
    """
    Spatiotemporal dataset for GSMaP stored in Zarr, using patch-based sampling.

    This dataset yields pairs (x, y) where:
      - x: input sequence from NRT  -> shape (Tin, 1, P, P)
      - y: target sequence from MVK -> shape (Tout, 1, P, P)

    Sampling behavior
    -----------------
    - If `steps_per_epoch` is provided: random sampling over (time_start, patch_origin).
      This is appropriate for training with spatial augmentation.
    - Otherwise: deterministic enumeration over the Cartesian product:
      (valid time starts) × (patch grid origins). This is appropriate for validation/test.

    Notes
    -----
    - This implementation uses xarray `.isel(...).to_numpy()` per sample.
      If you need higher throughput, consider:
        * caching chunk indices,
        * using dask-backed arrays carefully,
        * grouping indices per worker,
        * or prefetching with larger chunks.
    """

    def __init__(
        self,
        config: DatasetConfig,
        *,
        return_metadata: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.cfg = config
        self.return_metadata = return_metadata
        self.dtype = dtype

        self.zarr_path = Path(self.cfg.zarr_path)
        self.group = self.cfg.group

        self._validate_config()

        self.sequence_length = self.cfg.input_timesteps + self.cfg.output_timesteps
        self.stride = self._resolve_stride()

        # RNG for random sampling mode
        self.rng = np.random.default_rng(self.cfg.seed)

        logger.info(
            "Opening Zarr: path=%s group=%s stride_t=%d patch=%d patch_stride=%d consolidated=%s",
            str(self.zarr_path),
            self.group,
            self.stride,
            self.cfg.patch_size,
            self.cfg.patch_stride,
            self.cfg.consolidated,
        )

        self.ds = xr.open_zarr(
            self.zarr_path,
            group=self.group,
            consolidated=self.cfg.consolidated,
        )

        # Lazy arrays; do not materialize here.
        if self.cfg.input_var not in self.ds.variables:
            raise KeyError(f"Input variable '{self.cfg.input_var}' not found in dataset variables.")
        if self.cfg.target_var not in self.ds.variables:
            raise KeyError(f"Target variable '{self.cfg.target_var}' not found in dataset variables.")

        self.input_da = self.ds[self.cfg.input_var]
        self.target_da = self.ds[self.cfg.target_var]

        # replace all NaNs with zeros
        self.input_da = self.input_da.fillna(0.0)
        self.target_da = self.target_da.fillna(0.0)

        # Expect dims: time, lat, lon
        for dim in ("time", "lat", "lon"):
            if dim not in self.ds.sizes:
                raise ValueError(f"Expected dimension '{dim}' not found in dataset dims: {list(self.ds.sizes)}")

        self.H = int(self.ds.sizes["lat"])
        self.W = int(self.ds.sizes["lon"])
        self.total_timesteps = int(self.ds.sizes["time"])

        if self.cfg.patch_size > self.H or self.cfg.patch_size > self.W:
            raise ValueError(
                f"patch_size={self.cfg.patch_size} exceeds domain size ({self.H}x{self.W})."
            )

        # Valid temporal start indices
        max_t0 = self.total_timesteps - self.sequence_length
        if max_t0 < 0:
            raise ValueError(
                "Not enough timesteps to build one sequence: "
                f"total_timesteps={self.total_timesteps}, sequence_length={self.sequence_length}."
            )

        self.valid_t0: List[int] = list(range(0, max_t0 + 1, self.stride))

        # Patch grid origins (used for deterministic evaluation, also as sampling universe)
        self.patch_origins: List[PatchOrigin] = self._build_patch_grid(
            height=self.H,
            width=self.W,
            patch_size=self.cfg.patch_size,
            patch_stride=self.cfg.patch_stride,
        )

        logger.info(
            "Dataset ready: temporal_samples=%d spatial_patches=%d steps_per_epoch=%s",
            len(self.valid_t0),
            len(self.patch_origins),
            str(self.cfg.steps_per_epoch),
        )

    def _validate_config(self) -> None:
        if self.cfg.input_timesteps <= 0:
            raise ValueError("input_timesteps must be > 0.")
        if self.cfg.output_timesteps <= 0:
            raise ValueError("output_timesteps must be > 0.")
        if self.cfg.patch_size <= 0:
            raise ValueError("patch_size must be > 0.")
        if self.cfg.patch_stride <= 0:
            raise ValueError("patch_stride must be > 0.")
        if self.cfg.steps_per_epoch is not None and self.cfg.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be > 0 when provided.")
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Zarr path does not exist: {self.zarr_path}")

    def _resolve_stride(self) -> int:
        if self.cfg.stride is not None:
            if self.cfg.stride <= 0:
                raise ValueError("stride must be > 0.")
            return int(self.cfg.stride)

        # Default policy aligned with your original code
        if self.group == "train":
            return int(self.cfg.output_timesteps)
        return int(self.sequence_length)

    @staticmethod
    def _build_patch_grid(
        *,
        height: int,
        width: int,
        patch_size: int,
        patch_stride: int,
    ) -> List[PatchOrigin]:
        """
        Build a deterministic grid of patch origins (i0, j0) covering the full domain.

        The last patch is forced to land on the border to guarantee coverage.
        """
        max_i = height - patch_size
        max_j = width - patch_size

        i_list = list(range(0, max(max_i + 1, 1), patch_stride))
        j_list = list(range(0, max(max_j + 1, 1), patch_stride))

        if i_list[-1] != max_i:
            i_list.append(max_i)
        if j_list[-1] != max_j:
            j_list.append(max_j)

        return [(i, j) for i in i_list for j in j_list]

    def __len__(self) -> int:
        if self.cfg.steps_per_epoch is not None:
            return int(self.cfg.steps_per_epoch)
        return len(self.valid_t0) * len(self.patch_origins)

    def _sample_indices(self, idx: int) -> Tuple[int, int, int]:
        """
        Map dataset index to (t0, i0, j0).

        Random sampling is used when steps_per_epoch is provided; otherwise deterministic mapping.
        """
        if self.cfg.steps_per_epoch is not None:
            t0 = self.valid_t0[int(self.rng.integers(0, len(self.valid_t0)))]
            i0, j0 = self.patch_origins[int(self.rng.integers(0, len(self.patch_origins)))]
            return t0, i0, j0

        n_patches = len(self.patch_origins)
        t_index = idx // n_patches
        p_index = idx % n_patches

        t0 = self.valid_t0[t_index]
        i0, j0 = self.patch_origins[p_index]
        return t0, i0, j0

    def __getitem__(self, idx: int):
        t0, i0, j0 = self._sample_indices(idx)

        tin = self.cfg.input_timesteps
        tout = self.cfg.output_timesteps
        P = self.cfg.patch_size

        t_in_end = t0 + tin
        t_out_end = t0 + tin + tout
        i1 = i0 + P
        j1 = j0 + P

        # Slice input/target
        x_np = (
            self.input_da.isel(time=slice(t0, t_in_end), lat=slice(i0, i1), lon=slice(j0, j1))
            .to_numpy()
        )
        y_np = (
            self.target_da.isel(time=slice(t_in_end, t_out_end), lat=slice(i0, i1), lon=slice(j0, j1))
            .to_numpy()
        )

        # Add channel dimension -> (T, 1, H, W)
        x = torch.from_numpy(x_np[:, None, :, :]).to(dtype=self.dtype)
        y = torch.from_numpy(y_np[:, None, :, :]).to(dtype=self.dtype)

        if not self.return_metadata:
            return x, y

        meta = {"t0": t0, "i0": i0, "j0": j0}
        return x, y, meta


def create_gsmap_dataloaders(
    *,
    zarr_path: Union[str, Path],
    input_timesteps: int = 12,
    output_timesteps: int = 6,
    batch_size: int = 4,
    num_workers: int = 4,
    train_stride: int = 1,
    eval_stride: int = 6,
    patch_size: int = 320,
    patch_stride: int = 320,
    steps_per_epoch: Optional[int] = None,
    seed: int = 42,
    consolidated: bool = True,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = 2,
    drop_last_train: bool = True,
    dtype: torch.dtype = torch.float32,
) -> Dict[Split, DataLoader]:
    """
    Factory to create DataLoaders for train/validation/test.

    Parameters
    ----------
    zarr_path:
        Path to Zarr store.
    train_stride, eval_stride:
        Temporal stride policy for each split (explicit, to avoid hidden behavior).
    steps_per_epoch:
        When provided, train dataset uses random sampling; validation/test remain deterministic.
    pin_memory:
        If None, enabled automatically when CUDA is available.
    persistent_workers:
        If None, enabled when num_workers > 0.
    prefetch_factor:
        Only applies when num_workers > 0. Set None to use PyTorch defaults.
    drop_last_train:
        Drop last incomplete batch in training (often desirable for DDP stability).

    Returns
    -------
    dict
        Mapping split -> DataLoader
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    loaders: Dict[Split, DataLoader] = {}
    for split in ("train", "validation", "test"):
        stride = train_stride if split == "train" else eval_stride
        split_steps = steps_per_epoch if split == "train" else None

        cfg = DatasetConfig(
            zarr_path=zarr_path,
            group=split,
            input_timesteps=input_timesteps,
            output_timesteps=output_timesteps,
            stride=stride,
            patch_size=patch_size,
            patch_stride=patch_stride,
            steps_per_epoch=split_steps,
            seed=seed,
            consolidated=consolidated,
        )

        dataset = AINPPPBLATAMDataset(cfg, dtype=dtype)

        dl_kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        # prefetch_factor is only valid when num_workers > 0
        if num_workers > 0 and prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = prefetch_factor

        if split == "train":
            dl_kwargs["drop_last"] = drop_last_train

        loaders[split] = DataLoader(**dl_kwargs)

    return loaders


def _configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


if __name__ == "__main__":
    _configure_logging(logging.INFO)

    zarr_path = "/prj/ideeps/adriano.almeida/data/ainpp/legacy/AINPP-PB-LATAM.zarr"

    # Dataset sanity check
    ds_cfg = DatasetConfig(
        zarr_path=zarr_path,
        group="train",
        input_timesteps=12,
        output_timesteps=6,
        stride=1,  # explicit for clarity
        patch_size=320,
        patch_stride=320,
        steps_per_epoch=200,  # random sampling mode example
        seed=42,
        consolidated=True,
    )
    dataset = AINPPPBLATAMDataset(ds_cfg)

    logger.info("Dataset length: %d", len(dataset))
    x, y = dataset[0]
    logger.info("Sample shapes: x=%s y=%s", tuple(x.shape), tuple(y.shape))

    # DataLoaders sanity check
    loaders = create_gsmap_dataloaders(
        zarr_path=zarr_path,
        batch_size=2,
        num_workers=0,
        input_timesteps=12,
        output_timesteps=6,
        train_stride=1,
        eval_stride=6,
        patch_size=320,
        patch_stride=320,
        steps_per_epoch=200,
    )

    xb, yb = next(iter(loaders["train"]))
    logger.info("Batch shapes (train): x=%s y=%s", tuple(xb.shape), tuple(yb.shape))
