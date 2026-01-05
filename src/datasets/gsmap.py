import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)

PatchOrigin = Tuple[int, int]
TensorPair = Tuple[torch.Tensor, torch.Tensor]


class AINPPPBLATAMDataset(Dataset[TensorPair]):
    def __init__(
        self,
        zarr_path: str,
        group: str = "train",
        input_timesteps: int = 12,
        output_timesteps: int = 6,
        stride: Optional[int] = None,
        patch_size: int = 320,
        patch_stride: int = 320,
        steps_per_epoch: Optional[int] = None,
        seed: int = 42,
        consolidated: bool = True,
        input_var: str = "gsmap_nrt",
        target_var: str = "gsmap_mvk",
        dtype: str = "float32",
    ) -> None:
        super().__init__()
        self.zarr_path = Path(zarr_path)
        
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Path not found: {self.zarr_path}")

        self.group = group
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.steps_per_epoch = steps_per_epoch
        self.consolidated = consolidated
        self.input_var = input_var
        self.target_var = target_var
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        self.sequence_length = self.input_timesteps + self.output_timesteps
        self.stride = self._resolve_stride(stride)
        self.rng = np.random.default_rng(seed)

        logger.info(f"Opening Zarr: {self.zarr_path} (Group: {self.group})")
        
        self.ds = xr.open_zarr(
            self.zarr_path,
            group=self.group,
            consolidated=self.consolidated,
        )

        if self.input_var not in self.ds.variables:
            raise KeyError(f"Input '{self.input_var}' not found.")
        
        self.input_da = self.ds[self.input_var].fillna(0.0)
        self.target_da = self.ds[self.target_var].fillna(0.0)

        self.H = int(self.ds.sizes["lat"])
        self.W = int(self.ds.sizes["lon"])
        self.total_timesteps = int(self.ds.sizes["time"])

        self._validate_dimensions()

        self._init_indices()

    def _validate_dimensions(self):
        """Validações que dependem do dataset estar aberto"""
        if self.patch_size > self.H or self.patch_size > self.W:
            raise ValueError(
                f"patch_size ({self.patch_size}) maior que dimensões da imagem ({self.H}x{self.W})"
            )

    def _resolve_stride(self, explicit_stride: Optional[int]) -> int:
        if explicit_stride is not None:
            return int(explicit_stride)
        if self.group == "train":
            return int(self.output_timesteps)
        return int(self.sequence_length)

    def _init_indices(self):
        # Validação Temporal
        max_t0 = self.total_timesteps - self.sequence_length
        if max_t0 < 0:
            raise ValueError(f"Timesteps insuficientes: {self.total_timesteps}")
        
        self.valid_t0 = list(range(0, max_t0 + 1, self.stride))

        # Grid Espacial
        self.patch_origins = self._build_patch_grid(
            height=self.H, width=self.W, 
            patch_size=self.patch_size, patch_stride=self.patch_stride
        )

    @staticmethod
    def _build_patch_grid(height, width, patch_size, patch_stride):
        max_i = height - patch_size
        max_j = width - patch_size
        i_list = list(range(0, max(max_i + 1, 1), patch_stride))
        j_list = list(range(0, max(max_j + 1, 1), patch_stride))
        if i_list[-1] != max_i: i_list.append(max_i)
        if j_list[-1] != max_j: j_list.append(max_j)
        return [(i, j) for i in i_list for j in j_list]

    def __len__(self) -> int:
        if self.steps_per_epoch is not None:
            return int(self.steps_per_epoch)
        return len(self.valid_t0) * len(self.patch_origins)

    def __getitem__(self, idx: int):
        # Lógica de amostragem (mantida idêntica à original)
        if self.steps_per_epoch is not None:
            t0 = self.valid_t0[int(self.rng.integers(0, len(self.valid_t0)))]
            i0, j0 = self.patch_origins[int(self.rng.integers(0, len(self.patch_origins)))]
        else:
            n_patches = len(self.patch_origins)
            t0 = self.valid_t0[idx // n_patches]
            i0, j0 = self.patch_origins[idx % n_patches]

        t_in_end = t0 + self.input_timesteps
        t_out_end = t_in_end + self.output_timesteps
        i1, j1 = i0 + self.patch_size, j0 + self.patch_size

        x_np = self.input_da.isel(time=slice(t0, t_in_end), lat=slice(i0, i1), lon=slice(j0, j1)).to_numpy()
        y_np = self.target_da.isel(time=slice(t_in_end, t_out_end), lat=slice(i0, i1), lon=slice(j0, j1)).to_numpy()

        x = torch.from_numpy(x_np[:, None, :, :]).to(dtype=self.dtype)
        y = torch.from_numpy(y_np[:, None, :, :]).to(dtype=self.dtype)
        return x, y