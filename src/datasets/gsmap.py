import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Tipos auxiliares
PatchOrigin = Tuple[int, int]
TensorPair = Tuple[torch.Tensor, torch.Tensor]

class AINPPPBLATAMDataset(Dataset[TensorPair]):
    """
    Dataset GSMaP com suporte a patches retangulares e Hydra.
    """
    def __init__(
        self,
        zarr_path: str,
        group: str = "train",
        input_timesteps: int = 12,
        output_timesteps: int = 6,
        stride: Optional[int] = None,
        # Novos parâmetros retangulares
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
        return_metadata: bool = False
    ) -> None:
        super().__init__()
        
        # 1. Configuração Básica
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
        
        # Conversão de tipos
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.rng = np.random.default_rng(seed)

        # Configuração de Patches (Salva para resolver depois de abrir o Zarr)
        self._cfg_patch_h = patch_height
        self._cfg_patch_w = patch_width
        self._cfg_stride_h = patch_stride_h
        self._cfg_stride_w = patch_stride_w

        # Resolução Temporal
        self.sequence_length = self.input_timesteps + self.output_timesteps
        self.stride_t = self._resolve_stride(stride)

        # 2. Abrir Dataset
        logger.info(f"Opening Zarr: {self.zarr_path} (Group: {self.group})")
        self.ds = xr.open_zarr(
            self.zarr_path,
            group=self.group,
            consolidated=self.consolidated,
        )

        # Validação de Variáveis e NaN handling
        if self.input_var not in self.ds.variables:
            raise KeyError(f"Input '{self.input_var}' not found.")
        
        # Lazy loading + fillna
        self.input_da = self.ds[self.input_var].fillna(0.0)
        self.target_da = self.ds[self.target_var].fillna(0.0)

        # 3. Dimensões do Domínio
        self.H = int(self.ds.sizes["lat"])
        self.W = int(self.ds.sizes["lon"])
        self.total_timesteps = int(self.ds.sizes["time"])

        # 4. Resolver Patches Retangulares (Agora que temos H e W)
        self.patch_h, self.patch_w = self._resolve_patch_shape(self.H, self.W)
        self.stride_h, self.stride_w = self._resolve_patch_stride(self.patch_h, self.patch_w)

        # 5. Inicializar Índices
        self._init_indices()

        logger.info(
            f"Dataset Ready: {len(self)} samples | "
            f"Patch: {self.patch_h}x{self.patch_w} | "
            f"Grid Stride: {self.stride_h}x{self.stride_w}"
        )

    def _resolve_stride(self, explicit_stride: Optional[int]) -> int:
        if explicit_stride is not None:
            return int(explicit_stride)
        if self.group == "train":
            return int(self.output_timesteps)
        return int(self.sequence_length)

    def _resolve_patch_shape(self, H: int, W: int) -> Tuple[int, int]:
        ph = self._cfg_patch_h if self._cfg_patch_h is not None else H
        pw = self._cfg_patch_w if self._cfg_patch_w is not None else W

        if ph <= 0 or pw <= 0:
            raise ValueError("Patch dimensions must be > 0.")
        if ph > H or pw > W:
            raise ValueError(f"Patch ({ph}x{pw}) exceeds domain ({H}x{W}).")
        return int(ph), int(pw)

    def _resolve_patch_stride(self, ph: int, pw: int) -> Tuple[int, int]:
        sh = self._cfg_stride_h if self._cfg_stride_h is not None else ph
        sw = self._cfg_stride_w if self._cfg_stride_w is not None else pw
        return int(sh), int(sw)

    def _init_indices(self):
        # Temporal
        max_t0 = self.total_timesteps - self.sequence_length
        if max_t0 < 0:
            raise ValueError(f"Timesteps insuficientes: {self.total_timesteps}")
        self.valid_t0 = list(range(0, max_t0 + 1, self.stride_t))

        # Espacial (Grid Determinístico)
        self.patch_origins = self._build_patch_grid(
            height=self.H, width=self.W,
            patch_h=self.patch_h, patch_w=self.patch_w,
            stride_h=self.stride_h, stride_w=self.stride_w
        )

    @staticmethod
    def _build_patch_grid(height, width, patch_h, patch_w, stride_h, stride_w):
        max_i = height - patch_h
        max_j = width - patch_w

        i_list = list(range(0, max(max_i + 1, 1), stride_h))
        j_list = list(range(0, max(max_j + 1, 1), stride_w))

        # Garante cobertura das bordas
        if i_list[-1] != max_i: i_list.append(max_i)
        if j_list[-1] != max_j: j_list.append(max_j)

        return [(i, j) for i in i_list for j in j_list]

    def __len__(self) -> int:
        if self.steps_per_epoch is not None:
            return int(self.steps_per_epoch)
        return len(self.valid_t0) * len(self.patch_origins)

    def __getitem__(self, idx: int):
        # Lógica de Amostragem
        if self.steps_per_epoch is not None:
            # Random Sampling
            t0 = self.valid_t0[int(self.rng.integers(0, len(self.valid_t0)))]
            i0, j0 = self.patch_origins[int(self.rng.integers(0, len(self.patch_origins)))]
        else:
            # Deterministic Iteration
            n_patches = len(self.patch_origins)
            t0 = self.valid_t0[idx // n_patches]
            i0, j0 = self.patch_origins[idx % n_patches]

        # Slicing
        ph, pw = self.patch_h, self.patch_w
        t_in_end = t0 + self.input_timesteps
        t_out_end = t_in_end + self.output_timesteps
        
        i1 = i0 + ph
        j1 = j0 + pw

        x_np = self.input_da.isel(time=slice(t0, t_in_end), lat=slice(i0, i1), lon=slice(j0, j1)).to_numpy()
        y_np = self.target_da.isel(time=slice(t_in_end, t_out_end), lat=slice(i0, i1), lon=slice(j0, j1)).to_numpy()

        x = torch.from_numpy(x_np[:, None, :, :]).to(dtype=self.dtype)
        y = torch.from_numpy(y_np[:, None, :, :]).to(dtype=self.dtype)

        if self.return_metadata:
            return x, y, {"t0": t0, "i0": i0, "j0": j0, "ph": ph, "pw": pw}
            
        return x, y