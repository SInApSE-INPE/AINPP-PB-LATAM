import logging
from pathlib import Path

import torch
import pathlib
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.dataset import DatasetConfig, AINPPPBLATAMDataset, create_gsmap_dataloaders

def _configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

def _print_basic_dataset_info(ds) -> None:
    # Atributos úteis da classe
    print("==== Dataset Info ====")
    print(f"Split/group:        {ds.group}")
    print(f"Zarr path:          {ds.zarr_path}")
    print(f"Domain (H x W):     {ds.H} x {ds.W}")
    print(f"Total timesteps:    {ds.total_timesteps}")
    print(f"Tin / Tout:         {ds.cfg.input_timesteps} / {ds.cfg.output_timesteps}")
    print(f"Sequence length:    {ds.sequence_length}")
    print(f"Temporal stride:    {ds.stride_t}")
    print(f"Patch (h x w):      {ds.patch_h} x {ds.patch_w}")
    print(f"Patch stride (h,w): {ds.stride_h} x {ds.stride_w}")
    print(f"#valid_t0:          {len(ds.valid_t0)}")
    print(f"#patch_origins:     {len(ds.patch_origins)}")
    print(f"steps_per_epoch:    {ds.cfg.steps_per_epoch}")
    print(f"len(dataset):       {len(ds)}")
    print("======================\n")

def test_single_samples(ds, n: int = 3) -> None:
    print("==== Sample Checks ====")
    for k in range(n):
        out = ds[k]
        if len(out) == 2:
            x, y = out
            meta = None
        else:
            x, y, meta = out

        assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
        assert x.ndim == 4 and y.ndim == 4, f"Expected (T, C, H, W). Got x={x.shape}, y={y.shape}"

        tin, c1, ph, pw = x.shape
        tout, c2, ph2, pw2 = y.shape

        assert tin == ds.cfg.input_timesteps
        assert tout == ds.cfg.output_timesteps
        assert c1 == 1 and c2 == 1
        assert ph == ds.patch_h and pw == ds.patch_w
        assert ph2 == ds.patch_h and pw2 == ds.patch_w

        # Sanity: finito e sem NaN
        assert torch.isfinite(x).all(), "x contains non-finite values"
        assert torch.isfinite(y).all(), "y contains non-finite values"

        print(f"[ok] idx={k} x={tuple(x.shape)} y={tuple(y.shape)} meta={meta}")
    print("=======================\n")

def test_dataloader(dl, n_batches: int = 2) -> None:
    print("==== DataLoader Checks ====")
    it = iter(dl)
    for b in range(n_batches):
        batch = next(it)
        if len(batch) == 2:
            xb, yb = batch
        else:
            xb, yb, meta = batch  # se você habilitar return_metadata no dataset (não é o padrão)
        assert xb.ndim == 5 and yb.ndim == 5, f"Expected (B, T, C, H, W). Got xb={xb.shape}, yb={yb.shape}"
        assert torch.isfinite(xb).all(), "xb contains non-finite values"
        assert torch.isfinite(yb).all(), "yb contains non-finite values"
        print(f"[ok] batch={b} xb={tuple(xb.shape)} yb={tuple(yb.shape)} dtype={xb.dtype}")
    print("===========================\n")

if __name__ == "__main__":
    _configure_logging(logging.INFO)

    # Ajuste este caminho para o seu Zarr
    zarr_path = Path("/prj/ideeps/adriano.almeida/data/ainpp/legacy/AINPP-PB-LATAM.zarr")
    assert zarr_path.exists(), f"Zarr não encontrado: {zarr_path}"

    # ----------------------------
    # 1) Teste com patch retangular
    # ----------------------------
    cfg_rect = DatasetConfig(
        zarr_path=zarr_path,
        group="train",
        input_timesteps=12,
        output_timesteps=6,
        stride=1,                 # explícito
        patch_height=880,         # retangular
        patch_width=970,
        patch_stride_h=880,       # sem overlap vertical
        patch_stride_w=970,       # 50% overlap horizontal
        steps_per_epoch=20,       # modo aleatório (treino)
        seed=42,
        consolidated=True,
    )

    ds_rect = AINPPPBLATAMDataset(cfg_rect, return_metadata=True, dtype=torch.float32)
    _print_basic_dataset_info(ds_rect)
    test_single_samples(ds_rect, n=3)

    # DataLoader para esse dataset específico
    dl_rect = torch.utils.data.DataLoader(
        ds_rect,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    test_dataloader(dl_rect, n_batches=2)

    # ----------------------------
    # 2) Teste full-frame (tamanho real)
    # ----------------------------
    cfg_full = DatasetConfig(
        zarr_path=zarr_path,
        group="validation",
        input_timesteps=12,
        output_timesteps=6,
        stride=6,
        patch_height=None,        # full domain em lat
        patch_width=None,         # full domain em lon
        patch_stride_h=None,      # irrelevante, vira 1 patch
        patch_stride_w=None,      # irrelevante, vira 1 patch
        steps_per_epoch=None,     # determinístico
        seed=42,
        consolidated=True,
    )

    ds_full = AINPPPBLATAMDataset(cfg_full, return_metadata=True, dtype=torch.float32)
    _print_basic_dataset_info(ds_full)
    test_single_samples(ds_full, n=1)

    dl_full = torch.utils.data.DataLoader(
        ds_full,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_dataloader(dl_full, n_batches=1)

    # ----------------------------
    # 3) Teste da factory (train/val/test)
    # ----------------------------
    loaders = create_gsmap_dataloaders(
        zarr_path=zarr_path,
        input_timesteps=12,
        output_timesteps=6,
        batch_size=2,
        num_workers=0,
        train_stride=1,
        eval_stride=6,
        patch_height=160,
        patch_width=320,
        patch_stride_h=160,
        patch_stride_w=160,
        steps_per_epoch=30,  # só treino
        seed=42,
    )

    for split in ("train", "validation", "test"):
        print(f"\n--- Factory loader split={split} ---")
        test_dataloader(loaders[split], n_batches=1)
