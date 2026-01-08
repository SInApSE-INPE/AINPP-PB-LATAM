import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from typing import Tuple, Literal, Union, Optional
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F 
import torch
import numpy as np 
import xarray as xr
from enum import Enum
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))


from src.models.afno.forecaster import AFNO2D
SplitName = Literal["train", "validation", "test"]
InputSource = Literal["mvk", "nrt", "both"]
VarName = Literal["mvk", "nrt"]
ReturnLayout = Literal["CTHW", "TCHW"]


# Função auxiliar para Loss Espectral (Evita blur)
def spectral_loss(pred, target, weight=0.5):
    # 1. Cast para float32 (Resolve o erro de dimensão não ser potência de 2)
    pred = pred.float()
    target = target.float()
    
    # 2. Agora o FFT roda em float32 (seguro para qualquer tamanho)
    pred_fft = torch.fft.rfft2(pred, dim=(-2, -1), norm='ortho')
    target_fft = torch.fft.rfft2(target, dim=(-2, -1), norm='ortho')
    
    loss = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
    return weight * loss

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, rank=0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.rank = rank
        
        # Otimizador
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
        
        # Scheduler com Warmup (Essencial para Transformers/AFNO)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=5e-4, 
            steps_per_epoch=len(train_loader), 
            epochs=50 # Ajuste conforme necessário
        )
        
        # Mixed Precision Scaler
        self.scaler = GradScaler()
        
        # Loss base
        self.mse_crit = nn.MSELoss()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
            
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward com Mixed Precision
            with autocast():
                pred = self.model(x)
                
                # Loss Composta: MSE (pixel) + Spectral (nitidez)
                l_mse = self.mse_crit(pred, y)
                l_spec = spectral_loss(pred, y, weight=0.1)
                loss = l_mse + l_spec
            
            # Backward otimizado
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            
            # Gradient Clipping (Vital para estabilidade da AFNO)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            if self.rank == 0 and batch_idx % 20 == 0:
                print(f"[Epoch {epoch}][Batch {batch_idx}] Loss: {loss.item():.5f} (MSE: {l_mse:.5f}, Spec: {l_spec:.5f})")
                
        return total_loss / len(self.train_loader)

    def save_checkpoint(self, path):
        if self.rank == 0:
            # Salva o 'module' se for DDP
            state_dict = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
            torch.save(state_dict, path)


class GSMaPZarrSequenceDataset(Dataset):
    """
    Dataset PyTorch para carregar sequências do Zarr salvo no formato:
      group: train/validation/test
      variables: gsmap_mvk, gsmap_nrt
      dims: (time, lat, lon)

    Retorna:
      x:  (Tin, H, W)  se input_source in {"mvk","nrt"} e return_layout="TCHW" com C implícito (Tin)
          (C, Tin, H, W) se input_source="both" e return_layout="CTHW" (C=2)
      y:  (Tout, H, W) sempre (target é 1 canal), mantendo o formato (Tout, H, W)
    """

    def __init__(
        self,
        zarr_path: Union[str, Path],
        split: SplitName = "train",
        input_len: int = 12,
        target_len: int = 6,
        stride: int = 18, # passo entre amostras (em timesteps)
        input_source: InputSource = "nrt",   # para AFNO aqui: usar "mvk" (1 canal)
        target_var: VarName = "mvk",
        return_layout: ReturnLayout = "TCHW",  # para AFNO aqui: queremos [B, T, H, W]
        consolidated: bool = True,
        load_as_numpy: bool = False,
    ):
        super().__init__()
        self.zarr_path = str(Path(zarr_path))
        self.split = split
        self.input_len = int(input_len)
        self.target_len = int(target_len)
        self.stride = int(stride)
        self.input_source = input_source
        self.target_var = target_var
        self.return_layout = return_layout
        self.consolidated = consolidated
        self.load_as_numpy = load_as_numpy

        if self.input_len <= 0 or self.target_len <= 0:
            raise ValueError("input_len e target_len devem ser > 0.")
        if self.stride <= 0:
            raise ValueError("stride deve ser > 0.")

        # AFNO atual espera x: [B, T_in, H, W] (um canal), e y: [B, T_out, H, W]
        # então aqui vamos usar input_source="mvk" ou "nrt" (1 canal).
        if self.input_source == "both":
            raise ValueError(
                "Este script AFNO está configurado para 1 canal (T como canais). "
                "Use input_source='mvk' ou 'nrt'."
            )

        self._open()
        self._validate_vars()
        self._build_index()

    def _open(self) -> None:
        self.ds = xr.open_zarr(self.zarr_path, group=self.split, consolidated=self.consolidated)
        for dim in ("time", "lat", "lon"):
            if dim not in self.ds.dims:
                raise ValueError(f"Dimensão '{dim}' não encontrada no split '{self.split}'.")

        self.time_len = int(self.ds.sizes["time"])
        self.h = int(self.ds.sizes["lat"])
        self.w = int(self.ds.sizes["lon"])

    def _validate_vars(self) -> None:
        needed = [f"gsmap_{self.input_source}", f"gsmap_{self.target_var}"]
        missing = [v for v in needed if v not in self.ds.data_vars]
        if missing:
            raise ValueError(
                f"Variáveis ausentes no split '{self.split}': {missing}. "
                f"Disponíveis: {list(self.ds.data_vars)}"
            )

    def _build_index(self) -> None:
        total = self.input_len + self.target_len
        max_start = self.time_len - total
        if max_start < 0:
            raise ValueError(
                f"Split '{self.split}' tem {self.time_len} timesteps, precisa de {total}."
            )
        self.starts = np.arange(0, max_start + 1, self.stride, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.starts.size)

    def __getitem__(self, idx: int):
        start = int(self.starts[idx])

        x_t0 = start
        x_t1 = start + self.input_len
        y_t0 = x_t1
        y_t1 = x_t1 + self.target_len

        # x_raw: (Tin, H, W)
        x_raw = self.ds[f"gsmap_{self.input_source}"].isel(time=slice(x_t0, x_t1)).values
        # y_raw: (Tout, H, W)
        y_raw = self.ds[f"gsmap_{self.target_var}"].isel(time=slice(y_t0, y_t1)).values

        x_raw = x_raw.astype(np.float32, copy=False)
        y_raw = y_raw.astype(np.float32, copy=False)

        # transform to ensure there are no NaNs
        x_raw = np.nan_to_num(x_raw, nan=0.0)
        y_raw = np.nan_to_num(y_raw, nan=0.0)

        # print(f"Debug __getitem__: x_raw.shape={x_raw.shape}, y_raw.shape={y_raw.shape}, {np.where(np.isnan(x_raw))}, {np.where(np.isnan(y_raw))}")
        # Para este AFNO: queremos x = [T_in, H, W], y = [T_out, H, W]
        # (T atua como "canais" no patch_embed Conv2d, conforme o seu forward)
        if self.load_as_numpy:
            return x_raw, y_raw

        return torch.from_numpy(x_raw), torch.from_numpy(y_raw)

def main():
    # 1. Setup Distribuído (SLURM friendly)
    if 'RANK' in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        # Modo Debug/Single GPU
        rank = 0
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        world_size = 1

    # 2. Dataset (Usando sua classe existente GSMaPZarr)
    # Certifique-se que o dataset retorna [T, H, W] ou [C, H, W]
    # No seu caso: Input (12 frames) -> Output (6 frames)
    # Se o dataset retorna (input, target), deve ser:
    # input: [12, 880, 970], target: [6, 880, 970]
    
    # Placeholder para sua chamada de dataset
    train_dataset = GSMaPZarrSequenceDataset(
        split="train",
        zarr_path="/prj/ideeps/adriano.almeida/data/ainpp/legacy/AINPP-PB-LATAM.zarr", 
        input_len=12, 
        target_len=6, 
        stride=18
    )
    val_dataset = GSMaPZarrSequenceDataset(
        split="validation", 
        zarr_path="/prj/ideeps/adriano.almeida/data/ainpp/legacy/AINPP-PB-LATAM.zarr", 
        input_len=12, 
        target_len=6, 
        stride=18
    )

    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, # Cuidado com memória! AFNO consome VRAM em grids grandes
        sampler=train_sampler,
        num_workers=8, 
        pin_memory=True
    )
    
    # 3. Instanciar Modelo
    model = AFNO2D(
        img_size=(880, 970),
        in_chans=12,   # Ajuste conforme seu dataset (ex: 6 ou 12 tempos passados)
        out_chans=6,  # 6 tempos futuros
        embed_dim=256,
        depth=6,      # Comece com menos camadas se faltar memória
        num_blocks=8  # Deve dividir 256 (256/8 = 32)
    ).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # 4. Treinar
    trainer = Trainer(model, train_loader, None, device, rank)
    
    for epoch in range(50):
        avg_loss = trainer.train_epoch(epoch)
        if rank == 0:
            print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")
            trainer.save_checkpoint(f"checkpoints/afno_epoch_{epoch}.pth")

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()