# scripts/train.py
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Adiciona src ao path (como discutido antes)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.distributed import setup_distributed, cleanup_distributed, is_main_process
from src.engine import run_training

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Setup do Ambiente (DDP ou Single)
    use_ddp, local_rank, device = setup_distributed()

    # Log apenas no processo mestre para não poluir o terminal
    if is_main_process():
        print(f"Iniciando treino no device: {device}")

    # 2. Instanciar Dataset (Igual antes)
    train_ds = instantiate(cfg.dataset.dataset, **cfg.dataset.overrides.train)
    val_ds = instantiate(cfg.dataset.dataset, **cfg.dataset.overrides.validation)

    # 3. Configurar Sampler para DDP
    # Se usar DDP, o sampler é obrigatório. Se não, é None.
    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if use_ddp else None

    # 4. DataLoaders
    # ATENÇÃO: Se sampler não é None, shuffle no DataLoader DEVE ser False
    train_loader = DataLoader(
        train_ds,
        sampler=train_sampler,
        shuffle=(train_sampler is None), # Shuffle só se NÃO tiver sampler
        batch_size=cfg.dataset.train_loader.batch_size,
        num_workers=cfg.system.num_workers,
        pin_memory=cfg.system.pin_memory
    )
    
    val_loader = DataLoader(
        val_ds,
        sampler=val_sampler,
        shuffle=False,
        batch_size=cfg.dataset.val_loader.batch_size,
        num_workers=cfg.system.num_workers,
        pin_memory=cfg.system.pin_memory
    )

    # 5. Modelo e Distribuição
    model = instantiate(cfg.model).to(device)
    
    if use_ddp:
        # Converte Batch Norms comuns para SyncBatchNorm (opcional, recomendado para batch pequeno)
        if cfg.system.get("sync_bn", False):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            
        # Envelopa o modelo com DDP
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank],
            output_device=local_rank
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    
    # Instancia Loss
    criterion = instantiate(cfg.loss).to(device)

    # 4. EXECUÇÃO -> Chama o módulo
    print("Iniciando Engine de Treino...")
    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=cfg.training.epochs,
        criterion=criterion,
        early_stopping=cfg.training.early_stopping,
        checkpoint=cfg.training.checkpoint
    )

    # 5. Cleanup DDP
    cleanup_distributed()

if __name__ == "__main__":
    main()