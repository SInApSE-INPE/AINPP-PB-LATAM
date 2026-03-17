"""
Main training script entry point.

This script handles the initialization of the distributed environment (DDP),
loading of configuration via Hydra, instantiation of datasets and models,
and execution of the training engine.
"""

import os
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from hydra.utils import instantiate
import hydra
from omegaconf import DictConfig


from ainpp_pb_latam.distributed import setup_distributed, cleanup_distributed, is_main_process
from ainpp_pb_latam.engine import run_training


def _configure_threading() -> None:
    """
    Configures environment variables for OMP/MKL threading.
    
    This is crucial for Slurm environments to respect CPU allocation limits (cgroups)
    and avoid oversubscription, which degrades performance.
    """
    if "OMP_NUM_THREADS" not in os.environ:
        try:
            # Attempt to get only CPUs allocated by Slurm/cgroups
            num_cores = len(os.sched_getaffinity(0))
        except AttributeError:
            # Fallback for Windows/Mac (sees all cores)
            num_cores = multiprocessing.cpu_count()
        
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # Ensure at least 1 thread per process
        threads_per_proc = max(1, (num_cores // world_size))
        
        os.environ["OMP_NUM_THREADS"] = str(threads_per_proc)
        os.environ["MKL_NUM_THREADS"] = str(threads_per_proc)
        
        if os.environ.get("RANK", "0") == "0":
            print(f"[Auto-Tuning] Available CPUs: {num_cores} | Threads per process: {threads_per_proc}")

# Apply threading config before imports that initialize OpenMP might happen
_configure_threading()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point managed by Hydra.

    Args:
        cfg (DictConfig): The configuration object parsed from YAML files and CLI.
    """
    use_ddp, local_rank, device = setup_distributed()

    try:
        if is_main_process():
            print(f"Starting training on device: {device}")

        # Instantiate Datasets
        # We unpack (** overrides) specific configurations for train/val splits
        train_ds = instantiate(cfg.dataset.dataset, **cfg.dataset.overrides.train)
        val_ds = instantiate(cfg.dataset.dataset, **cfg.dataset.overrides.validation)

        # Configure Samplers (Required for DDP)
        train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
        val_sampler = DistributedSampler(val_ds, shuffle=False) if use_ddp else None

        # Instantiate DataLoaders
        train_loader = DataLoader(
            train_ds,
            sampler=train_sampler,
            shuffle=(train_sampler is None), # Shuffle only if not using a sampler
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

        # Instantiate Model
        model: nn.Module = instantiate(cfg.model).to(device)
        
        if use_ddp:
            if cfg.system.get("sync_bn", False):
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            
            model = nn.parallel.DistributedDataParallel(
                model, 
                device_ids=[local_rank],
                output_device=local_rank
            )

        optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
        criterion: nn.Module = instantiate(cfg.loss).to(device)

        print("Initializing Training Engine...")
        
        run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            epochs=cfg.training.epochs,
            criterion=criterion,
            early_stopping=cfg.training.early_stopping,
            checkpoint=cfg.training.checkpoint,
            train_sampler=train_sampler 
        )

    except Exception as e:
        if is_main_process():
            print(f"Fatal error during training: {e}")
        raise e
        
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()