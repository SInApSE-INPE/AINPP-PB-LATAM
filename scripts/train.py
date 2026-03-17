

import hydra
import os
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler
from hydra.utils import instantiate
from ainpp_pb_latam.datasets import NowcastingDataset
from ainpp_pb_latam.trainer import Trainer

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # DDP Setup
    if cfg.training.distributed:
        # torchrun sets these env vars
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            
            dist.init_process_group(
                backend=cfg.training.dist_backend,
                init_method="env://",
                world_size=world_size,
                rank=rank
            )
            print(f"Initialized DDP: Rank {rank}/{world_size}, Local Rank {local_rank}")
        else:
            print("Warning: Distributed mode enabled but RANK/WORLD_SIZE not found. Falling back to single device.")
            cfg.training.distributed = False
            rank = 0
            local_rank = 0
            world_size = 1
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    if rank == 0:
        print(f"Training with config:\n{cfg}")
    
    # 1. Load Datasets
    train_dataset = NowcastingDataset(cfg, split='train')
    val_dataset = NowcastingDataset(cfg, split='val')
    
    # Samplers
    if cfg.training.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle = False # Sampler handles shuffling
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.dataset.batch_size, 
        shuffle=shuffle, 
        sampler=train_sampler,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.dataset.batch_size, 
        shuffle=False, 
        sampler=val_sampler,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory
    )
    
    # 2. Instantiate Model
    model = instantiate(cfg.model)
    
    # 3. Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader, cfg, rank=rank, local_rank=local_rank)
    
    # 4. Start Training
    trainer.fit(train_sampler=train_sampler)
    
    # Cleanup DDP
    if cfg.training.distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
