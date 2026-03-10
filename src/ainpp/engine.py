import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import DictConfig
from tqdm import tqdm

from ainpp.utils import EarlyStopping, save_epoch_checkpoint
from ainpp.distributed import is_main_process
from ainpp.visualization.samples import save_epoch_sample

logger = logging.getLogger(__name__)

def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    criterion: nn.Module,
    early_stopping: DictConfig,
    checkpoint: DictConfig,
    train_sampler: Optional[DistributedSampler] = None,
) -> None:
    """
    Executes the main training loop with support for DDP, Early Stopping, and Checkpointing.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader containing the training dataset.
        val_loader (DataLoader): DataLoader containing the validation dataset.
        optimizer (optim.Optimizer): The optimizer (e.g., Adam, SGD).
        device (torch.device): The execution device (CPU or CUDA).
        epochs (int): Total number of epochs to train.
        criterion (nn.Module): The loss function.
        early_stopping (DictConfig): Hydra configuration containing 'patience', 'delta', 'enabled'.
        checkpoint (DictConfig): Hydra configuration containing 'interval', 'dir', 'enabled'.
        train_sampler (Optional[DistributedSampler]): Distributed sampler used during training.
            Essential for ensuring correct data shuffling in multi-GPU environments. Defaults to None.
    """
    
    # Setup Early Stopping
    # We use early_stopping (the config object) to instantiate the logic class
    best_model_path = Path(checkpoint.dir) / "best_model.pt"
    
    early_stopper = EarlyStopping(
        patience=early_stopping.patience,
        delta=early_stopping.delta,
        path=best_model_path,
        enabled=early_stopping.enabled
    )

    if is_main_process():
        print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        # DDP Requirement: set_epoch ensures the shuffle seed changes every epoch.
        # Without this, the data order would be identical across all epochs.
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        
        # Progress Bar:
        # leave=True: Keeps the bar history in the terminal.
        # disable=not is_main_process(): Prevents 4 GPUs from printing 4 bars simultaneously.
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch}/{epochs}", 
            leave=True, 
            disable=not is_main_process()
        )
        
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if is_main_process():
                # Display instantaneous loss on the bar
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Phase
        avg_val_loss = run_validation(model, val_loader, criterion, device)
        
        # ============================================================
        # SYNCHRONIZED STOPPING LOGIC (DDP Handling)
        # ============================================================
        
        # 1. Flag: 0 = Continue, 1 = Stop
        stop_signal = torch.tensor(0, device=device)

        # Only Rank 0 decides whether to stop or continue
        if is_main_process():
            logger.info(f"Epoch {epoch} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")
            logger.info("Plotting and saving sample prediction...")
            save_epoch_sample(model, val_loader, epoch, device, save_dir="samples")

            # Periodic Checkpoint
            if checkpoint.enabled and (epoch % checkpoint.interval == 0):
                save_epoch_checkpoint(model, epoch, checkpoint.dir)

            # Early Stopping Check
            early_stopper(avg_val_loss, model)
            
            if early_stopper.early_stop:
                logger.info("Early stopping triggered on Master. Initiating global stop.")
                stop_signal = torch.tensor(1, device=device)

        # 2. SYNCHRONIZATION (Critical)
        # If running in DDP, Rank 0 broadcasts its decision to all other ranks.
        # Without broadcast, Rank 0 exits the loop while Rank 1 waits forever (Deadlock).
        if dist.is_initialized():
            dist.broadcast(stop_signal, src=0)

        # 3. All processes execute the stop together
        if stop_signal.item() == 1:
            break

    if is_main_process():
        logger.info("Training finished successfully.")


def run_validation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Performs a full validation pass (without gradient calculation).

    Args:
        model (nn.Module): The model to be evaluated.
        loader (DataLoader): Validation DataLoader.
        criterion (nn.Module): Loss function to calculate metrics.
        device (torch.device): Execution device.

    Returns:
        float: The average loss over the entire validation dataset.
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = model(x)
            total_loss += criterion(pred, y).item()
            
    return total_loss / len(loader)