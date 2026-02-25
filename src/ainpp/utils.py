import logging
from pathlib import Path
from typing import Optional, Union, Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Attributes:
        patience (int): How long to wait after last time validation loss improved.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        mode (str): One of 'min' or 'max'.
        path (Path): Path for the checkpoint to be saved to.
        enabled (bool): Whether early stopping is enabled.
    """

    def __init__(
        self,
        patience: int = 5,
        delta: float = 0,
        mode: str = 'min',
        path: Union[str, Path] = 'checkpoint.pt',
        enabled: bool = True
    ) -> None:
        """
        Initializes the EarlyStopping object.

        Args:
            patience (int): Number of epochs to wait. Defaults to 5.
            delta (float): Minimum change to qualify as improvement. Defaults to 0.
            mode (str): 'min' or 'max'. Defaults to 'min'.
            path (Union[str, Path]): Checkpoint save path. Defaults to 'checkpoint.pt'.
            enabled (bool): Enable or disable early stopping. Defaults to True.
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.path = Path(path)
        self.enabled = enabled
        
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.val_loss_min = np.inf if mode == 'min' else -np.inf

        if not enabled:
            logger.info("Early Stopping disabled.")

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        """
        Checks if training should stop.

        Args:
            val_loss (float): Current validation loss.
            model (nn.Module): The model to save if loss improved.
        """
        if not self.enabled:
            return

        score = -val_loss if self.mode == 'min' else val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module) -> None:
        """
        Saves model when validation loss decreases.

        Args:
            val_loss (float): Current validation loss.
            model (nn.Module): The model to save.
        """
        if self.mode == 'min':
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def save_epoch_checkpoint(model: nn.Module, epoch: int, save_dir: Union[str, Path], **kwargs: Any) -> None:
    """
    Saves a periodic checkpoint of the model.

    Args:
        model (nn.Module): The model to save.
        epoch (int): The current epoch number.
        save_dir (Union[str, Path]): Directory to save the checkpoint.
        **kwargs (Any): Additional options (e.g., prefix).
    """
    prefix = kwargs.get("prefix", "checkpoint")
    path = Path(save_dir) / f"{prefix}_model_epoch_{epoch:03d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Periodic checkpoint saved: {path}")
