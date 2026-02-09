import torch
import logging
from pathlib import Path
import numpy as np


logger = logging.getLogger(__name__)


class EarlyStopping:
    """Para o treino se a loss de validação não melhorar após 'patience' épocas."""
    def __init__(self, patience=5, delta=0, mode='min', path='checkpoint.pt', enabled=True):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.path = Path(path)
        self.enabled = enabled
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf if mode == 'min' else -np.inf

        if not enabled:
            logger.info("Early Stopping desativado.")

    def __call__(self, val_loss, model):
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

    def save_checkpoint(self, val_loss, model):
        """Salva o modelo quando a loss decresce."""
        if self.mode == 'min':
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        # Cria diretório se não existir
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def save_epoch_checkpoint(model, epoch, save_dir, **kwargs):
    """Salva checkpoint periódico."""
    prefix = kwargs.get("prefix", "checkpoint")
    path = Path(save_dir) / f"{prefix}_model_epoch_{epoch:03d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Checkpoint periódico salvo: {path}")
