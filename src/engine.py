import torch
import logging
from tqdm import tqdm
from src.utils import EarlyStopping, save_epoch_checkpoint

logger = logging.getLogger(__name__)

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.distributed import is_main_process

def run_training(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        device, 
        epochs,
        criterion,
        early_stopping,
        checkpoint,
        train_sampler=None,
):
    """
    Args:
        training_cfg: A parte 'training' do DictConfig do Hydra
    """
    
    # --- Configura Callbacks ---
    
    # 1. Setup Early Stopping (que também salva o 'best_model.pt')
    
    best_model_path = f"{checkpoint.dir}/best_model.pt"
    
    early_stopper = EarlyStopping(
        patience=early_stopping.patience,
        delta=early_stopping.delta,
        path=best_model_path,
        enabled=early_stopping.enabled
    )

    # Loop de Treino
    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validação
        avg_val_loss = run_validation(model, val_loader, criterion, device)
        
        if is_main_process():
            logger.info(f"Epoch {epoch} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")

            # 1. Checkpoint Periódico (Intervalo fixo)
            if checkpoint.enabled and (epoch % checkpoint.interval == 0):
                save_epoch_checkpoint(model, epoch, checkpoint.dir)

            # 2. Early Stopping & Best Model Save
            # O early_stopper verifica internamente se deve salvar o melhor modelo
            early_stopper(avg_val_loss, model)
        
            if early_stopper.early_stop:
                logger.info("Early stopping ativado. Parando treino.")
            break

    logger.info("Treino finalizado.")

def run_validation(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += criterion(pred, y).item()
    return total_loss / len(loader)