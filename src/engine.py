import torch
import logging
from tqdm import tqdm
from src.utils import EarlyStopping, save_epoch_checkpoint

logger = logging.getLogger(__name__)

import sys
from pathlib import Path
import torch.distributed as dist

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
    # Setup Early Stopping (apenas referência, a lógica muda abaixo)
    best_model_path = f"{checkpoint.dir}/best_model.pt"
    early_stopper = EarlyStopping(
        patience=early_stopping.patience,
        delta=early_stopping.delta,
        path=best_model_path,
        enabled=early_stopping.enabled
    )

    print("Running for {} epochs".format(epochs))
    
    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        
        # Dica visual: Desabilitar barra de progresso nos workers para não duplicar log
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True, disable=not is_main_process())
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            if is_main_process():
                pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validação
        avg_val_loss = run_validation(model, val_loader, criterion, device)
        
        # --- LÓGICA DE PARADA SINCRONIZADA ---
        
        # 1. Criamos um tensor flag: 0 = Continuar, 1 = Parar
        stop_signal = torch.tensor(0, device=device)

        if is_main_process():
            logger.info(f"Epoch {epoch} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")

            # Checkpoint Periódico
            if checkpoint.enabled and (epoch % checkpoint.interval == 0):
                save_epoch_checkpoint(model, epoch, checkpoint.dir)

            # Early Stopping
            early_stopper(avg_val_loss, model)
            
            if early_stopper.early_stop:
                logger.info("Early stopping ativado no Mestre. Iniciando parada geral.")
                stop_signal = torch.tensor(1, device=device) # Muda flag para 1

        # 2. SINCRONIZAÇÃO (O Pulo do Gato)
        # Se estivermos em DDP, o Rank 0 avisa todo mundo qual é o valor de stop_signal
        if dist.is_initialized():
            dist.broadcast(stop_signal, src=0)

        # 3. Todos os processos verificam a flag Sincronizada
        if stop_signal.item() == 1:
            break  # Agora TODOS quebram o loop juntos

    if is_main_process():
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