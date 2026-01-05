# scripts/train.py
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch

# Adiciona src ao path (como discutido antes)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

# Importa a engine que acabamos de criar
from src.engine import run_training

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Configuração de Ambiente
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executando em: {device}")

    # 2. Instanciação (Hydra faz o trabalho pesado)
    # Note como o script não sabe se é GSMaP ou Radar, nem se é UNet ou ConvLSTM
    train_ds = instantiate(cfg.dataset.dataset, **cfg.dataset.overrides.train)
    val_ds = instantiate(cfg.dataset.dataset, **cfg.dataset.overrides.validation)
    
    train_loader = torch.utils.data.DataLoader(train_ds, **cfg.dataset.train_loader)
    val_loader = torch.utils.data.DataLoader(val_ds, **cfg.dataset.val_loader)
    
    model = instantiate(cfg.model).to(device)
    
    # 3. Otimizador (Pode ser configurado via Hydra também)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    criterion = instantiate(cfg.loss)

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

if __name__ == "__main__":
    main()