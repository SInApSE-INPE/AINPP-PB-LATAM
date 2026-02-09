import torch
import matplotlib.pyplot as plt
import numpy as np
import pathlib

# Garantir que o diretório raiz do projeto está no sys.path
import sys
FILE = pathlib.Path(__file__).resolve()
ROOT = FILE.parents[1]  # Diretório raiz do projeto
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from src.datasets.gsmap import AINPPPBLATAMDataset
# Importe a classe do seu modelo específico aqui. Exemplo:
from src.models.unet.forecaster import UNetMultiHorizon 
# from src.models.afno.net import AFNO2D 
# from src.models.gan.generator import Generator

def load_model(model_class, checkpoint_path, device, model_args={}):
    """
    Carrega a arquitetura e os pesos do modelo.
    """
    model = model_class(**model_args).to(device)
    
    # Carregar pesos
    # Nota: Se salvou com Hydra/DDP, as chaves podem ter prefixo 'module.' ou 'model.'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Tratamento comum para remover prefixo 'module.' de treinos distribuídos
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") # remove 'module.'
        new_state_dict[name] = v
        
    try:
        model.load_state_dict(new_state_dict)
        print("Pesos carregados com sucesso!")
    except RuntimeError as e:
        print(f"Erro ao carregar pesos (tente ajustar strict=False): {e}")
        model.load_state_dict(new_state_dict, strict=False)

    model.eval()
    return model

def visualize_prediction(model, dataset, idx=0, device='cuda', save_path='resultado.png'):
    """
    Plota Observação (Ground Truth) vs Previsão.
    """
    # 1. Pegar dados
    x, y = dataset[idx] 
    # x shape: (Tin, C, H, W) -> Precisa virar (1, Tin, C, H, W) para o modelo
    x_tensor = x.unsqueeze(0).to(device)
    
    # 2. Inferência
    with torch.no_grad():
        pred = model(x_tensor)
    
    # 3. Processar para Plot (Remover Batch e passar para CPU)
    # y shape: (Tout, C, H, W)
    # pred shape: (1, Tout, C, H, W) -> (Tout, C, H, W)
    tgt_seq = y.squeeze(1).numpy() 
    pred_seq = pred.squeeze(0).squeeze(1).cpu().numpy()
    
    # Garantir valores não negativos para visualização
    pred_seq = np.maximum(pred_seq, 0)

    num_timesteps = tgt_seq.shape[0]
    
    # 4. Configurar Plot
    fig, axes = plt.subplots(2, num_timesteps, figsize=(num_timesteps * 3, 6))
    
    # Definir limites de cor fixos para comparação justa (vmin, vmax)
    vmin, vmax = 0, np.max(tgt_seq) * 1.0 # Ajuste conforme a chuva máxima
    
    for t in range(num_timesteps):
        # --- Linha 1: Observações (Target) ---
        ax_obs = axes[0, t]
        im_obs = ax_obs.imshow(tgt_seq[t], cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
        ax_obs.set_title(f"Obs T+{t+1}")
        ax_obs.axis('off')

        # --- Linha 2: Previsões (Model) ---
        ax_pred = axes[1, t]
        im_pred = ax_pred.imshow(pred_seq[t], cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
        ax_pred.set_title(f"Pred T+{t+1}")
        ax_pred.axis('off')

    # Adicionar Colorbar compartilhada
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    fig.colorbar(im_obs, cax=cbar_ax, label='Precipitation (mm/h)')
    
    plt.suptitle(f"Amostra #{idx} - Observação vs Previsão", fontsize=16)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Gráfico salvo em: {save_path}")
    plt.show()

# --- BLOCO PRINCIPAL ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Configurar Dataset (Use os mesmos parametros do treino)
    dataset = AINPPPBLATAMDataset(
        zarr_path="/prj/ideeps/adriano.almeida/data/ainpp/legacy/AINPP-PB-LATAM.zarr",
        group="test",
        input_timesteps=4,
        output_timesteps=4,
        patch_height=440,  # Importante ser igual ao treino
        patch_width=475,
        stride=1
    )

    # 2. Configurar Modelo (Exemplo com UNetMultiHorizon)
    # Substitua pela classe que você usou (AFNO2D, GraphCastNet, etc)
    model_args = {
        "input_timesteps": 4,
        "input_channels": 1,
        "output_timesteps": 4,
        "output_channels": 1
    }
    
    # Instanciar e carregar
    # Aponte para o seu checkpoint .pth
    CHECKPOINT = "/prj/ideeps/adriano.almeida/benchmark/scripts/outputs/2026-01-09/16-06-39/early_stopping/discriminator_model_epoch_005.pt" 
    
    model = load_model(UNetMultiHorizon, CHECKPOINT, device, model_args)

    # 3. Gerar Plot
    # idx=10 pega a décima amostra do dataset de validação
    visualize_prediction(model, dataset, idx=10, device=device)