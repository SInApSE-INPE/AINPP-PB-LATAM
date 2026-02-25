import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def save_epoch_sample(model, loader, epoch, device, save_dir="samples"):
    """
    Gera e salva um plot comparativo (Obs vs Pred) para a primeira amostra da validação.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # 1. Pegar um único batch da validação
    inputs, targets = next(iter(loader))
    
    # 2. Preparar dados (mesma lógica do treino)
    inputs = inputs.squeeze(2).to(device) # [Batch, T_in, H, W]
    targets = targets.squeeze(2).to(device) # [Batch, T_out, H, W]
    print(f"Max target value (mm/h): {targets.max().item():.2f}, Shape: {targets.shape}")
    
    with torch.no_grad():
        # Entrada em Log (como o modelo espera)
        # inputs_log = torch.log1p(inputs)
        # Previsão em Log
        # add dimension for channel
        inputs = inputs.unsqueeze(2)  # [Batch, T_in, 1, H, W]
        outputs_log = model(inputs)
        
        # 3. Reverter para mm/h para visualização (expm1)
        preds_mm = outputs_log.cpu().numpy()
        targets_mm = targets.cpu().numpy() # Targets originais já estão em mm/h

    # 4. Configurar o Plot
    batch_idx = 0  # Pega a primeira amostra do batch
    timesteps = targets_mm.shape[1] # Quantos tempos futuros (M_OUT)
    
    # Cria figura: 2 linhas (Obs, Pred) x N colunas (Tempos)
    fig, axes = plt.subplots(2, timesteps, figsize=(4 * timesteps, 6))
    
    # Tratamento caso seja apenas 1 timestep (para axes funcionar como matriz)
    if timesteps == 1: 
        axes = axes.reshape(2, 1)

    # Definir escala de cor comum (baseada no máximo da observação para comparação justa)
    # vmax = targets_mm[batch_idx].max() + 1.0 

    for t in range(timesteps):
        # Linha Superior: Observação (Ground Truth)
        ax_obs = axes[0, t]
        ax_obs.imshow(targets_mm[batch_idx, t], cmap='jet', origin='upper')
        ax_obs.set_title(f"Obs T+{t+1}")
        ax_obs.axis('off')

        # Linha Inferior: Previsão
        ax_pred = axes[1, t]
        ax_pred.imshow(preds_mm[batch_idx, t, 0], cmap='jet', origin='upper')
        ax_pred.set_title(f"Pred T+{t+1}")
        ax_pred.axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch_{epoch+1}_sample.png")
    plt.close(fig) # Fecha para liberar memória RAM
