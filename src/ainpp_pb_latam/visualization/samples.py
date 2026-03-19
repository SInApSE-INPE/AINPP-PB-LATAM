import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def save_epoch_sample(model, loader, epoch, device, save_dir="samples"):
    """
    Generates and saves a comparative plot (Obs vs Pred) for the first validation sample.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # 1. Get a single validation batch
    inputs, targets = next(iter(loader))

    # 2. Prepare data (same logic as training)
    inputs = inputs.squeeze(2).to(device)  # [Batch, T_in, H, W]
    targets = targets.squeeze(2).to(device)  # [Batch, T_out, H, W]
    print(f"Max target value (mm/h): {targets.max().item():.2f}, Shape: {targets.shape}")

    with torch.no_grad():
        # Log input (as model expects)
        # inputs_log = torch.log1p(inputs)
        # Log Prediction
        # add dimension for channel
        inputs = inputs.unsqueeze(2)  # [Batch, T_in, 1, H, W]
        outputs_log = model(inputs)

        # 3. Revert to mm/h for visualization (expm1)
        preds_mm = outputs_log.cpu().numpy()
        targets_mm = targets.cpu().numpy()  # Original Targets are already in mm/h

    # 4. Configurar o Plot
    batch_idx = 0  # Pega a primeira amostra do batch
    timesteps = targets_mm.shape[1]  # Quantos tempos futuros (M_OUT)

    # Cria figura: 2 linhas (Obs, Pred) x N colunas (Tempos)
    fig, axes = plt.subplots(2, timesteps, figsize=(4 * timesteps, 6))

    # Treatment if there is only 1 timestep (for axes to behave like a matrix)
    if timesteps == 1:
        axes = axes.reshape(2, 1)

    # Define common color scale (based on maximum observation for fair comparison)
    # vmax = targets_mm[batch_idx].max() + 1.0

    for t in range(timesteps):
        # Top Row: Observation (Ground Truth)
        ax_obs = axes[0, t]
        ax_obs.imshow(targets_mm[batch_idx, t], cmap="jet", origin="upper")
        ax_obs.set_title(f"Obs T+{t+1}")
        ax_obs.axis("off")

        # Bottom Row: Prediction
        ax_pred = axes[1, t]
        ax_pred.imshow(preds_mm[batch_idx, t, 0], cmap="jet", origin="upper")
        ax_pred.set_title(f"Pred T+{t+1}")
        ax_pred.axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch_{epoch+1}_sample.png")
    plt.close(fig)  # Close to free RAM memory
