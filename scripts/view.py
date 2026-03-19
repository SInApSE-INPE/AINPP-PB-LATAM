import matplotlib.pyplot as plt
import numpy as np
import torch

from ainpp_pb_latam.datasets.gsmap import AINPPPBLATAMDataset

# Import the class of your specific model here. Example:
from ainpp_pb_latam.models.unet.forecaster import UNetMultiHorizon

# from ainpp_pb_latam.models.afno.net import AFNO2D
# from ainpp_pb_latam.models.gan.generator import Generator


def load_model(model_class, checkpoint_path, device, model_args={}):
    """
    Carrega a arquitetura e os pesos do modelo.
    """
    model = model_class(**model_args).to(device)

    # Carregar pesos
    # Note: If saved with Hydra/DDP, keys might have prefix 'module.' or 'model.'
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Common treatment to remove prefix 'module.' from distributed training
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove 'module.'
        new_state_dict[name] = v

    try:
        model.load_state_dict(new_state_dict)
        print("Pesos carregados com sucesso!")
    except RuntimeError as e:
        print(f"Erro ao carregar pesos (tente ajustar strict=False): {e}")
        model.load_state_dict(new_state_dict, strict=False)

    model.eval()
    return model


def visualize_prediction(model, dataset, idx=0, device="cuda", save_path="resultado.png"):
    """
    Plots Observation (Ground Truth) vs Prediction.
    """
    # 1. Get data
    x, y = dataset[idx]
    # x shape: (Tin, C, H, W) -> Needs to become (1, Tin, C, H, W) for the model
    x_tensor = x.unsqueeze(0).to(device)

    # 2. Inference
    with torch.no_grad():
        pred = model(x_tensor)

    # 3. Process for Plot (Remove Batch and pass to CPU)
    # y shape: (Tout, C, H, W)
    # pred shape: (1, Tout, C, H, W) -> (Tout, C, H, W)
    tgt_seq = y.squeeze(1).numpy()
    pred_seq = pred.squeeze(0).squeeze(1).cpu().numpy()

    # Ensure non-negative values for visualization
    pred_seq = np.maximum(pred_seq, 0)

    num_timesteps = tgt_seq.shape[0]

    # 4. Configurar Plot
    fig, axes = plt.subplots(2, num_timesteps, figsize=(num_timesteps * 3, 6))

    # Define fixed color limits for fair comparison (vmin, vmax)
    vmin, vmax = 0, np.max(tgt_seq) * 1.0  # Adjust according to maximum rain

    for t in range(num_timesteps):
        # --- Linha 1: Observations (Target) ---
        ax_obs = axes[0, t]
        im_obs = ax_obs.imshow(tgt_seq[t], cmap="jet", vmin=vmin, vmax=vmax, origin="lower")
        ax_obs.set_title(f"Obs T+{t+1}")
        ax_obs.axis("off")

        # --- Linha 2: Predictions (Model) ---
        ax_pred = axes[1, t]
        im_pred = ax_pred.imshow(pred_seq[t], cmap="jet", vmin=vmin, vmax=vmax, origin="lower")
        ax_pred.set_title(f"Pred T+{t+1}")
        ax_pred.axis("off")

    # Adicionar Colorbar compartilhada
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im_obs, cax=cbar_ax, label="Precipitation (mm/h)")

    plt.suptitle(f"Sample #{idx} - Observation vs Prediction", fontsize=16)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Chart saved in: {save_path}")
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
        patch_height=440,  # Important to be equal to training
        patch_width=475,
        stride=1,
    )

    # 2. Configure Model (Example with UNetMultiHorizon)
    # Replace with the class you used (AFNO2D, GraphCastNet, etc)
    model_args = {
        "input_timesteps": 4,
        "input_channels": 1,
        "output_timesteps": 4,
        "output_channels": 1,
    }

    # Instanciar e carregar
    # Point to your checkpoint .pth
    CHECKPOINT = "/prj/ideeps/adriano.almeida/benchmark/scripts/outputs/2026-01-09/16-06-39/early_stopping/discriminator_model_epoch_005.pt"

    model = load_model(UNetMultiHorizon, CHECKPOINT, device, model_args)

    # 3. Gerar Plot
    # idx=10 gets the tenth sample of the validation dataset
    visualize_prediction(model, dataset, idx=10, device=device)
