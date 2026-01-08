# Nowcasting Benchmark for Latin America

## Objective
The goal of this project is to establish a robust and modular benchmark for **precipitation nowcasting** (up to 6 hours) using deep learning models in a **supercomputing environment**.

The study area covers **Latin America** (Lat: -55 to 33, Lon: -120 to -23). The system is designed to handle large-scale **Zarr** datasets containing satellite precipitation estimates (`gsmap_nrt` as input, `gsmap_mvk` as target).

## Key Features
- **Modular Design**: Separated modules for Dataset, Models, Training, and Evaluation.
- **Configurable**: Powered by **Hydra** for flexible configuration of hyperparameters, models, and data paths.
- **Experiment Tracking**: Integrated with **MLFlow** for logging metrics, parameters, and artifacts.
- **Distributed Training**: Native support for **Multi-GPU/Multi-Node** training using `torchrun` and `DistributedDataParallel`.
- **Model Support**: Extensible Model Factory supporting state-of-the-art architectures like AFNO, U-Net, and more.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Model Zoo

This benchmark includes a variety of architectures tailored for spatiotemporal forecasting.

| Model | Type | Description | Key Reference |
| :--- | :--- | :--- | :--- |
| **AFNO** | Transformer | Adaptive Fourier Neural Operator. Uses spectral mixing for efficient global context modeling. | [Guibas et al., 2021](https://arxiv.org/abs/2111.13587) |
| **U-Net** | CNN | Classic encoder-decoder with skip connections. High precision for local features. | [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597) |
| **ConvLSTM** | RNN | Convolutional LSTM. Captures temporal dynamics explicitly through recurrent states. | [Shi et al., 2015](https://arxiv.org/abs/1506.04214) |
| **ResNet50** | CNN | Deep residual network adapted for dense prediction. Good at feature extraction. | [He et al., 2016](https://arxiv.org/abs/1512.03385) |
| **InceptionV4**| CNN | Multi-scale inception modules. Captures features at different spatial scales simultaneously. | [Szegedy et al., 2017](https://arxiv.org/abs/1602.07261) |
| **Xception** | CNN | Depthwise separable convolutions. Efficient and parameter-light. | [Chollet, 2017](https://arxiv.org/abs/1610.02357) |

### Configuring Models
Select a model using the Hydra `model` parameter:
```bash
python scripts/train.py model=afno
python scripts/train.py model=unet/direct
```

## Loss Functions

We provide specialized loss functions to handle the challenges of precipitation nowcasting (e.g., sparsity, high-intensity events, blurring).

### 1. Pixel-wise Losses
- **`WeightedMSE`**: Standard MSE with a dynamic weight mask.
  - *Formula*: $L = (y - \hat{y})^2 \times (1 + \alpha \cdot y)$
  - *Use case*: Penalizing errors in heavy rain regions more than light rain/zeros.
- **`LogCosh`**: Smooth approximation of MAE.
  - *Use case*: Robust to outliers, essentially a differentiable Huber loss.
- **`HuberLoss`**: Combination of MSE (near 0) and MAE (far from 0).
  - *Use case*: Handling noisy data.

### 2. Structural & Perceptual Losses
- **`SSIMLoss`**: Structural Similarity Index (inverted).
  - *Use case*: Preserving structural consistency and reducing the "blurring" effect common in MSE models.
- **`SpectralLoss`**: Frequency domain loss (Amplitude + Phase) using FFT.
  - *Use case*: Ensuring the model captures global textures and high-frequency details.
- **`PerceptualLoss`**: Feature-level MSE using a pre-trained VGG16 network.
  - *Use case*: Forcing the model to generate realistically looking rain patterns.

### 3. Classification & Hybrid Losses
- **`DiceLoss`**: Overlap metric for binary rain masks.
  - *Use case*: Improving the spatial extent prediction of storm cells.
- **`BinaryFocalLoss`**: Focuses on hard-to-classify examples (heavy rain vs. no rain).
- **`HybridLoss`**: Weighted combination of multiple losses.
  - *Example*: `1.0 * MSE + 0.1 * SSIM`

Configuration via Hydra (`conf/loss`):
```bash
python scripts/train.py loss=hybrid_mse_ssim
```

## Training Methods & Configuration

The training engine is built on `PyTorch` and `Hydra`, optimized for HPC environments.

### Main Configuration (`conf/config.yaml`)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `input_timesteps` | 12 | Number of past frames (e.g., 1 hour at 5-min intervals). |
| `output_timesteps`| 6 | Number of future frames to predict (e.g., 30 mins). |
| `training.batch_size` | 16 | Batch size per GPU. |
| `training.epochs` | 100 | Maximum epochs. |
| `training.lr` | 0.001 | Learning rate (Adam optimizer). |

### Distributed Data Parallel (DDP)
The framework uses `torch.distributed` for synchronous multi-GPU training.
- **Rank 0 (Master)**: Handles logging (tqdm, MLFlow), checkpointing, and validation aggregation.
- **Synchronization**: gradients are averaged across all GPUs. Loss calculation is local.

To run on 4 GPUs:
```bash
torchrun --nproc_per_node=4 scripts/train.py
```

### Early Stopping & Checkpointing
- **Early Stopping**: Monitors validation loss. If it doesn't improve for `patience` (default: 10) epochs, training stops.
  - *Logic*: All GPUs synchronize the stop signal to avoid deadlocks.
- **Checkpointing**: Saves model state, optimizer, and epoch.
  - *Best Model*: Always saves the model with the lowest validation loss to `best_model.pt`.
  - *Periodic*: Saves every `N` epochs.

## Usage

### Single Device Training
```bash
python scripts/train.py
```

### Evaluation
To evaluate a trained model, use the `scripts/evaluate.py` script. This module calculates continuous, categorical, and probabilistic metrics and generates performance diagrams.

```bash
python scripts/evaluate.py checkpoint=outputs/my_run/best_model.pt
```

#### Evaluation Metrics
-   **Continuous**: MSE, RMSE, MAE, R², Pearson Correlation.
-   **Categorical** (per threshold): POD, FAR, CSI (TS), ETS.
-   **Probabilistic**: CRPS, Reliability Diagrams.

## Project Structure
- `conf/`: Hydra configuration files.
  - `model/`: Configurations for AFNO, UNet, etc.
  - `loss/`: Configurations for Hybrid, MSE, etc.
- `src/`: Source code.
  - `dataset.py`: Zarr data loading and time slicing.
  - `models/`: Model architectures and factory.
  - `losses.py`: Implementation of all loss functions.
  - `engine.py`: Training loop with DDP and MLFlow support.
- `scripts/`: Entry points for training and evaluation.

## Data
The dataset is expected to be in **Zarr** format with the following structure:
- **Variables**: `gsmap_nrt`, `gsmap_mvk`
- **Dimensions**: Time, Latitude, Longitude
- **Splits**:
  - Train: 2018-2022
  - Validation: 2023
  - Test: 2024
