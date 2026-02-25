# AINPP Precipitation Benchmark

![Build](https://img.shields.io/badge/CI-local--check-lightgrey)
![Docs](https://img.shields.io/badge/docs-MkDocs%2FMaterial-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

Unified Hydra-driven CLI and documentation for precipitation nowcasting in Latin America.

## Table of Contents
- [Quick Start](#quick-start)
- [Unified CLI (Hydra)](#unified-cli-hydra)
- [Tests & Quality Gates](#tests--quality-gates)
- [Documentation](#documentation)
- [Model Zoo (select via Hydra model)](#model-zoo-select-via-hydra-model)
- [Training Methods & Configuration](#training-methods--configuration)
- [Project Structure (key parts)](#project-structure-key-parts)

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,docs]
```

### Unified CLI (Hydra)
Run all stages via `main.py`:
- Preprocess: `python main.py task=preprocess preprocessing.region.name=amazon-basin`
- Train: `python main.py task=train training.epochs=5 dataset.dataset.zarr_path=/path/to/zarr`
- Evaluate: `python main.py task=evaluate checkpoint=outputs/checkpoint.pth`

Override any config with Hydra syntax: `key=value` (configs live in `conf/`).

### Tests & Quality Gates
```bash
./scripts/check_all.sh    # pytest + coverage + lint/typecheck
```

### Documentation
- Local preview: `mkdocs serve`
- Build: `mkdocs build`
- (If published) GitHub Pages link: _add repo Pages URL here_.

## Model Zoo (select via Hydra `model=...`)

| Model | Type | Highlight |
| :--- | :--- | :--- |
| AFNO | Transformer | Spectral mixing for global context |
| U-Net | CNN | Encoder-decoder with skips |
| ConvLSTM | RNN | Explicit temporal dynamics |
| ResNet50 | CNN | Deep residual features |
| InceptionV4 | CNN | Multi-scale inception blocks |
| Xception | CNN | Depthwise separable, lightweight |

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

Configure losses via Hydra (`loss=...`, see `conf/loss/`).

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
python main.py task=train
```

### Evaluation
```bash
python main.py task=evaluate checkpoint=outputs/my_run/best_model.pt
```

#### Evaluation Metrics
-   **Continuous**: MSE, RMSE, MAE, R², Pearson Correlation.
-   **Categorical** (per threshold): POD, FAR, CSI (TS), ETS.
-   **Probabilistic**: CRPS, Reliability Diagrams.

## Project Structure (key parts)
- `main.py` — unified Hydra CLI for preprocess/train/evaluate
- `conf/` — configuration hierarchy
- `src/ainpp/` — library code (datasets, models, preprocessing, evaluation, visualization)
- `docs/` — MkDocs sources (API via mkdocstrings)
- `scripts/` — supporting utilities (legacy entry points now superseded by `main.py`)

## Data
The dataset is expected to be in **Zarr** format with the following structure:
- **Variables**: `gsmap_nrt`, `gsmap_mvk`
- **Dimensions**: Time, Latitude, Longitude
- **Splits**:
  - Train: 2018-2022
  - Validation: 2023
  - Test: 2024
