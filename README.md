# Nowcasting Benchmark for Latin America

## Objective
The goal of this project is to establish a robust and modular benchmark for **precipitation nowcasting** (up to 6 hours) using deep learning models in a **supercomputing environment**.

The study area covers **Latin America** (Lat: -55 to 33, Lon: -120 to -23). The system is designed to handle large-scale **Zarr** datasets containing satellite precipitation estimates (`gsmap_nrt` as input, `gsmap_mvk` as target).

## Key Features
- **Modular Design**: Separated modules for Dataset, Models, Training, and Evaluation.
- **Configurable**: Powered by **Hydra** for flexible configuration of hyperparameters, models, and data paths.
- **Experiment Tracking**: Integrated with **MLFlow** for logging metrics, parameters, and artifacts.
- **Distributed Training**: Native support for **Multi-GPU/Multi-Node** training using `torchrun` and `DistributedDataParallel`.
- **Model Support**: Extensible Model Factory supporting U-Net and easily adaptable for ResNet, Inception, etc.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Configuration
The main configuration is located in `conf/config.yaml`. You can override parameters via command line.

### Single Device Training
```bash
python scripts/train.py
```

### Distributed Training (Multi-GPU)
To run on a single node with multiple GPUs (e.g., 4 GPUs):
```bash
torchrun --nproc_per_node=4 scripts/train.py training.distributed=true
```

### Evaluation
```bash
python scripts/evaluate.py
```

## Project Structure
- `conf/`: Hydra configuration files.
- `src/`: Source code.
  - `dataset.py`: Zarr data loading and time slicing.
  - `models/`: Model architectures and factory.
  - `trainer.py`: Training loop with DDP and MLFlow support.
- `scripts/`: Entry points for training and evaluation.
- `tests/`: Verification scripts.

## Data
The dataset is expected to be in **Zarr** format with the following structure:
- **Variables**: `gsmap_nrt`, `gsmap_mvk`
- **Dimensions**: Time, Latitude, Longitude
- **Splits**:
  - Train: 2018-2022
  - Validation: 2023
  - Test: 2024
