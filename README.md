# AINPP Precipitation Benchmark

Unified scientific benchmark library for precipitation nowcasting in Latin America using deep learning on high-performance computing (HPC) environments.

## Key Features

- **Extensive Model Zoo**: AFNO, ConvLSTM, GAN, Graph NN, InceptionV4, ResNet50, UNet, and Xception.
- **Scalable HPC Training**: Built-in support for Single GPU, Multi-GPU (Distributed Data Parallel), and Multi-Node clusters.
- **Standardized Data Formats**: Optimized data loading and processing utilizing Zarr archives with daily/hourly grid matrices.
- **Config-Driven Architecture**: Fully modular, parameterized via Hydra to completely decouple code from experiments.
- **Metrics & Evaluation**: Three-pronged evaluation module covering Spatial, Continuous, and Probabilistic metrics.

---

## 🛠 Tech Stack

- **Language**: Python 3.10+
- **Deep Learning**: PyTorch, Torchvision, TIMM
- **Experiment Tracking**: MLflow
- **Configuration**: Hydra, OmegaConf
- **Data Processing**: Zarr, Xarray, Dask, Pandas, Numpy
- **Metrics & Science**: Scikit-Learn, SciPy
- **Package Manager**: `uv`

---

## Prerequisites

- Python 3.10 or higher.
- NVIDIA GPUs available (CUDA environment) for practical training and evaluation.
- `uv` installed on the system (a blazing-fast Python package installer and resolver).

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/user/ainpp-pb-latam.git
cd ainpp-pb-latam
```

### 2. Set Up the Environment

The project relies on `uv` to maintain its isolated environment. Create a virtual environment:

```bash
uv venv
```

Activate the environment:

```bash
# On Linux/macOS
source .venv/bin/activate
```

### 3. Install Dependencies

**CRITICAL**: The package must always be installed in editable mode (`-e`) using `uv`. Do not use `sys.path` hacks in your experiments.

```bash
# Install the core package along with dev and docs dependencies
uv pip install -e .[dev,docs]
```

### 4. Verify Installation

You can check if the CLI is ready and Hydra configuration is locatable:

```bash
python main.py --help
```

---

## Architecture Overview

### Data Constraints & Design
The benchmark operates under strict spatio-temporal properties tailored for the Brazilian/Latin-American precipitation models:
- **Base Format**: `.zarr` file stores.
- **Data Splits**: 
  - `train`: 2018 to 2022
  - `validation`: 2023
  - `test`: 2024
- **Structural Properties**: 
  - Matrices of `880 x 970` spatial grids.
  - Hourly granularity.
- **Temporal Configuration**:
  - Input: 12 consecutive hours originating from `gsmap_nrt`.
  - Target (Prediction): 6 consecutive hours pointing to `gsmap_mvk`.
- **Training Strategies**: Models can be trained using either an **Autoregressive** (predict 1 step, feed-back, repeat) or **Direct** (predict all 6 steps at once) approach.

### Directory Structure

```
├── conf/                     # Hydra configuration YAMLs
│   ├── config.yaml           # Root configuration
│   ├── dataset/              # Dataloader & data path configs
│   ├── discriminator/        # GAN discriminators (e.g. patchgan)
│   ├── evaluation/           # Evaluation metric definitions
│   ├── loss/                 # Loss functions (e.g., mse, ssim)
│   ├── model/                # Architecture configurations (unet, afno, etc.)
│   ├── preprocessing/        # Data prep configurations
│   ├── training/             # Optimizer, lr scheduler, epochs
│   └── visualization/        # Plotting parameters
├── docs/                     # MkDocs documentation
├── scripts/                  # Utilities (legacy running blocks, bash scripts)
│   ├── check_all.sh          # Full quality-gate workflow
│   └── enforce_coverage.py   # Coverage tools
├── src/
│   └── ainpp/                # Core Python package
│       ├── datasets/         # Zarr loading and sampling logic
│       ├── evaluation/       # Benchmark metric calculators
│       ├── layers/           # Reusable neural network layers
│       ├── models/           # Model Zoo definitions (UNet, ResNet, etc.)
│       ├── preprocessing/    # Data preparation pipelines
│       ├── visualization/    # Handlers for model output plotting
│       ├── distributed.py    # DDP sync rules for Multi-Node
│       ├── engine.py         # Standard training loops
│       ├── engine_gan.py     # specialized loops for GAN-based setups
│       ├── losses.py         # Specialized precipitation loss implementations
│       └── utils.py          # Object builders (Loss, Optimizer)
├── tests/                    # Pytest suite
├── main.py                   # Unified CLI Entry point
├── pyproject.toml            # Build system definitions
└── uv.lock                   # Deterministic python dependency tree
```

### Request Lifecycle

1. You run `python main.py task=<TASK_TYPE>`.
2. **Hydra** merges `conf/config.yaml` with the sub-dictionaries provided (loss, models, training parameters) and command-line overrides.
3. Depending on the task (`preprocess`, `train`, `evaluate`):
   - Initializes the Zarr Datasets via `ainpp.datasets`.
   - Compiles the Model defined in `conf/model/` and ships it to GPU (or configures `DistributedDataParallel`).
   - Hooks into `ainpp.engine` or `ainpp.evaluation` and streams data until completion.

---

## Configuration via Hydra

This project delegates all configurations (hyperparameters, variables, dataset paths, training parameters) to **Hydra**. We do **not** use `argparse` or `.env` files for architecture controls.

### Modifying Parameters
You can override any parameter on the command line using the simple `.yaml` trajectory:

```bash
# Change the learning rate and batch size for a training run:
python main.py task=train training.lr=0.0005 dataset.train_loader.batch_size=32

# Change the model to an AFNO and loss to a Hybrid scheme:
python main.py task=train model=afno loss=hybrid_mse_ssim
```

### Understanding Loss Functions
We provide specialized loss functions designed for high-intensity precipitation tasks:
- **Pixel-wise**: `WeightedMSE` (penalizes heavy rain errors), `LogCosh`, `HuberLoss`.
- **Structural**: `SSIMLoss` (anti-blurring), `SpectralLoss`, `PerceptualLoss` (Feature MSE).
- **Hybrid**: `HybridLoss` (configurable weighted summation).

---

## Available Commands

Run any stage via `main.py`.

| Command | Description |
|---|---|
| `python main.py task=preprocess` | Run the standard dataset preprocessing algorithms. You can override settings like `preprocessing.region.name`. |
| `python main.py task=train` | Kickoff training. By default outputs runs to `./outputs/<date>/<time>`. |
| `python main.py task=evaluate checkpoint=/path/to/my_model.pt` | Run spatial, continuous, and probabilistic metric validation on the held-out test data. |
| `./scripts/check_all.sh` | Run all Linters, typecheck, and coverage reports at once. |
| `mkdocs serve` | Host documentation locally mimicking github pages structure. |

---

## Testing

Quality assurance is mandated globally via the Makefile/Shell scripts. We rely on `pytest`, `coverage`, `black`, `isort`, and `mypy`.

### Running Tests

```bash
# Run all automated tests (Minitest equivalent in Python)
pytest tests/

# Run tests with coverage map pointing at src/ainpp
pytest --cov=src/ainpp tests/

# Shortcut for linting, typing and testing standardly
./scripts/check_all.sh
```

---

## Training and Deployment in HPC Workspace

The framework utilizes `torch.distributed` and is meant to be run transparently on massive multi-GPU or multi-Node bounds. 

### Single Node, Multi GPU Deployment
Run the process under `torchrun` and declare how many GPUs to map per node:

```bash
# Running DDP with 4 GPUs
torchrun --nproc_per_node=4 main.py task=train dataset.train_loader.batch_size=16
```
*(Variables are aggregated per-batch across GPUs, keeping the effective batch-size as GPUs * node_batch_size).*

### Compute / Checkpointing Rules
- **Early Stopping**: Models monitor validation loss. If improvement ceases before parameter `patience`, training terminates.
- **Checkpointing**: Every epoch emits a checkpoint, defaulting the minimum validation loss state to `best_model.pt`.

---

## Troubleshooting

### ImportErrors on ainpp.*
**Error**: `ModuleNotFoundError: No module named 'ainpp'`
**Solution**: Ensure you actually installed the package into the current uv environment via the editable command.
```bash
uv pip install -e .
```

### CUDA out of memory
**Error**: `torch.cuda.OutOfMemoryError / CUDA out of memory.`
**Solution**: Drop the batch size or hidden dimensions through Hydra.
```bash
python main.py task=train dataset.train_loader.batch_size=4 model.hidden_channels=[16,16,16]
```

### Deadlocks in DistributedDataParallel
**Error**: The system freezes on epoch conclusion or validation phases.
**Solution**: DDP often hangs if the dataset isn't perfectly sliced, or if evaluation metrics attempt to reduce uneven tensors. Try lowering the number of workers per dataloader using `system.num_workers=0` for debug traces.

## License

This architecture operates under an MIT open-source license.
