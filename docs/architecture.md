# Architecture

## Project Scope

AINPP-PB-LATAM is a scientific benchmark library for precipitation nowcasting in Latin America. The project is designed around reproducible experiments, Hydra-driven configuration, and execution on HPC environments ranging from a single GPU to multi-node distributed runs.

## Data Contract

The benchmark assumes a fixed spatiotemporal structure:

- Base storage format: `.zarr`
- Spatial grid: `880 x 970`
- Temporal resolution: hourly
- Training split: `2018-2022`
- Validation split: `2023`
- Test split: `2024`
- Default input source: `gsmap_nrt`
- Default target source: `gsmap_mvk`
- Default sequence layout: `12` input steps and `6` forecast steps

These defaults are exposed in Hydra configs and can be overridden per experiment without changing code.

## Training Modes

The codebase is structured to support two forecasting strategies:

- Autoregressive: the model predicts one horizon at a time and feeds predictions back into the next step.
- Direct: the model predicts the full multi-step horizon in a single forward pass.

Current model configs live under `conf/model/` and include AFNO, ConvLSTM, InceptionV4, ResNet50, UNet, and Xception. GAN training is configured separately through `conf/training/gan.yaml` and `conf/discriminator/patchgan.yaml`.

## Configuration Model

Hydra is the source of truth for:

- model architecture,
- dataset and dataloader definitions,
- training hyperparameters,
- inference mode,
- evaluation thresholds and lead times,
- visualization outputs,
- system-level options such as workers and memory pinning.

The root composition is defined in `conf/config.yaml`.

## Runtime Pipeline

The CLI entry point is `main.py` and dispatches to three main tasks:

- `train`
- `evaluate`
- `infer`

Each task instantiates the dataset and model from Hydra configs and keeps experiment outputs under timestamped `outputs/` directories.

## Scientific Evaluation Layers

The benchmark follows a layered evaluation design:

- `metrics/`: pure mathematical metric implementations
- `evaluation/`: task orchestration across thresholds, horizons, and batches
- `aggregation/`: statistical consolidation into report-friendly tables
- `visualization/`: plots and benchmark figures from aggregated outputs

This separation is important because visualization should consume evaluated and aggregated results instead of recomputing the scientific core.

## Distributed Execution

The project is intended for HPC usage and should remain compatible with:

- single GPU,
- multi-GPU single-node,
- multi-GPU multi-node.

Distributed helpers are kept in the package and should be configured through Hydra rather than ad hoc environment-specific code paths.
