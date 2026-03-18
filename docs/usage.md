# Usage

## CLI Entry Point

The unified CLI lives at `main.py` and dispatches execution from the Hydra configuration tree in `conf/`.

## Available Tasks

- `task=train`
- `task=evaluate`
- `task=infer`

The active defaults are composed in `conf/config.yaml`.

### Train

```bash
python main.py task=train \
  model=unet/direct \
  training=default \
  dataset=gsmap \
  training.epochs=5 \
  dataset.dataset.zarr_path=/path/to/dataset.zarr
```

### Evaluate

```bash
python main.py task=evaluate \
  model=unet/direct \
  checkpoint=outputs/checkpoint.pth
```

### Infer

```bash
python main.py task=infer \
  inference.mode=historical \
  model=unet/direct \
  checkpoint=outputs/checkpoint.pth
```

## Hydra Composition

- Override any config key with `key=value`.
- Select configuration groups such as `model=unet/direct` or `training=gan`.
- Keep structural parameters in Hydra YAML files instead of adding ad hoc CLI parsers.

## Common Configuration Areas

- `conf/dataset/`: dataset object and dataloader settings
- `conf/model/`: model families and forecasting modes
- `conf/training/`: optimizer, scheduler, epochs, checkpoints
- `conf/evaluation/`: thresholds and benchmark behavior
- `conf/inference/`: inference mode and outputs
- `conf/visualization/`: figure generation options

## Execution Patterns

### Direct Forecasting

Use direct models when the network should predict all future horizons in one pass:

```bash
python main.py task=train model=unet/direct
```

### Autoregressive Forecasting

Use autoregressive configs when each horizon depends on the previous predicted step:

```bash
python main.py task=train model=unet/autoregressive
```

### GAN Training

Use the GAN training profile when discriminator and adversarial loss components are required:

```bash
python main.py task=train model=unet/direct training=gan discriminator=patchgan
```

## HPC Notes

- Keep the package installed with `uv pip install -e .`.
- Control data loading and worker behavior through Hydra system and dataset settings.
- Prefer configuration-driven scaling for single-GPU, multi-GPU, and multi-node execution.

## Tips

- Use `HYDRA_FULL_ERROR=1` for detailed tracebacks.
- Set `CUDA_VISIBLE_DEVICES` to control GPU selection.
- Inspect generated experiment folders under `outputs/`.
