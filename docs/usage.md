# Usage

## CLI Entry Point
The unified CLI lives at `main.py` and supports Hydra overrides for all tasks.

### Preprocess
```bash
python main.py task=preprocess preprocessing.region.name=amazon-basin
```

### Train
```bash
python main.py task=train training.epochs=5 dataset.dataset.zarr_path=/path/to/zarr
```

### Evaluate
```bash
python main.py task=evaluate checkpoint=outputs/checkpoint.pth
```

## Hydra Overrides
- Override any config key via `key=value`.
- Compose configs from `conf/` using the `defaults` lists already defined.

## Tips
- Use `HYDRA_FULL_ERROR=1` for detailed tracebacks.
- Set `CUDA_VISIBLE_DEVICES` to control GPU selection.
