# Workflow Examples

See the [complete parameter reference](parameters.md) for defaults, overrides,
and implementation status.

These examples use the unified `main.py` CLI and the WPR benchmark Zarr on the
CPTEC filesystem. Run commands from the repository root in the same Bash session.
For model configuration details, see [Training Models](training.md).

## 1. Prepare the environment and data

```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[dl,verification,docs]'
python main.py --help
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

export BENCHMARK_ZARR=/prj/cptec/nowcasting/data/benchmark/benchmark_dataset_wpr.zarr
export BENCHMARK_RUN=outputs/unet_wpr
```

Replace `BENCHMARK_ZARR` on other machines. The store must be readable and contain
`train`, `validation`, and `test` groups, with `gsmap_nrt` and `gsmap_mvk` arrays
using `(time, lat, lon)` dimensions. The default loader expects consolidated Zarr
metadata. For a store without it, add `dataset.dataset.consolidated=false`.

The examples use 12 hourly input frames, 6 hourly target frames, and 320 × 320
patches. Other variables in the WPR store are not automatically used. Run training
on a compute node with an allocated GPU; the CLI falls back to CPU if CUDA is
unavailable.

## 2. Run a small training check

```bash
python main.py task=train model=unet/direct training=default \
  dataset.dataset.zarr_path="$BENCHMARK_ZARR" \
  dataset.train_loader.batch_size=1 \
  dataset.val_loader.batch_size=1 \
  training.epochs=1 \
  dataset.overrides.train.steps_per_epoch=8 \
  dataset.overrides.validation.steps_per_epoch=4 \
  training.checkpoint.dir="${BENCHMARK_RUN}_smoke/checkpoints" \
  hydra.run.dir="${BENCHMARK_RUN}_smoke"
```

This checks data loading, training, validation, and checkpoint writing with 8
training samples and 4 validation samples. It does not produce a scientifically
validated model. Despite its name, `steps_per_epoch` controls the number of
**samples**, not batches.

## 3. Train the supervised baseline

```bash
python main.py task=train model=unet/direct training=default \
  loss=hybrid_mse_ssim \
  dataset.dataset.zarr_path="$BENCHMARK_ZARR" \
  dataset.train_loader.batch_size=2 \
  dataset.val_loader.batch_size=2 \
  training.epochs=50 \
  training.checkpoint.dir="$BENCHMARK_RUN/checkpoints" \
  hydra.run.dir="$BENCHMARK_RUN"
```

The default configuration samples 500 items from each of `train` and `validation`
per epoch. Early stopping may end the run before epoch 50. To traverse validation
deterministically, add `dataset.overrides.validation.steps_per_epoch=null`.
This can take substantially longer.

The best validation checkpoint is saved as:

```text
outputs/unet_wpr/checkpoints/best_model.pt
```

Periodic checkpoints are saved in the same directory. Epoch sample images are
written to `samples/` by the training engine. Choose a new `BENCHMARK_RUN` for each
experiment to keep outputs distinct.

### Change the model or variables

Use the training command above with `model=convlstm/direct` or
`model=unet/autoregressive`, and choose a separate output directory. For AFNO,
align its image size with the dataset patches, for example
`model=afno/direct '+model.img_size=[320,320]'`.

To select variable names that are not listed in the dataset YAML, add:

```text
+dataset.dataset.input_var=gsmap_nrt
+dataset.dataset.target_var=gsmap_mvk
```

Selecting `training=gan` alone does **not** activate adversarial training in
`main.py`: its training handler currently calls the supervised engine. Likewise,
this entry point does not initialize distributed training merely because it is
launched with `torchrun`. See [Training Models](training.md) for implementation
context.

## 4. Check the checkpoint before inference or evaluation

```bash
export BENCHMARK_CHECKPOINT="$BENCHMARK_RUN/checkpoints/best_model.pt"
test -f "$BENCHMARK_CHECKPOINT" && echo "Checkpoint found"
```

Continue only if the checkpoint exists. The current CLI logs a warning and uses
fresh model weights if the file is missing. If only the small training check was
run, select `${BENCHMARK_RUN}_smoke/checkpoints/best_model.pt` instead.

Use the same model architecture and temporal dimensions as training, including
any custom model parameters. `checkpoint` loads weights for inference/evaluation;
it does not implement training resumption with optimizer and epoch state.

All following commands explicitly select `test`: without
`+dataset.overrides.test.group=test`, the current dataset defaults to `train`.

## 5. Generate one forecast

```bash
python main.py task=infer model=unet/direct \
  checkpoint="$BENCHMARK_CHECKPOINT" \
  dataset.dataset.zarr_path="$BENCHMARK_ZARR" \
  +dataset.overrides.test.group=test \
  inference.mode=single \
  inference.single.output_format=nc \
  inference.output_dir="$BENCHMARK_RUN/inference"
```

This predicts six frames for the **first test patch**, not a live observation or
a user-selected timestamp. The current CLI supplies a fixed timestamp, so the
output path is:

```text
outputs/unet_wpr/inference/single/2026/03/16/pred_gsmap_20260316_1200.nc
```

That filename is not the observation date. The NetCDF contains `precipitation`
with `(time, lat, lon)` dimensions, but the CLI does not preserve real timestamps
or geographic coordinates. For a raw PyTorch tensor instead, change
`inference.single.output_format=pt`.

## 6. Generate historical forecasts

```bash
python main.py task=infer model=unet/direct \
  checkpoint="$BENCHMARK_CHECKPOINT" \
  dataset.dataset.zarr_path="$BENCHMARK_ZARR" \
  +dataset.overrides.test.group=test \
  dataset.val_loader.batch_size=1 \
  inference.mode=historical \
  inference.batch_size=1 \
  inference.output_dir="$BENCHMARK_RUN/inference"
```

This traverses the test samples and writes
`outputs/unet_wpr/inference/predictions.zarr`. The `predictions` array has shape
`(samples, 6, 1, 320, 320)`. Outputs are individual patches; they are not stitched
into full-domain maps and do not include sample timestamps or patch coordinates.
A run overwrites an existing store at the same output path.

`dataset.val_loader.batch_size` controls inference batches; `inference.batch_size`
controls the output Zarr chunk size. For a small I/O check, add
`+dataset.overrides.test.steps_per_epoch=4`; this selects random samples and is
not a complete historical traversal.

Inspect the array with Zarr directly:

```bash
python - <<'PY'
import os
import zarr
store = zarr.open_group(os.environ['BENCHMARK_RUN'] + '/inference/predictions.zarr', mode='r')
print(store['predictions'].shape)
PY
```

## 7. Evaluate the trained model

Start with a small sampled evaluation:

```bash
python main.py task=evaluate model=unet/direct \
  checkpoint="$BENCHMARK_CHECKPOINT" \
  dataset.dataset.zarr_path="$BENCHMARK_ZARR" \
  +dataset.overrides.test.group=test \
  +dataset.overrides.test.steps_per_epoch=8 \
  dataset.val_loader.batch_size=1 \
  'evaluation.lead_times_min=[60,120,180,240,300,360]' \
  'evaluation.thresholds_mm_h=[0.1,1.0,5.0,10.0]' \
  +evaluation.output_dir="$BENCHMARK_RUN/evaluation_smoke" \
  +visualization.output_dir="$BENCHMARK_RUN/figures_smoke"
```

The lead times above correspond to the six **hourly** targets. The default YAML
uses 10–60 minutes. Record the hourly values for experiment configuration; the
current evaluator still labels results `T+1`, `T+2`, etc., without applying the
minute values to the output.
Evaluation runs the model directly against targets; it does not consume the
historical prediction store.

For full test evaluation, remove `+dataset.overrides.test.steps_per_epoch=8` and
change the output directories to `$BENCHMARK_RUN/evaluation` and
`$BENCHMARK_RUN/figures`. The sampled run is a functional check, not the final
benchmark. `evaluation.max_batches` is currently not enforced by the evaluator.

Outputs include:

- `evaluation_summary.csv`: aggregated metrics.
- `evaluation_summary.parquet`: also written if a Parquet engine is installed.
- Figures in the selected visualization directory, generated after evaluation.

To enable Parquet export, install `pyarrow` with `uv pip install pyarrow`.
To disable a metric family, add an override such as
`evaluation.probabilistic=false`.

## 8. Preview and publish the documentation

```bash
python -m mkdocs serve
```

Open `http://127.0.0.1:8000` on the machine running the server. Validate the site:

```bash
python -m mkdocs build --strict
```

See [Publishing Documentation](publishing.md) for GitHub Pages setup and deployment.

## Troubleshooting

- **CUDA out of memory:** reduce both loader batch sizes to 1; reduce patch sizes
  if needed and keep model spatial requirements aligned.
- **Worker debugging:** use `dataset.train_loader.num_workers=0`,
  `dataset.train_loader.prefetch_factor=null`, and
  `dataset.val_loader.num_workers=0` together for training. For evaluation or
  historical inference, only the validation loader is used.
- **Hydra errors:** use `+key=value` for missing configuration keys, and
  `key=value` for existing keys. Quote list overrides in the shell.
- **Detailed traceback:** prefix the command with `HYDRA_FULL_ERROR=1`.
- **Inspect configuration without running:** append `--cfg job --resolve` to a
  CLI command to check the composed configuration before using compute resources.
