# Model Usage Examples

This page provides a training example for every model configuration available in
`conf/model/`, followed by shared inference and evaluation commands. All examples
use the WPR Zarr, 12 hourly input frames, 6 hourly targets, one channel, and
320 × 320 spatial patches.

See [Workflow Examples](usage.md) for the complete data workflow and
[Parameter Reference](parameters.md) for architecture options and limitations.

## Preparation

Run from the repository root, using Bash and an environment with the package
installed. Run training on a compute node with an allocated GPU.

```bash
source .venv/bin/activate
uv pip install -e '.[dl,verification]'
export BENCHMARK_ZARR=/prj/cptec/nowcasting/data/benchmark/benchmark_dataset_wpr.zarr
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Create the environment first if needed, following [Installation](installation.md).
Replace the Zarr path on other machines. The store must contain `train`,
`validation`, and `test` groups with `gsmap_nrt` inputs and `gsmap_mvk` targets.

Each recipe is a **one-epoch functional check**, with 8 training samples and 4
validation samples. These are sample counts, not batch counts. A checkpoint from
this check is not a validated forecasting model. Batch size 1 reduces memory use,
but does not guarantee that every architecture fits every GPU.

The Bash `MODEL_ARGS` array preserves the selected architecture arguments for
later inference and evaluation. Run a model's block before using those commands.
Each model writes checkpoints to a separate run directory.

## UNet — direct

Predicts all six future frames in one forward pass. This is the default model configuration.

```bash
MODEL_ARGS=(model=unet/direct)
export MODEL_RUN=outputs/model_examples/unet_direct
python main.py task=train "${MODEL_ARGS[@]}" training=default \
  loss=hybrid_mse_ssim \
  dataset.dataset.zarr_path="$BENCHMARK_ZARR" \
  dataset.dataset.patch_height=320 \
  dataset.dataset.patch_width=320 \
  dataset.train_loader.batch_size=1 \
  dataset.val_loader.batch_size=1 \
  dataset.overrides.train.steps_per_epoch=8 \
  dataset.overrides.validation.steps_per_epoch=4 \
  training.epochs=1 \
  training.checkpoint.dir="$MODEL_RUN/checkpoints" \
  hydra.run.dir="$MODEL_RUN"
```

## UNet — autoregressive

Predicts one frame at a time and feeds predictions back into the context window. This YAML fixes 12 input frames and 6 output frames; if changing the root temporal settings, also override `model.input_timesteps` and `model.output_timesteps`.

```bash
MODEL_ARGS=(model=unet/autoregressive)
export MODEL_RUN=outputs/model_examples/unet_autoregressive
python main.py task=train "${MODEL_ARGS[@]}" training=default \
  loss=hybrid_mse_ssim \
  dataset.dataset.zarr_path="$BENCHMARK_ZARR" \
  dataset.dataset.patch_height=320 \
  dataset.dataset.patch_width=320 \
  dataset.train_loader.batch_size=1 \
  dataset.val_loader.batch_size=1 \
  dataset.overrides.train.steps_per_epoch=8 \
  dataset.overrides.validation.steps_per_epoch=4 \
  training.epochs=1 \
  training.checkpoint.dir="$MODEL_RUN/checkpoints" \
  hydra.run.dir="$MODEL_RUN"
```

## ConvLSTM

Uses recurrent convolutional states to encode the sequence and generate future frames. The configured hidden widths are `[32,32,32]`; use a quoted override such as `'hidden_channels=[16,16,16]'` to reduce them, and reuse that setting when loading the checkpoint.

```bash
MODEL_ARGS=(model=convlstm/direct)
export MODEL_RUN=outputs/model_examples/convlstm
python main.py task=train "${MODEL_ARGS[@]}" training=default \
  loss=hybrid_mse_ssim \
  dataset.dataset.zarr_path="$BENCHMARK_ZARR" \
  dataset.dataset.patch_height=320 \
  dataset.dataset.patch_width=320 \
  dataset.train_loader.batch_size=1 \
  dataset.val_loader.batch_size=1 \
  dataset.overrides.train.steps_per_epoch=8 \
  dataset.overrides.validation.steps_per_epoch=4 \
  training.epochs=1 \
  training.checkpoint.dir="$MODEL_RUN/checkpoints" \
  hydra.run.dir="$MODEL_RUN"
```

## AFNO

Uses Fourier-domain blocks. Its spectral weights are sized for the input grid; `model.img_size` is taken automatically from `dataset.dataset.patch_height`/`patch_width`, and the model raises a clear error if an input of another size is passed. The internal `model.patch_size` defaults to 10 and must divide both dimensions; it is separate from the dataset crop size.

```bash
MODEL_ARGS=(model=afno/direct)
export MODEL_RUN=outputs/model_examples/afno
python main.py task=train "${MODEL_ARGS[@]}" training=default \
  loss=hybrid_mse_ssim \
  dataset.dataset.zarr_path="$BENCHMARK_ZARR" \
  dataset.dataset.patch_height=320 \
  dataset.dataset.patch_width=320 \
  dataset.train_loader.batch_size=1 \
  dataset.val_loader.batch_size=1 \
  dataset.overrides.train.steps_per_epoch=8 \
  dataset.overrides.validation.steps_per_epoch=4 \
  training.epochs=1 \
  training.checkpoint.dir="$MODEL_RUN/checkpoints" \
  hydra.run.dir="$MODEL_RUN"
```

## ResNet50

Uses a ResNet50 encoder with a forecasting decoder. This example initializes the encoder without downloaded pretrained weights. Set `+model.pretrained=true` to use pretrained encoder weights when available in the cache or through network access.

```bash
MODEL_ARGS=(model=resnet50/direct +model.pretrained=false)
export MODEL_RUN=outputs/model_examples/resnet50
python main.py task=train "${MODEL_ARGS[@]}" training=default \
  loss=hybrid_mse_ssim \
  dataset.dataset.zarr_path="$BENCHMARK_ZARR" \
  dataset.dataset.patch_height=320 \
  dataset.dataset.patch_width=320 \
  dataset.train_loader.batch_size=1 \
  dataset.val_loader.batch_size=1 \
  dataset.overrides.train.steps_per_epoch=8 \
  dataset.overrides.validation.steps_per_epoch=4 \
  training.epochs=1 \
  training.checkpoint.dir="$MODEL_RUN/checkpoints" \
  hydra.run.dir="$MODEL_RUN"
```

## InceptionV4

Uses an InceptionV4 encoder with a forecasting decoder. `+model.pretrained=false` avoids an encoder-weight download. Use `true` to initialize from pretrained encoder weights.

```bash
MODEL_ARGS=(model=inceptionv4/direct +model.pretrained=false)
export MODEL_RUN=outputs/model_examples/inceptionv4
python main.py task=train "${MODEL_ARGS[@]}" training=default \
  loss=hybrid_mse_ssim \
  dataset.dataset.zarr_path="$BENCHMARK_ZARR" \
  dataset.dataset.patch_height=320 \
  dataset.dataset.patch_width=320 \
  dataset.train_loader.batch_size=1 \
  dataset.val_loader.batch_size=1 \
  dataset.overrides.train.steps_per_epoch=8 \
  dataset.overrides.validation.steps_per_epoch=4 \
  training.epochs=1 \
  training.checkpoint.dir="$MODEL_RUN/checkpoints" \
  hydra.run.dir="$MODEL_RUN"
```

## Xception

Uses an Xception encoder with a forecasting decoder. `+model.pretrained=false` avoids an encoder-weight download. Use `true` to initialize from pretrained encoder weights.

```bash
MODEL_ARGS=(model=xception/direct +model.pretrained=false)
export MODEL_RUN=outputs/model_examples/xception
python main.py task=train "${MODEL_ARGS[@]}" training=default \
  loss=hybrid_mse_ssim \
  dataset.dataset.zarr_path="$BENCHMARK_ZARR" \
  dataset.dataset.patch_height=320 \
  dataset.dataset.patch_width=320 \
  dataset.train_loader.batch_size=1 \
  dataset.val_loader.batch_size=1 \
  dataset.overrides.train.steps_per_epoch=8 \
  dataset.overrides.validation.steps_per_epoch=4 \
  training.epochs=1 \
  training.checkpoint.dir="$MODEL_RUN/checkpoints" \
  hydra.run.dir="$MODEL_RUN"
```

## Run a longer experiment

In the selected model's training command, change:

- `training.epochs=50` for up to 50 epochs, subject to early stopping.
- `dataset.overrides.train.steps_per_epoch=500` for 500 sampled training items.
- `dataset.overrides.validation.steps_per_epoch=500` for sampled validation, or
  `null` for deterministic validation over the full split.
- `MODEL_RUN` to a fresh directory to keep the small check separate.

Keep model, data, loss, and sampling settings recorded when comparing results.
These recipes demonstrate usage; they are not tuned hyperparameters for each
architecture. See [Training Models](training.md) for model-specific details.

## Use any trained model for inference

After running one of the blocks above, keep its `MODEL_ARGS` and `MODEL_RUN`
values in the same shell. The following command works with each recipe's selected
architecture and checkpoint.

```bash
export MODEL_CHECKPOINT="$MODEL_RUN/checkpoints/best_model.pt"
if test -f "$MODEL_CHECKPOINT"; then
  python main.py task=infer "${MODEL_ARGS[@]}" \
    checkpoint="$MODEL_CHECKPOINT" \
    dataset.dataset.zarr_path="$BENCHMARK_ZARR" \
    +dataset.overrides.test.group=test \
    inference.mode=single \
    inference.single.output_format=pt \
    inference.output_dir="$MODEL_RUN/inference"
else
  echo "Checkpoint missing: $MODEL_CHECKPOINT"
fi
```

The guard matters: the current CLI otherwise continues with fresh weights when
a checkpoint file is missing. Single inference predicts the first test patch and
uses a fixed timestamp in the output path:
`$MODEL_RUN/inference/single/2026/03/16/pred_gsmap_20260316_1200.pt`.
This timestamp is not the sample's actual date. Use `inference.single.output_format=nc`
for NetCDF, or follow [historical inference](usage.md#6-generate-historical-forecasts)
for batch output to Zarr.

If starting a new shell, activate the environment and restore `BENCHMARK_ZARR`,
`MODEL_ARGS`, and `MODEL_RUN` from the chosen recipe. Do not rerun its training
command just to load the model. Any custom architecture overrides must also be
restored before loading the checkpoint.

## Evaluate any trained model

With the same variables, run a small evaluation on sampled test data:

```bash
if test -f "$MODEL_CHECKPOINT"; then
  python main.py task=evaluate "${MODEL_ARGS[@]}" \
    checkpoint="$MODEL_CHECKPOINT" \
    dataset.dataset.zarr_path="$BENCHMARK_ZARR" \
    +dataset.overrides.test.group=test \
    +dataset.overrides.test.steps_per_epoch=8 \
    dataset.val_loader.batch_size=1 \
    'evaluation.lead_times_min=[60,120,180,240,300,360]' \
    +evaluation.output_dir="$MODEL_RUN/evaluation" \
    +visualization.output_dir="$MODEL_RUN/figures"
else
  echo "Checkpoint missing: $MODEL_CHECKPOINT"
fi
```

Remove `+dataset.overrides.test.steps_per_epoch=8` for deterministic evaluation
across the test samples. Metrics are written to
`$MODEL_RUN/evaluation/evaluation_summary.csv`, with figures under
`$MODEL_RUN/figures`. The evaluator still labels horizons `T+1` through `T+6`;
for this dataset, they represent 1 through 6 hours despite the minute override
being recorded in configuration.

## GAN and discriminator availability

The repository also contains a GAN training engine and a PatchGAN discriminator.
They are not an additional selectable `model=gan` configuration. The unified
`main.py` training handler runs supervised training only and refuses
`training=gan` with an error; adversarial training uses `scripts/train_gan.py`. The examples above cover all seven model YAML configurations currently
available through this CLI, including both UNet forecasting strategies.
