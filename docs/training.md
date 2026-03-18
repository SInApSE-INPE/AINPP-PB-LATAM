# Training Models

## Overview

This chapter documents the training stack used by AINPP-PB-LATAM from the point of view of the current codebase. It focuses on what is actually configurable today through Hydra and what that means for scientific benchmark experiments.

The training workflow is built around:

- `main.py` as the CLI entry point,
- Hydra for model, dataset, loss, and runtime composition,
- `AINPPPBLATAMDataset` for Zarr-based sequence sampling,
- `run_training` for supervised training,
- `run_gan_training` for adversarial training,
- model definitions under `src/ainpp_pb_latam/models/`.

## End-to-End Training Flow

At runtime the project follows this sequence:

1. `main.py` loads `conf/config.yaml`.
2. Hydra composes the selected `model`, `dataset`, `training`, `loss`, `evaluation`, and `inference` groups.
3. The dataset object is instantiated from `conf/dataset/gsmap.yaml`.
4. The model is instantiated from the selected file in `conf/model/`.
5. The loss is instantiated from the selected file in `conf/loss/`.
6. The optimizer is created from the `training` config.
7. `run_training` performs the epoch loop, validation, checkpointing, and early stopping.

Typical command:

```bash
python main.py task=train model=unet/direct training=default dataset=gsmap loss=hybrid_mse_ssim
```

## Configuration Topology

The root training-related defaults are defined in `conf/config.yaml`:

```yaml
defaults:
  - _self_
  - model: unet/direct
  - discriminator: patchgan
  - training: gan
  - dataset: gsmap
  - loss: hybrid_mse_ssim
  - inference: default
  - evaluation: default
```

Important implications:

- The root file defines shared dimensions such as `input_timesteps`, `output_timesteps`, `input_channels`, `hidden_channels`, and `kernel_size`.
- Several model configs interpolate these shared values instead of redefining them.
- You can modify the global temporal configuration once and let multiple configs inherit it.
- Some models expose additional constructor parameters that are not listed in the YAML by default; Hydra still allows overriding them explicitly.

Example:

```bash
python main.py task=train \
  model=afno/direct \
  input_timesteps=12 \
  output_timesteps=6 \
  +model.embed_dim=384 \
  +model.depth=12
```

## Tensor Shapes and Training Contract

The dataset returns:

- input tensor: `(B, Tin, C, H, W)`
- target tensor: `(B, Tout, C, H, W)`

With the current default dataset and model assumptions:

- `Tin = 12`
- `Tout = 6`
- `C = 1`
- `H = patch_height`
- `W = patch_width`

For the default GSMaP setup:

```text
Input  shape: (B, 12, 1, 320, 320)
Target shape: (B,  6, 1, 320, 320)
```

This contract is central. If you change `input_timesteps`, `output_timesteps`, patch size, or the number of channels, the model and dataset must still agree on the same shape semantics.

## Dataset Configuration in Detail

The dataset configuration lives in `conf/dataset/gsmap.yaml` and instantiates:

```yaml
_target_: ainpp_pb_latam.datasets.gsmap.AINPPPBLATAMDataset
```

### Core Temporal Parameters

- `input_timesteps`: number of historical frames fed into the model
- `output_timesteps`: number of future frames the model must predict
- `stride`: temporal stride between valid samples for each split
- `steps_per_epoch`: if defined, enables random sampling mode instead of deterministic traversal

The current split overrides are:

- `train.group=train`, `stride=1`, `steps_per_epoch=500`
- `validation.group=validation`, `stride=6`, `steps_per_epoch=500`

This means:

- training samples are dense in time and randomly drawn,
- validation samples are sparser in time,
- both training and validation are capped by a fixed number of sampled items per epoch instead of scanning the full split.

### Spatial Parameters

- `patch_height`
- `patch_width`
- `patch_stride_h`
- `patch_stride_w`

Current default:

```yaml
patch_height: 320
patch_width: 320
patch_stride_h: null
patch_stride_w: null
```

When a stride is `null`, the dataset uses the patch size itself. That yields non-overlapping tiles unless the domain edge requires a final snapped patch for full coverage.

### Data Variables

The dataset class supports:

- `input_var`, default `gsmap_nrt`
- `target_var`, default `gsmap_mvk`
- `group`, default `train`
- `dtype`, default `float32`
- `consolidated`, default `true`
- `return_metadata`, default `false`

These fields are useful when experimenting with alternate variables, storage conventions, or metadata-aware debugging.

### Changing Input and Target Lengths

To modify the historical window and forecast horizon:

```bash
python main.py task=train \
  model=unet/direct \
  input_timesteps=18 \
  output_timesteps=12
```

Because the dataset and most model configs interpolate these root values, this is the preferred way to change sequence lengths.

### Changing Spatial Crop Size

To train on full-domain inputs:

```bash
python main.py task=train \
  model=unet/direct \
  dataset.dataset.patch_height=null \
  dataset.dataset.patch_width=null
```

To train on smaller patches:

```bash
python main.py task=train \
  model=unet/direct \
  dataset.dataset.patch_height=256 \
  dataset.dataset.patch_width=256
```

To create overlapping windows:

```bash
python main.py task=train \
  model=unet/direct \
  dataset.dataset.patch_height=320 \
  dataset.dataset.patch_width=320 \
  dataset.dataset.patch_stride_h=160 \
  dataset.dataset.patch_stride_w=160
```

### Changing Sampling Density

To use fewer random samples per epoch for fast prototyping:

```bash
python main.py task=train \
  model=unet/direct \
  dataset.overrides.train.steps_per_epoch=100 \
  dataset.overrides.validation.steps_per_epoch=50
```

To force deterministic validation over the full split:

```bash
python main.py task=train \
  model=unet/direct \
  dataset.overrides.validation.steps_per_epoch=null
```

### Changing Input and Target Variables

If the Zarr store contains alternative variable names:

```bash
python main.py task=train \
  model=unet/direct \
  dataset.dataset.input_var=gsmap_nrt \
  dataset.dataset.target_var=gsmap_mvk
```

### Dataloader Parameters

The dataset config also controls:

- `train_loader.batch_size`
- `train_loader.num_workers`
- `train_loader.prefetch_factor`
- `train_loader.pin_memory`
- `val_loader.batch_size`
- `val_loader.num_workers`
- `val_loader.pin_memory`

Example:

```bash
python main.py task=train \
  model=unet/direct \
  dataset.train_loader.batch_size=4 \
  dataset.train_loader.num_workers=8 \
  dataset.val_loader.batch_size=4
```

## Supervised Training Configuration

The default supervised profile is `conf/training/default.yaml`.

Key fields:

- `mode: supervised`
- `epochs: 50`
- `lr: 0.001`
- `batch_size: 16`
- `scheduler.patience`
- `scheduler.factor`
- `checkpoint.enabled`
- `checkpoint.dir`
- `checkpoint.interval`
- `checkpoint.save_best`
- `early_stopping.enabled`
- `early_stopping.patience`
- `early_stopping.delta`
- `early_stopping.mode`

Notes from the current implementation:

- The optimizer is always Adam through `build_optimizer`.
- `lr` is used unless the config only defines `lr_g`.
- `beta1` and `beta2` are also read if present in the training config.
- The scheduler block exists in YAML, but no scheduler is currently attached inside `run_training`.

Example of a slower, more conservative run:

```bash
python main.py task=train \
  model=unet/direct \
  training=default \
  training.epochs=100 \
  training.lr=0.0003 \
  training.early_stopping.patience=20 \
  training.checkpoint.interval=10
```

## GAN Training Configuration

The adversarial profile is `conf/training/gan.yaml`.

Key fields:

- `mode: gan`
- `epochs: 100`
- `lr_g`
- `lr_d`
- `beta1`
- `beta2`
- `lambda_pixel`
- checkpoint settings
- early stopping settings

The intended logic in `run_gan_training` is:

- generator predicts future rainfall,
- discriminator sees the concatenated history and future sequence,
- discriminator learns to distinguish real future targets from generated future targets,
- generator balances adversarial realism and pixel-level accuracy.

The paired discriminator config is:

```yaml
_target_: ainpp_pb_latam.models.gan.discriminator.PatchDiscriminator3D
input_channels: 1
ndf: 64
n_layers: 1
norm_type: "instance"
```

A typical adversarial run would conceptually look like:

```bash
python main.py task=train \
  model=unet/direct \
  training=gan \
  discriminator=patchgan
```

However, note one practical detail: the current `main.py` path dispatches `task=train` through `run_training`. If you want full GAN training behavior, the CLI path still needs to branch into `run_gan_training` when `training.mode=gan`.

## Model Catalog

This section summarizes the models currently wired into Hydra under `conf/model/`.

### UNet Direct

Config file:

```yaml
_target_: ainpp_pb_latam.models.unet.forecaster.UNetMultiHorizon

input_timesteps: ${input_timesteps}
input_channels: 1
output_timesteps: ${output_timesteps}
output_channels: 1
features: [64, 128, 256, 512]
kernel_size: 3
bilinear: true
nonnegativity: "relu"
```

Training behavior:

- flattens the temporal dimension into channels,
- applies a 2D U-Net to the stacked history,
- predicts all future horizons in one forward pass,
- reshapes the output back to `(B, Tout, C, H, W)`,
- applies a non-negativity constraint at the end.

Best when:

- you want a strong baseline,
- you need explicit control over encoder depth,
- you want predictable behavior on patch-based training.

Important parameters:

- `features`: controls encoder and decoder width at each level
- `kernel_size`: spatial receptive field per block
- `bilinear`: chooses bilinear upsampling instead of transposed convolution
- `nonnegativity`: `relu`, `softplus`, or `none`

Examples:

```bash
python main.py task=train \
  model=unet/direct \
  model.features=[32,64,128,256] \
  model.kernel_size=5 \
  model.bilinear=false \
  model.nonnegativity=softplus
```

### UNet Autoregressive

Config file:

```yaml
_target_: ainpp_pb_latam.models.unet.forecaster.UNetAutoRegressive

input_timesteps: 12
input_channels: 1
output_timesteps: 6
features: [64, 128, 256, 512]
kernel_size: 3
bilinear: true
nonnegativity: "relu"
```

Training behavior:

- predicts one future frame at a time,
- appends each prediction back into the context window,
- rolls forward until the requested forecast horizon is reached.

Best when:

- your scientific question values sequential dependency between horizons,
- you want the model to explicitly learn rollout dynamics,
- error propagation across future steps is acceptable or desired to study.

Important caution:

- this file currently hardcodes `input_timesteps=12` and `output_timesteps=6` rather than interpolating root values,
- if you change the global root dimensions, also override the model fields explicitly for the autoregressive U-Net.

Example:

```bash
python main.py task=train \
  model=unet/autoregressive \
  model.input_timesteps=18 \
  model.output_timesteps=12 \
  dataset.dataset.input_timesteps=18 \
  dataset.dataset.output_timesteps=12
```

### ConvLSTM

Config file:

```yaml
_target_: ainpp_pb_latam.models.convlstm.forecaster.ConvLSTMMultiHorizon

input_channels: ${input_channels}
hidden_channels: ${hidden_channels}
kernel_size: ${kernel_size}
output_timesteps: ${output_timesteps}
```

Training behavior:

- uses a ConvLSTM encoder-decoder,
- processes the input sequence recurrently,
- decodes future steps autoregressively from latent state,
- maps hidden features into one precipitation channel through a small output head.

Best when:

- temporal recurrence is central to the experiment,
- you want a sequence model without flattening time into channels,
- you want to study the effect of hidden-state depth and recurrent receptive field.

Important parameters:

- `hidden_channels`: number and width of ConvLSTM layers
- `kernel_size`: convolution kernel used inside recurrent cells
- `output_timesteps`: rollout horizon

Examples:

```bash
python main.py task=train \
  model=convlstm/direct \
  hidden_channels=[32,32,64] \
  kernel_size=5
```

```bash
python main.py task=train \
  model=convlstm/direct \
  input_channels=1 \
  output_timesteps=12 \
  dataset.dataset.output_timesteps=12
```

### AFNO

Config file:

```yaml
_target_: ainpp_pb_latam.models.afno.forecaster.AFNO2D

input_timesteps: ${input_timesteps}
output_timesteps: ${output_timesteps}
```

Additional constructor parameters supported by the code:

- `img_size`
- `input_channels`
- `output_channels`
- `embed_dim`
- `depth`
- `patch_size`
- `num_blocks`

Training behavior:

- flattens time into channels,
- embeds the spatial field into non-overlapping patches,
- applies a stack of Fourier blocks in latent space,
- reconstructs the output with a transposed convolution head.

Best when:

- global spatial coupling matters,
- you want a spectral model,
- patch-token processing is more attractive than deep CNN decoding.

Critical caveat:

- `AFNO2D` defaults to `img_size=(880, 970)` and `patch_size=10`,
- if your dataset crops are `320 x 320`, you should override `model.img_size` accordingly,
- `patch_size` must divide both image dimensions used by the model.

Examples:

```bash
python main.py task=train \
  model=afno/direct \
  dataset.dataset.patch_height=320 \
  dataset.dataset.patch_width=320 \
  +model.img_size=[320,320] \
  +model.patch_size=10 \
  +model.embed_dim=384 \
  +model.depth=12 \
  +model.num_blocks=12
```

```bash
python main.py task=train \
  model=afno/direct \
  dataset.dataset.patch_height=256 \
  dataset.dataset.patch_width=256 \
  +model.img_size=[256,256] \
  +model.patch_size=16
```

### ResNet50

Config file:

```yaml
_target_: ainpp_pb_latam.models.resnet50.forecaster.ResNet50MultiHorizon

input_timesteps: ${input_timesteps}
output_timesteps: ${output_timesteps}
```

Additional constructor parameter supported by the code:

- `pretrained`

Training behavior:

- collapses time into the channel dimension,
- uses `timm` `resnet50d` as a feature extractor,
- decodes multi-scale features through U-Net-like upsampling blocks,
- predicts all future steps jointly.

Best when:

- you want an ImageNet-style convolutional encoder,
- transfer learning from pretrained image backbones is acceptable,
- the experiment benefits from a robust CNN feature hierarchy.

Examples:

```bash
python main.py task=train \
  model=resnet50/direct \
  +model.pretrained=true
```

```bash
python main.py task=train \
  model=resnet50/direct \
  input_timesteps=18
```

Because `in_chans=input_timesteps`, changing `input_timesteps` changes the first convolution shape in the `timm` backbone.

### InceptionV4

Config file:

```yaml
_target_: ainpp_pb_latam.models.inceptionv4.forecaster.InceptionV4MultiHorizon

input_timesteps: ${input_timesteps}
output_timesteps: ${output_timesteps}
```

Additional constructor parameter supported by the code:

- `pretrained`

Training behavior:

- uses a `timm` Inception-V4 encoder with `features_only=True`,
- decodes the multiscale feature pyramid back to full resolution,
- predicts all horizons in one shot,
- enforces non-negative rainfall through `relu`.

Best when:

- you want a deeper multi-branch CNN encoder,
- you want strong spatial feature extraction with pretrained weights,
- you are benchmarking classic computer vision backbones against nowcasting-specific designs.

Example:

```bash
python main.py task=train \
  model=inceptionv4/direct \
  +model.pretrained=false
```

### Xception

Config file:

```yaml
_target_: ainpp_pb_latam.models.xception.forecaster.XceptionMultiHorizon

input_timesteps: ${input_timesteps}
output_timesteps: ${output_timesteps}
```

Additional constructor parameter supported by the code:

- `pretrained`

Training behavior:

- uses a `timm` Xception encoder adapted to `in_chans=input_timesteps`,
- decodes through skip-connected upsampling blocks,
- predicts all future frames simultaneously,
- applies `relu` to the output before restoring the channel dimension.

Best when:

- you want depthwise-separable convolutional features,
- you want a lighter alternative to some classical heavy backbones,
- you want to compare pretrained encoder transfer against U-Net and ConvLSTM baselines.

Example:

```bash
python main.py task=train \
  model=xception/direct \
  +model.pretrained=true
```

## Loss Functions

Loss functions are instantiated from `conf/loss/`.

### `weighted_mse`

Config:

```yaml
_target_: ainpp_pb_latam.losses.WeightedMSELoss
alpha: 5.0
threshold: 0.1
```

Use this when:

- you want to upweight heavy-rain pixels,
- standard MSE underestimates intense precipitation,
- the benchmark should favor amplitude accuracy in high-value regions.

Higher `alpha` increases the emphasis on rain cores. `threshold` defines from which target intensity the extra weighting starts.

Example:

```bash
python main.py task=train \
  model=unet/direct \
  loss=weighted_mse \
  loss.alpha=10.0 \
  loss.threshold=1.0
```

### `huber`

Robust against outliers and often a safer regression baseline than plain MSE on noisy fields.

Example:

```bash
python main.py task=train model=unet/direct loss=huber
```

### `logcosh`

Smooth transition between L2-like and L1-like behavior. Useful when you want stable optimization but less sensitivity to extreme residuals than MSE.

### `dice`

Optimizes overlap on a rain mask rather than continuous intensity values.

Important caveat:

- this is best for event extent, not calibrated rainfall amplitude,
- it binarizes the target using the configured threshold.

Example:

```bash
python main.py task=train \
  model=unet/direct \
  loss=dice \
  loss.threshold=0.5
```

### `focal`

Binary focal loss over thresholded rainfall occurrence.

Important caveat:

- the implementation expects logits and internally applies `binary_cross_entropy_with_logits`,
- this is more appropriate for event detection than direct rainfall regression.

Example:

```bash
python main.py task=train \
  model=unet/direct \
  loss=focal \
  loss.threshold=0.1 \
  loss.alpha=0.25 \
  loss.gamma=2.0
```

### `spectral`

Frequency-domain loss to preserve structure and reduce blur.

Example:

```bash
python main.py task=train \
  model=afno/direct \
  loss=spectral \
  loss.alpha=1.0 \
  loss.beta=0.5
```

### `torrential`

Tiered intensity weighting for severe rainfall events.

Example:

```bash
python main.py task=train \
  model=unet/direct \
  loss=torrential \
  loss.thresholds=[5.0,20.0,50.0] \
  loss.weights=[2.0,5.0,10.0]
```

### `hybrid_mse_ssim`

Current config:

```yaml
_target_: ainpp_pb_latam.losses.HybridLoss
weights: [1.0, 0.2]

losses:
  - _target_: ainpp_pb_latam.losses.WeightedMSELoss
    alpha: 2.0
    threshold: 0.0
  - _target_: ainpp_pb_latam.losses.SSIMLoss
    window_size: 11
    in_channels: 1
```

This is a practical default when you want:

- amplitude fidelity,
- structural coherence,
- reduced blur compared with pure pixel losses.

Example:

```bash
python main.py task=train \
  model=unet/direct \
  loss=hybrid_mse_ssim \
  loss.weights=[1.0,0.1] \
  loss.losses[0].alpha=4.0 \
  loss.losses[1].window_size=7
```

### `sota`

Current config combines:

- `AdvancedTorrentialLoss`
- `SpectralLoss`
- `PerceptualLoss`

This profile is the most ambitious option in the repository because it combines amplitude weighting, frequency structure, and image-feature realism.

Important caution:

- `PerceptualLoss` attempts to load VGG16 pretrained weights,
- if weights are unavailable, it falls back to MSE-like behavior,
- this may make runs less reproducible across environments if internet or cached weights differ.

## Recommended Override Patterns

### Change Only the Model Family

```bash
python main.py task=train model=convlstm/direct training=default loss=hybrid_mse_ssim
```

### Change Forecast Horizon

```bash
python main.py task=train \
  model=unet/direct \
  input_timesteps=24 \
  output_timesteps=12 \
  dataset.dataset.input_timesteps=24 \
  dataset.dataset.output_timesteps=12
```

### Change Patch Geometry

```bash
python main.py task=train \
  model=resnet50/direct \
  dataset.dataset.patch_height=256 \
  dataset.dataset.patch_width=384 \
  dataset.dataset.patch_stride_h=128 \
  dataset.dataset.patch_stride_w=192
```

### Change Rain-Event Emphasis

```bash
python main.py task=train \
  model=unet/direct \
  loss=weighted_mse \
  loss.alpha=8.0 \
  loss.threshold=1.0
```

### Disable Pretrained Weights

```bash
python main.py task=train \
  model=xception/direct \
  +model.pretrained=false
```

### Increase Recurrent Capacity

```bash
python main.py task=train \
  model=convlstm/direct \
  hidden_channels=[64,64,128,128] \
  kernel_size=5
```

## Practical Constraints and Caveats

These are important when designing experiments:

- `UNetAutoRegressive` does not interpolate root dimensions in its YAML by default.
- `AFNO2D` requires `img_size` and `patch_size` compatibility with the actual crop size.
- The scheduler block exists in config but is not currently consumed by the supervised engine.
- The current `task=train` path in `main.py` does not yet branch into `run_gan_training` automatically.
- `timm` backbones depend on the installed deep learning extras and pretrained weight availability.
- Large patch sizes and deep CNN encoders will increase memory pressure quickly on HPC jobs.

## Experiment Design Suggestions

For a reliable benchmark progression:

1. Start with `unet/direct` and `loss=hybrid_mse_ssim`.
2. Tune patch size and `steps_per_epoch` until I/O and GPU memory are stable.
3. Sweep rain-emphasis losses such as `weighted_mse` and `torrential`.
4. Benchmark recurrent behavior with `convlstm/direct`.
5. Benchmark transfer-learning encoders with `resnet50/direct`, `inceptionv4/direct`, and `xception/direct`.
6. Use `afno/direct` only after aligning `model.img_size` with the actual crop geometry.

## Minimal Reproducible Training Recipes

### Strong Baseline

```bash
python main.py task=train \
  model=unet/direct \
  training=default \
  loss=hybrid_mse_ssim \
  dataset.train_loader.batch_size=4 \
  dataset.val_loader.batch_size=4
```

### Heavy-Rain Baseline

```bash
python main.py task=train \
  model=unet/direct \
  training=default \
  loss=torrential \
  loss.thresholds=[10.0,30.0,50.0] \
  loss.weights=[2.0,5.0,10.0]
```

### Recurrent Baseline

```bash
python main.py task=train \
  model=convlstm/direct \
  training=default \
  loss=weighted_mse \
  hidden_channels=[32,64,64]
```

### Spectral Baseline

```bash
python main.py task=train \
  model=afno/direct \
  training=default \
  loss=spectral \
  dataset.dataset.patch_height=320 \
  dataset.dataset.patch_width=320 \
  +model.img_size=[320,320]
```
