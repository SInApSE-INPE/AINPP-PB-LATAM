# Parameter Reference

This page describes every field in the files under `conf/`, the additional
arguments supported by selectable datasets, models, and losses, and how `main.py`
uses them. Defaults below come from the repository; Python constructor defaults
are identified separately. For complete commands using the WPR dataset, see
[Workflow Examples](usage.md). For internal function signatures, see the
[API](api.md).

## How to configure parameters

Run commands from the repository root with the package installed in the active
environment. Hydra composes `conf/config.yaml` and the selected groups. `_target_`
is the Python path of the class Hydra instantiates; normally change the group
rather than this field. `${name}` references another configuration field, not a
Bash variable.

| Syntax | Purpose | Example |
| --- | --- | --- |
| `group=option` | Select a YAML configuration | `model=unet/direct` |
| `key=value` | Override an existing field | `training.epochs=10` |
| `+key=value` | Add a missing field | `+dataset.dataset.seed=123` |
| `++key=value` | Add or override a field | `++dataset.overrides.test.group=test` |
| `null` | Null value, without inner quotes | `dataset.dataset.patch_height=null` |
| `true`, `false` | Boolean values | `model.bilinear=false` |
| `'key=[a,b]'` | List; quotes protect against shell expansion | `'model.features=[32,64,128,256]'` |
| `--cfg job --resolve` | Display the resolved configuration without running the task | `python main.py training=default --cfg job --resolve` |
| `--help` | List groups and configuration | `python main.py --help` |
| `--multirun` | Run combinations of values | `python main.py --multirun training=default training.lr=0.001,0.0003` |

A multirun actually starts training jobs. Add the Zarr path and use separate
checkpoint directories for each combination.

```bash
python main.py task=train training=default model=unet/direct \
  dataset.dataset.zarr_path=/prj/cptec/nowcasting/data/benchmark/benchmark_dataset_wpr.zarr \
  training.lr=0.0003 dataset.train_loader.batch_size=2 \
  --cfg job --resolve
```

## Groups and global parameters

| Field | Default | Description and usage |
| --- | --- | --- |
| `task` | `train` | `train`, `infer`, or `evaluate`. The unified CLI does not provide `task=visualize`. |
| `checkpoint` | `null` | Weights file for inference/evaluation; must match the architecture. Does not resume training. A missing file results in a warning and a model with fresh weights. |
| `model` | `unet/direct` | Options: `unet/direct`, `unet/autoregressive`, `convlstm/direct`, `afno/direct`, `resnet50/direct`, `inceptionv4/direct`, `xception/direct`. |
| `training` | `gan` | Profiles: `default` and `gan`. For supervised training, explicitly use `training=default`. The CLI calls the supervised engine for both. |
| `dataset` | `gsmap` | Configures the dataset and loaders; does not automatically select the WPR store. |
| `loss` | `hybrid_mse_ssim` | See the loss catalog below. |
| `discriminator` | `patchgan` | Available configuration, but not instantiated by training in `main.py`. |
| `inference` | `default` | Inference profile. |
| `evaluation` | `default` | Metrics profile. |
| `input_timesteps` | `12` | Number of input frames, a positive integer; 12 hours for hourly data. |
| `output_timesteps` | `6` | Number of predicted frames, a positive integer. Does not configure the temporal frequency of the data. |
| `input_channels` | `1` | Referenced by the ConvLSTM YAML. Does not automatically change the loader or every model. |
| `hidden_channels` | `[32,32,32]` | ConvLSTM layer widths; one entry per layer. |
| `kernel_size` | `7` | ConvLSTM kernel. UNet has its own `model.kernel_size=3`. |
| `system.num_workers` | `4` | Declared, but not applied by the loaders in `main.py`. Use `dataset.*_loader.num_workers`. |
| `system.pin_memory` | `true` | Declared, but not applied by the CLI. Use the loader fields. |
| `system.sync_bn` | `false` | Does not activate SyncBatchNorm conversion in the CLI. |

The current dataset returns one channel per frame. Increasing `input_channels`
does not stack variables: that requires adapting the loader. The autoregressive
UNet fixes 12/6 in its YAML; also override `model.input_timesteps` and
`model.output_timesteps` when changing the global values.

## Dataset: `dataset.dataset`

The double prefix is intentional: `dataset` contains the `dataset` object,
loaders, and split overrides. `_target_` points to
`ainpp_pb_latam.datasets.gsmap.AINPPPBLATAMDataset`.

| Field after `dataset.dataset.` | Default | Effect and constraints |
| --- | --- | --- |
| `zarr_path` | `/prj/ideeps/adriano.almeida/data/ainpp/legacy/AINPP-PB-LATAM.zarr` | Accessible local Zarr path; replace it with your store. |
| `input_timesteps` | `${input_timesteps}` | Input window length. |
| `output_timesteps` | `${output_timesteps}` | Target window length, immediately following the input. |
| `patch_height` | `320` | Height in pixels; positive and no larger than the grid. `null` uses the full height. |
| `patch_width` | `320` | Width in pixels; `null` uses the full width. |
| `patch_stride_h` | `null` | Vertical spatial step, a positive integer. `null` uses the patch height. |
| `patch_stride_w` | `null` | Horizontal step; `null` uses the patch width. A smaller step creates overlap. |
| `consolidated` | `true` | Opens consolidated metadata; use `false` if the store does not have it. |
| `dtype` | `float32` | Tensor type resolved through `torch`, not automatic model conversion. Other types must be compatible with the weights and operations. |

The following arguments exist in the constructor but not in this object's YAML.
Add them with `+dataset.dataset.<field>=...` when needed.

| Argument | Python default | Usage |
| --- | --- | --- |
| `group` | `train` | Zarr group; split overrides take precedence. |
| `stride` | `null` | Temporal step in frames; when null, uses `output_timesteps` for `train` and input+output length for other groups. |
| `steps_per_epoch` | `null` | A positive integer limits the number of random samples. `null` traverses time/patch combinations deterministically. |
| `seed` | `42` | Seed for the dataset's NumPy generator; not a global PyTorch seed and does not guarantee reproducibility across workers. |
| `input_var` | `gsmap_nrt` | Input variable with `time`, `lat`, `lon` dimensions. |
| `target_var` | `gsmap_mvk` | Target variable with the same geometry. |
| `return_metadata` | `false` | Returns metadata as a third item; supervised training expects `(x,y)` pairs, so keep it `false` for that workflow. |

The loader replaces NaNs with zero. Windows are index-based: it does not validate
hourly timestamp continuity. The frame count must accommodate input+output length.
The final patches may overlap at the boundary even when the stride equals the
patch size.

### Splits: `dataset.overrides`

| Full field | Default | Usage |
| --- | --- | --- |
| `dataset.overrides.train.group` | `train` | Training group. |
| `dataset.overrides.train.stride` | `1` | Step between eligible temporal starting points. |
| `dataset.overrides.train.steps_per_epoch` | `500` | 500 random samples per epoch, not 500 batches. |
| `dataset.overrides.validation.group` | `validation` | Validation group during training. |
| `dataset.overrides.validation.stride` | `6` | Validation temporal step. |
| `dataset.overrides.validation.steps_per_epoch` | `500` | Sampled validation; `null` enables full traversal. |
| `dataset.overrides.test.group` | Absent | Add `+dataset.overrides.test.group=test` for inference/evaluation on the test split. Otherwise, the constructor uses `train`. |
| `dataset.overrides.test.stride` | Absent | Add to change the test temporal step. |
| `dataset.overrides.test.steps_per_epoch` | Absent | Add an integer for a quick random check; omit to retain deterministic traversal. |

Full validation example: `dataset.overrides.validation.steps_per_epoch=null`.
With 500 samples and batch size 2, the loader has 250 batches. Reducing the stride
increases the number of candidate windows; it does not change the 500-sample limit
when that limit is set.

### Loaders

| Field | Default | Effect |
| --- | --- | --- |
| `dataset.train_loader.batch_size` | `16` | Samples per training batch; reduce to lower memory usage. |
| `dataset.val_loader.batch_size` | `16` | Batch size for validation, evaluation, and historical inference. |
| `dataset.train_loader.num_workers` | `4` | Training data-loading processes; `0` runs in the main process. |
| `dataset.val_loader.num_workers` | `4` | Data-loading processes for the other workflows that use a loader. |
| `dataset.train_loader.prefetch_factor` | `2` | Prefetched batches per worker; use `null` when `num_workers=0`. |
| `dataset.train_loader.pin_memory` | `true` | Pinned memory for device transfers. |
| `dataset.val_loader.pin_memory` | `true` | The same option for the other loaders. |

These dictionaries are passed to `DataLoader`; additional arguments depend on the
installed PyTorch API. Avoid adding `shuffle` to the validation loader: the CLI
already passes `shuffle=False` explicitly for evaluation/historical inference.

## Training: `training`

| Field | `default` / `gan` | Effect in the current CLI |
| --- | --- | --- |
| `mode` | `supervised` / `gan` | Label; does not select the engine in `main.py`. |
| `epochs` | `50` / `100` | Maximum epoch count; positive integer. |
| `lr` | `0.001` / absent | Adam learning rate; takes precedence over `lr_g` if present. |
| `lr_g` | absent / `0.0002` | Fallback learning rate for the CLI's single optimizer. |
| `lr_d` | absent / `0.0002` | Not used by the supervised CLI. |
| `beta1` | fallback `0.9` / `0.5` | First Adam coefficient. Use `+training.beta1=...` with the `default` profile. |
| `beta2` | fallback `0.999` / `0.999` | Second Adam coefficient. |
| `lambda_pixel` | absent / `100.0` | Content weight in the separate GAN engine; not used by the CLI. |
| `batch_size` | `16` / absent | Does not set loader batch sizes. |
| `scheduler.patience` | `5` / absent | Declared; the engine does not create a scheduler. |
| `scheduler.factor` | `0.1` / absent | Declared; does not automatically reduce the learning rate. |

Adam is fixed in `build_optimizer`; there is no optimizer selector in the YAML.

### Checkpoints and early stopping

Both profiles share these values.

| Field after `training.` | Default | Actual effect |
| --- | --- | --- |
| `checkpoint.enabled` | `true` | Enables periodic checkpoints; does not disable saving through early stopping. |
| `checkpoint.dir` | `outputs/<date>/<time>/early_stopping` | Directory for all checkpoints from this engine. |
| `checkpoint.interval` | `5` | Saves every N epochs; use an integer greater than zero. |
| `checkpoint.save_best` | `true` | Declared, but not checked by the engine. |
| `early_stopping.enabled` | `true` | Enables early stopping and best-model saving. |
| `early_stopping.patience` | `10` | Epochs without sufficient improvement before stopping. |
| `early_stopping.delta` | `0.001` | Minimum validation-loss reduction considered an improvement. |
| `early_stopping.mode` | `min` | The engine does not forward this field; it minimizes loss even if configured as `max`. |

Best model: `best_model.pt`. Periodic checkpoints:
`checkpoint_model_epoch_005.pt`, etc. These are `state_dict`s without Adam state
or an epoch counter. Disabling early stopping also prevents best-model saving
through that mechanism. Sample images are written to `samples/`, a fixed path in
the engine.

## Models: `model`

All models must respect the `(B,T,C,H,W)` tensor contract. When changing the
architecture, use the same arguments when loading its checkpoint.

### Direct and autoregressive UNet

| Field after `model.` | YAML default | Usage |
| --- | --- | --- |
| `input_timesteps` | Global for direct; `12` for autoregressive | Context frames. |
| `output_timesteps` | Global for direct; `6` for autoregressive | Predicted horizons. |
| `input_channels` | `1` | Channels per frame. |
| `output_channels` | `1`, direct only | Output channels; the autoregressive model uses the input channel count. |
| `features` | `[64,128,256,512]` | Level widths; larger values increase capacity and memory usage. |
| `kernel_size` | `3` | Kernel size; use a positive odd number for the expected geometry. |
| `bilinear` | `true` | Bilinear upsampling; `false` uses transposed convolution. |
| `nonnegativity` | `relu` | `relu` clips negative values, `softplus` produces smooth positive values, and `none` leaves outputs unconstrained. |

```bash
python main.py training=default model=unet/direct \
  'model.features=[32,64,128,256]' model.nonnegativity=softplus --cfg job --resolve
```

### ConvLSTM

| Field after `model.` | Composed default | Usage |
| --- | --- | --- |
| `input_channels` | `${input_channels}` = `1` | Channels per frame. |
| `hidden_channels` | `${hidden_channels}` = `[32,32,32]` | Widths and number of recurrent layers. The standalone constructor uses `[64,64,64]`. |
| `kernel_size` | `${kernel_size}` = `7` | Recurrent spatial kernel. |
| `output_timesteps` | `${output_timesteps}` = `6` | Steps predicted by the decoder. |

This constructor has no `model.input_timesteps` argument: context comes from the
tensor.

### AFNO

The temporal fields exist in the YAML; the others require `+model.<field>=...`.

| Field | Default | Usage/constraint |
| --- | --- | --- |
| `input_timesteps` | Global `12` (Python: `6`) | Frames stacked at the input. |
| `output_timesteps` | Global `6` | Output frames. |
| `img_size` | Python `[880,970]` | Must match the height/width of the supplied data; use `[320,320]` for the default dataset patches. |
| `input_channels` | Python `1` | Channels per input frame. |
| `output_channels` | Python `1` | Channels per predicted frame. |
| `embed_dim` | Python `256` | Embedding dimension, divisible by `num_blocks`. |
| `depth` | Python `8` | Number of AFNO blocks. |
| `patch_size` | Python `10` | Internal network patch size; must divide both `img_size` dimensions. This is not the dataset crop size. |
| `num_blocks` | Python `8` | Channel partitions in the spectral operation. |

Example: `model=afno/direct '+model.img_size=[320,320]' +model.depth=4`.

### ResNet50, InceptionV4, and Xception

All expose `model.input_timesteps=${input_timesteps}` and
`model.output_timesteps=${output_timesteps}`. The additional argument
`pretrained` has a Python default of `true`: it loads encoder weights and may
require a download/cache. Use `+model.pretrained=false` to initialize without
those weights. These constructors do not expose `input_channels`; the workflow
assumes one channel per frame. Example:
`model=resnet50/direct +model.pretrained=false`.

## Losses: `loss`

Thresholds use the target units (mm/h for untransformed hourly GSMaP).
They do not automatically convert units or normalize data.

| Group | YAML fields and defaults | Meaning |
| --- | --- | --- |
| `weighted_mse` | `alpha=5.0`, `threshold=0.1` | Above the threshold, the weight is `1 + alpha * target`; otherwise, it is 1. `alpha<=0` disables weighting. |
| `logcosh` | No arguments | Mean of `log(cosh(pred-target))`. |
| `huber` | No additional fields | Quadratic/linear transition; Python argument `delta=1.0`, adjustable with `+loss.delta=2.0`. |
| `dice` | `threshold=0.5`, `smooth=1e-6` | Binary target above the threshold; sigmoid-smoothed prediction. `smooth` stabilizes division. |
| `focal` | `alpha=0.25`, `gamma=2.0`, `threshold=0.1` | Loss multiplier, focus on difficult examples, and target threshold. The implementation multiplies the entire term by `alpha`, without separate class weights. |
| `torrential` | `thresholds=[5,20,50]`, `weights=[2,5,10]` | MSE with weights by range; the last reached threshold determines the weight. Lists must have equal lengths, with thresholds in ascending order. |
| `spectral` | `alpha=1.0`, `beta=1.0` | Weights for FFT amplitude error and complex-plane error, respectively. |
| `hybrid_mse_ssim` | `weights=[1.0,0.2]`, `losses` with 2 components | Weighted sum of WeightedMSE (`alpha=2`, `threshold=0`) and SSIM (`window_size=11`, `in_channels=1`). |
| `sota` | `weights=[1.0,0.1,0.05]`, `losses` with 3 components | Torrential (`thresholds=[10]`, `weights=[5]`), Spectral (`alpha=1`, `beta=0.5`), and Perceptual (`layer_ids=[3,8,15]`). |

Hybrid losses require one weight per component. Use an index to edit an item:

```bash
python main.py training=default loss=hybrid_mse_ssim \
  'loss.weights=[1.0,0.3]' loss.losses.0.alpha=3.0 \
  loss.losses.1.window_size=7 --cfg job --resolve
```

`window_size` is the SSIM spatial window (a positive odd number); `in_channels`
must match the data. `layer_ids` selects VGG16 feature layers. Perceptual loss
attempts to load pretrained weights and falls back to MSE if that fails.
The Python default for `layer_ids` is `[3,8,15,22]`, unlike the `sota` group.

`BinaryFocalLoss` expects logits; changing the loss of a network with ReLU outputs
does not turn the pipeline into a suitable classification setup.
`CrossEntropyLoss` is available through the API, without a YAML group: its
`weights=None` argument accepts class weights, and it requires logits/integer
classes, which are not directly compatible with the current regression targets.
Standalone SSIM and Perceptual losses do not have their own YAML groups either.

## Discriminator: `discriminator`

`patchgan` points to `PatchDiscriminator3D`. These fields only take effect when a
workflow instantiates and uses that discriminator; `main.py` does not.

| Field | Default | Usage |
| --- | --- | --- |
| `input_channels` | `1` | Channels in the `(B,C,T,H,W)` tensor. The GAN engine concatenates history and future along time, not channels. |
| `ndf` | `64` | Base filter count. |
| `n_layers` | `1` | Discriminator depth. |
| `norm_type` | `instance` | `batch` uses BatchNorm3d; `instance` uses InstanceNorm3d (also the fallback for other values). |

## Inference: `inference`

| Field | Default | Effect |
| --- | --- | --- |
| `mode` | `historical` | `single` predicts the first sample; `historical` traverses the loader. |
| `output_dir` | `outputs/inference` | Base directory interpolated into the specific paths. |
| `batch_size` | `16` | First axis of the Zarr chunk; does not control loader batch size. |
| `historical.output_format` | `zarr` | Declared; the implementation writes Zarr regardless of this value. |
| `historical.zarr_store` | `${inference.output_dir}/predictions.zarr` | Destination store, overwritten on the first write. |
| `single.output_format` | `nc` | `nc` writes NetCDF; use `pt` for a PyTorch tensor. Other values fall back to `.pt`. |
| `single.output_dir` | `${inference.output_dir}/single` | Base directory for the year/month/day hierarchy. |

Use `dataset.val_loader.batch_size` to control memory/batches for historical
inference. Single-sample prediction uses a fixed CLI timestamp (`20260316_1200`);
there is no date-selection argument. Current files do not preserve real patch
timestamps/coordinates. The historical store contains `predictions` with axes
`(sample,horizon,channel,height,width)`, without mosaic reconstruction.

## Evaluation: `evaluation`

| Field | Default | Effect |
| --- | --- | --- |
| `region` | `ainpp-amazon-basin` | Declared; does not crop the dataset or filter a region in the evaluator. |
| `checkpoint` | Empty string | Does not load weights; use the root `checkpoint` field. |
| `thresholds_mm_h` | `[0.1,1.0,5.0,10.0]` | Thresholds used in event-dependent calculations. |
| `lead_times_min` | `[10,20,30,40,50,60]` | Read, but not applied to labels: results use `T+1`, `T+2`, etc. Does not resample data. |
| `categorical` | `true` | Enables event-occurrence metrics. |
| `continuous` | `true` | Enables errors and association between continuous values. |
| `probabilistic` | `true` | Enables the implemented probabilistic branch; does not turn the model into an ensemble. |
| `object_based` | `true` | Enables spatial object/event metrics. |
| `sharpness` | `true` | Enables structure/sharpness metrics. |
| `consistency` | `true` | Enables distribution-consistency metrics. |
| `max_batches` | `null` | Not applied by the current loop. |
| `output_dir` | Absent; fallback `outputs/evaluation` | Add `+evaluation.output_dir=...`; destination for `evaluation_summary.csv` and optional Parquet output. |

For hourly data, record
`'evaluation.lead_times_min=[60,120,180,240,300,360]'` in the experiment
configuration, but interpret `T+1` as the first predicted frame: label conversion
to minutes has not been implemented. For a quick check, use
`+dataset.overrides.test.steps_per_epoch=8` (random sampling), not
`evaluation.max_batches`. Parquet requires `pyarrow` or `fastparquet`.

## Visualization: `visualization`

The `visualization/default.yaml` group is not in the root `defaults`.
`+visualization=default` includes it, but the CLI evaluator only uses
`visualization.output_dir`; it does not pass the style profile to the figure
generator.

| Field | Profile default | Usage |
| --- | --- | --- |
| `output_dir` | Absent; CLI uses `outputs/figures` | `+visualization.output_dir=...` sets the destination for evaluation figures. |
| `style.context` | `paper` | Seaborn context in `VisualizationRunner`/`set_style`. |
| `style.style` | `whitegrid` | Seaborn style. |
| `style.palette` | `deep` | Seaborn palette. |
| `style.font_family` | `sans-serif` | Matplotlib font family. |
| `style.dpi` | `300` | Figure resolution. |
| `maps.cmap` | `viridis` | Precipitation colormap in the runner. |
| `maps.diff_cmap` | `coolwarm` | Difference colormap in the runner. |
| `animation.fps` | `5` | Frames per second for runner animations. |

These style fields are for the separate API/runner, which expects `metrics.json`
and/or `sample_*.npz`. Do not assume the evaluation CSVs feed this runner.

## Hydra, paths, and environment

| Field/configuration | Project default | Usage |
| --- | --- | --- |
| `hydra.run.dir` | `outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}` | Hydra execution directory for a single run. |
| `hydra.sweep.dir` | `multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}` | Base directory for multirun. |
| `hydra.sweep.subdir` | `${hydra.job.num}` | Subdirectory per combination. |
| `hydra.job.chdir` | Not defined in the project | Avoid enabling without reviewing relative data/checkpoint paths. |
| `CUDA_VISIBLE_DEVICES` | External environment | Selects visible GPUs, for example `CUDA_VISIBLE_DEVICES=0 python main.py ...`. |
| `HYDRA_FULL_ERROR` | External environment | `HYDRA_FULL_ERROR=1` displays the full traceback. |

`hydra.run.dir` does not automatically redirect every artifact. Configure
`training.checkpoint.dir`, `inference.output_dir`, `evaluation.output_dir`, and
`visualization.output_dir` for the workflow. Hydra logs/configurations record
overrides, but do not guarantee scientific determinism.

There are no active CLI parameters for AMP, gradient accumulation, optimizer
selection, complete training resumption, or DDP initialization. Use the fields
actually consumed above; adding an arbitrary key with `+` does not implement
the corresponding functionality.
