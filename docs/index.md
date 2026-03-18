# AINPP Precipitation Benchmark

AINPP-PB-LATAM is a benchmark-oriented Python library for precipitation nowcasting in Latin America. The project combines deep learning model experimentation, reproducible scientific evaluation, and documentation that stays close to the source code.

## What This Project Covers

- standardized loading of precipitation datasets stored as `.zarr`,
- Hydra-based experiment configuration,
- training workflows for direct and autoregressive forecasting,
- evaluation pipelines for benchmark metrics,
- visualization utilities for scientific analysis and reporting,
- support for local and HPC-oriented execution.

## Benchmark Assumptions

The reference benchmark follows a fixed operational setup:

- train split: `2018-2022`
- validation split: `2023`
- test split: `2024`
- input window: `12` hourly steps from `gsmap_nrt`
- forecast horizon: `6` hourly steps targeting `gsmap_mvk`
- grid shape: `880 x 970`

These assumptions are represented in the configuration layer and can be extended for new experiments.

## Documentation Map

- `Installation`: environment creation and dependency installation with `uv`
- `Architecture`: project structure, configuration model, and scientific pipeline
- `Training`: model-by-model training guide, Hydra overrides, dataset tuning, and loss selection
- `Usage`: how to run train, evaluate, and infer tasks through Hydra
- `Publishing`: how the documentation site is deployed to GitHub Pages
- `API`: package reference generated with `mkdocstrings`

## Local Preview

To work on the docs locally:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[docs]
uv run mkdocs serve
```
