# Installation

## Requirements

- Python `3.10+`
- `uv`
- CUDA-capable environment for practical model training

This project should be installed in editable mode. Avoid import workarounds such as `sys.path.append(...)`.

## Steps

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[dev,docs]
```

## Optional Dependency Groups

- `.[dev]`: formatting, typing, tests, and coverage
- `.[docs]`: MkDocs and API reference generation
- `.[dl]`: deep learning stack
- `.[verification]`: scientific verification and notebook tooling
- `.[all]`: convenience bundle for the main scientific extras

## Sanity Checks

Verify that the package and the CLI entry point can be imported:

```bash
python main.py --help
```

## Local Docs Preview

```bash
uv run mkdocs serve
```

Then open `http://127.0.0.1:8000`.

## Production Build

```bash
uv run mkdocs build --strict
```

The static site is emitted to `site/`.
