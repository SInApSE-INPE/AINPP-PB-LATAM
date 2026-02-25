# Installation

## Requirements
- Python 3.10+
- pip / virtualenv (recommended)

## Steps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,docs]
```

## Local Docs Preview
```bash
mkdocs serve
```
Then open the served URL (default: http://127.0.0.1:8000).

## Production Build
```bash
mkdocs build
```
The static site is emitted to `site/`.
