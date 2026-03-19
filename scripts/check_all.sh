#!/usr/bin/env bash
set -euo pipefail

python -m pytest "$@"
python scripts/enforce_coverage.py coverage.json 10
python -m isort --check-only .
python -m black --check .