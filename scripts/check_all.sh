#!/usr/bin/env bash
set -euo pipefail

python -m pytest --cov=src/ainpp_pb_latam --cov-report=term-missing --cov-report=json:coverage.json --cov-branch "$@"
python scripts/enforce_coverage.py coverage.json 10
black --check .
isort --check-only .
mypy src
