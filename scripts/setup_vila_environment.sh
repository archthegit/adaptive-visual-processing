#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv-vila
. .venv-vila/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-vila.txt

cat <<'MSG'
VILA environment installed in .venv-vila.

Activate it with:
  . .venv-vila/bin/activate

This environment is intentionally separate from the Qwen Experiment 1 environment.
The real checkpoint path still must expose an exact prepare_experiment1_inputs mapping;
the runner will fail loudly if VILA truncates, resamples, duplicates, or reorders frames.
MSG
