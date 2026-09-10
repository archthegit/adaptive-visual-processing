#!/usr/bin/env bash
set -euo pipefail

VILA_REPO_URL="${VILA_REPO_URL:-https://github.com/NVlabs/VILA.git}"
VILA_COMMIT="${VILA_COMMIT:-0f1426e8da9181e6e6653e10bc15f62d515fa2f6}"
VILA_SRC_DIR="${VILA_SRC_DIR:-external/VILA}"
VENV_DIR="${VENV_DIR:-.venv-vila}"

python -m venv "${VENV_DIR}"
. "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-vila.txt

if [ ! -d "${VILA_SRC_DIR}/.git" ]; then
  mkdir -p "$(dirname "${VILA_SRC_DIR}")"
  git clone "${VILA_REPO_URL}" "${VILA_SRC_DIR}"
fi

git -C "${VILA_SRC_DIR}" fetch --tags origin "${VILA_COMMIT}"
git -C "${VILA_SRC_DIR}" checkout --detach "${VILA_COMMIT}"

if [ -f "${VILA_SRC_DIR}/requirements.txt" ]; then
  python -m pip install -r "${VILA_SRC_DIR}/requirements.txt"
fi

python -m pip install -e "${VILA_SRC_DIR}"

cat <<'MSG'
VILA environment installed in .venv-vila.

Activate it with:
  . .venv-vila/bin/activate

This environment is intentionally separate from the Qwen Experiment 1 environment.
Official NVLabs/VILA has been cloned under external/VILA and checked out at:
  0f1426e8da9181e6e6653e10bc15f62d515fa2f6

The Experiment 1 runner uses VILA's official model builder, tokenizer,
image processor and conversation template. It bypasses VILA MP4 sampling by
passing already-decoded ordered RGB frames as image inputs.
MSG
