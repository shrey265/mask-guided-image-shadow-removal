#!/usr/bin/env bash
# Create an isolated virtualenv with a matched torch/torchvision pair.
#
# Usage:
#   bash setup_env.sh                 # CUDA 12.1 wheels (default)
#   bash setup_env.sh cu124           # CUDA 12.4 wheels
#   bash setup_env.sh cu118           # CUDA 11.8 wheels (older drivers)
#   bash setup_env.sh cpu             # CPU only
#   VENV_DIR=/path/to/env bash setup_env.sh cu121
#   bash setup_env.sh cu121 --data-dir /mnt/shadow   # pip's download cache (~2-3 GB of torch
#                                                    # wheels) goes to /mnt/shadow/cache/pip
set -euo pipefail

CUDA_TAG="cu121"
DATA_DIR="${SHADOW_DATA_DIR:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) DATA_DIR="${2:?--data-dir needs a path}"; shift 2 ;;
    --data-dir=*) DATA_DIR="${1#*=}"; shift ;;
    *) CUDA_TAG="$1"; shift ;;
  esac
done
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -n "$DATA_DIR" ]; then
  mkdir -p "$DATA_DIR/cache/pip"
  export PIP_CACHE_DIR="$(cd "$DATA_DIR" && pwd)/cache/pip"
  echo "pip cache -> $PIP_CACHE_DIR"
fi

case "$CUDA_TAG" in
  cu118|cu121|cu124|cpu) ;;
  *) echo "Unsupported CUDA tag '$CUDA_TAG' (use cu118 | cu121 | cu124 | cpu)"; exit 1 ;;
esac

PY_VER=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MINOR=$("$PYTHON_BIN" -c 'import sys; print(sys.version_info.minor)')
if [ "$PY_MINOR" -lt 9 ] || [ "$PY_MINOR" -gt 12 ]; then
  echo "Python $PY_VER detected; torch 2.5.1 wheels need Python 3.9-3.12."
  echo "Set PYTHON_BIN to a supported interpreter, e.g. PYTHON_BIN=python3.11 bash setup_env.sh"
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "== NVIDIA driver =="
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
fi

echo "== Creating virtualenv at $VENV_DIR (Python $PY_VER) =="
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "== Installing torch/torchvision ($CUDA_TAG) =="
# Uninstall any stray copies first so a mismatched pair can never survive.
pip uninstall -y torch torchvision >/dev/null 2>&1 || true
pip install -r requirements-torch.txt --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "== Installing remaining dependencies (torch pinned via constraints) =="
pip install -r requirements.txt -c constraints.txt

echo "== Verifying environment =="
if [ -n "$DATA_DIR" ]; then python check_env.py --data-dir "$DATA_DIR"; else python check_env.py; fi

echo
echo "Done. Activate with:  source $VENV_DIR/bin/activate"
