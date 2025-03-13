#!/usr/bin/env bash
# End-to-end: verify env -> download ISTD -> train (with val) -> test (GT + predicted masks).
#
#   bash scripts/run_pipeline.sh --gpu 0
#   bash scripts/run_pipeline.sh --gpu 2 --data-dir /mnt/shadow
#   bash scripts/run_pipeline.sh --gpu 2 --data-dir /mnt/shadow train.epochs=100 train.batch_size=16
#
# With --data-dir everything (dataset, runs, checkpoints, results, caches) goes under that folder.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=0
DATA_DIR="${SHADOW_DATA_DIR:-}"
OPTS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    *) OPTS+=("$1"); shift ;;
  esac
done

DD_ARGS=()
if [ -n "$DATA_DIR" ]; then
  DATA_DIR="$(mkdir -p "$DATA_DIR" && cd "$DATA_DIR" && pwd)"
  DD_ARGS=(--data-dir "$DATA_DIR")
fi

python check_env.py --gpu "$GPU" "${DD_ARGS[@]}"
bash scripts/download_istd.sh "${DD_ARGS[@]}"
python train.py --config configs/istd.yaml --gpu "$GPU" "${DD_ARGS[@]}" --opts "${OPTS[@]}"
echo "Results are in the newest run under ${DATA_DIR:-.}/outputs/ (results/test_gt, results/test_pred)."
