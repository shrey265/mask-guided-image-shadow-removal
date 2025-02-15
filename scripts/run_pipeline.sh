#!/usr/bin/env bash
# End-to-end: verify env -> download ISTD -> train (with val) -> test (GT + predicted masks).
#
#   bash scripts/run_pipeline.sh 0                 # use cuda:0
#   bash scripts/run_pipeline.sh 2 train.epochs=100 train.batch_size=16
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-0}"
shift || true

python check_env.py --gpu "$GPU"
bash scripts/download_istd.sh
python train.py --config configs/istd.yaml --gpu "$GPU" --opts "$@"
echo "Results are in the newest folder under outputs/ (results/test_gt, results/test_pred)."
