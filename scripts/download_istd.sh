#!/usr/bin/env bash
# Download and extract the ISTD dataset (Wang et al., CVPR 2018: ST-CGAN).
#   1,330 train / 540 test triplets: shadow image (A), shadow mask (B), shadow-free image (C)
#
# Usage:
#   bash scripts/download_istd.sh                 # -> data/ISTD_Dataset
#   DATA_DIR=/mnt/datasets bash scripts/download_istd.sh
#   ISTD_GDRIVE_ID=<id> bash scripts/download_istd.sh   # if the default link moves
#   bash scripts/download_istd.sh /path/to/ISTD_Dataset.rar   # use an archive you already have
#
# Official source: https://github.com/DeepInsight-PCALab/ST-CGAN (Google Drive link in README).
# If the automatic download fails (Drive quota), download the archive manually from
# that page and pass its path as the first argument.
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
GDRIVE_ID="${ISTD_GDRIVE_ID:-1I0qw-65KBA6np8vIZzO6oeiOvcDBttAY}"
mkdir -p "$DATA_DIR"

if [ -d "$DATA_DIR/ISTD_Dataset/train/train_A" ]; then
  echo "ISTD already present at $DATA_DIR/ISTD_Dataset"
  exit 0
fi

ARCHIVE="${1:-}"
if [ -z "$ARCHIVE" ]; then
  ARCHIVE="$DATA_DIR/ISTD_Dataset.rar"
  if [ ! -f "$ARCHIVE" ]; then
    echo "Downloading ISTD from Google Drive (id=$GDRIVE_ID) ..."
    if ! python -m gdown "$GDRIVE_ID" -O "$ARCHIVE"; then
      echo "Automatic download failed. Download ISTD manually from"
      echo "  https://github.com/DeepInsight-PCALab/ST-CGAN"
      echo "then run: bash scripts/download_istd.sh /path/to/ISTD_Dataset.rar"
      exit 1
    fi
  fi
fi

echo "Extracting $ARCHIVE -> $DATA_DIR"
case "$ARCHIVE" in
  *.zip) unzip -q -o "$ARCHIVE" -d "$DATA_DIR" ;;
  *)
    if command -v 7z >/dev/null 2>&1; then 7z x -y -o"$DATA_DIR" "$ARCHIVE" >/dev/null
    elif command -v unrar >/dev/null 2>&1; then unrar x -o+ "$ARCHIVE" "$DATA_DIR/" >/dev/null
    elif command -v bsdtar >/dev/null 2>&1; then bsdtar -xf "$ARCHIVE" -C "$DATA_DIR"
    else
      echo "Need 7z, unrar or bsdtar to extract .rar (e.g. sudo apt install p7zip-full)"; exit 1
    fi ;;
esac

# Normalise folder name (some mirrors extract as 'ISTD' or nested folders)
if [ ! -d "$DATA_DIR/ISTD_Dataset/train/train_A" ]; then
  FOUND=$(find "$DATA_DIR" -type d -name train_A -path "*train*" | head -n 1 || true)
  if [ -n "$FOUND" ]; then
    ROOT=$(dirname "$(dirname "$FOUND")")
    if [ "$ROOT" != "$DATA_DIR/ISTD_Dataset" ]; then
      mv "$ROOT" "$DATA_DIR/ISTD_Dataset"
    fi
  fi
fi

for d in train/train_A train/train_B train/train_C test/test_A test/test_B test/test_C; do
  n=$(find "$DATA_DIR/ISTD_Dataset/$d" -type f 2>/dev/null | wc -l)
  echo "  $d : $n files"
done
echo "ISTD ready at $DATA_DIR/ISTD_Dataset"
