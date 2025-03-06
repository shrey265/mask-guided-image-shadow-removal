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

istd_complete() {
  local d n
  for d in train/train_A train/train_B train/train_C test/test_A test/test_B test/test_C; do
    n=$(find "$DATA_DIR/ISTD_Dataset/$d" -type f 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] || return 1
  done
}

if istd_complete; then
  echo "ISTD already present at $DATA_DIR/ISTD_Dataset"
  exit 0
elif [ -d "$DATA_DIR/ISTD_Dataset" ]; then
  echo "Removing incomplete $DATA_DIR/ISTD_Dataset left by an earlier failed extraction"
  rm -rf "$DATA_DIR/ISTD_Dataset"
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
# The ISTD archive is RAR5. The old p7zip `7z` (Ubuntu's p7zip-full 16.02) cannot decode
# RAR5 and fails with "ERROR: Unsupported Method". So we try, in order:
#   unrar  ->  7zz (modern 7-Zip)  ->  official 7-Zip binary (auto-downloaded)  ->  bsdtar  ->  old 7z
# and accept a method only if it exits cleanly AND produced the image files.
# Everything is extracted into a private temp folder first, so a failed attempt can be
# wiped without touching anything else in DATA_DIR.
EXTRACT_DIR="$DATA_DIR/.istd_extract"
ERR_LOG="$DATA_DIR/.extract_err"
rm -rf "$EXTRACT_DIR"; mkdir -p "$EXTRACT_DIR"

extracted_ok() {
  local n
  n=$(find "$EXTRACT_DIR" -path "*train_C*" -type f -name "*.png" 2>/dev/null | wc -l)
  [ "$n" -gt 0 ]
}

clean_partial() {
  rm -rf "$EXTRACT_DIR"; mkdir -p "$EXTRACT_DIR"
}

try_extract() {  # $1 = label, rest = command
  local label="$1"; shift
  echo "  trying $label ..."
  if "$@" >/dev/null 2>"$ERR_LOG" && extracted_ok; then
    echo "  extracted with $label"; return 0
  fi
  echo "  $label failed: $(grep -m1 -iE 'error|unsupported|cannot' "$ERR_LOG" || true)"
  clean_partial
  return 1
}

get_7zz() {  # official 7-Zip for Linux (supports RAR5), no root needed
  local dir="$DATA_DIR/.7zz"
  if [ ! -x "$dir/7zz" ]; then
    mkdir -p "$dir"
    local arch="x64"; [ "$(uname -m)" = "aarch64" ] && arch="arm64"
    curl -fsSL "https://www.7-zip.org/a/7z2408-linux-${arch}.tar.xz" -o "$dir/7z.tar.xz" &&
      tar -xJf "$dir/7z.tar.xz" -C "$dir" 7zz || return 1
  fi
  echo "$dir/7zz"
}

case "$ARCHIVE" in
  *.zip) unzip -q -o "$ARCHIVE" -d "$EXTRACT_DIR" ;;
  *)
    ok=1
    if [ $ok -ne 0 ] && command -v unrar >/dev/null 2>&1; then
      try_extract "unrar" unrar x -o+ -idq "$ARCHIVE" "$EXTRACT_DIR/" && ok=0
    fi
    if [ $ok -ne 0 ] && command -v 7zz >/dev/null 2>&1; then
      try_extract "7zz" 7zz x -y -o"$EXTRACT_DIR" "$ARCHIVE" && ok=0
    fi
    if [ $ok -ne 0 ] && SEVENZZ=$(get_7zz); then
      try_extract "7-Zip 24.08 (official binary)" "$SEVENZZ" x -y -o"$EXTRACT_DIR" "$ARCHIVE" && ok=0
    fi
    if [ $ok -ne 0 ] && command -v bsdtar >/dev/null 2>&1; then
      try_extract "bsdtar" bsdtar -xf "$ARCHIVE" -C "$EXTRACT_DIR" && ok=0
    fi
    if [ $ok -ne 0 ] && command -v 7z >/dev/null 2>&1; then
      try_extract "7z (p7zip)" 7z x -y -o"$EXTRACT_DIR" "$ARCHIVE" && ok=0
    fi
    rm -f "$ERR_LOG"
    if [ $ok -ne 0 ]; then
      echo "Could not extract $ARCHIVE (it is RAR5)."
      echo "Install a RAR5-capable tool, e.g.:  sudo apt install unrar   (or download 7zz from https://www.7-zip.org)"
      exit 1
    fi ;;
esac

# Move the extracted dataset root (whatever it was called inside the archive) into place
FOUND=$(find "$EXTRACT_DIR" -type d -name train_A -path "*train*" | head -n 1 || true)
if [ -z "$FOUND" ]; then
  echo "Extraction produced no train/train_A folder - unexpected archive layout:"
  find "$EXTRACT_DIR" -maxdepth 3 -type d | head -n 20
  exit 1
fi
ROOT=$(dirname "$(dirname "$FOUND")")
rm -rf "$DATA_DIR/ISTD_Dataset"
mv "$ROOT" "$DATA_DIR/ISTD_Dataset"
rm -rf "$EXTRACT_DIR"

for d in train/train_A train/train_B train/train_C test/test_A test/test_B test/test_C; do
  n=$(find "$DATA_DIR/ISTD_Dataset/$d" -type f 2>/dev/null | wc -l)
  echo "  $d : $n files"
done
if ! istd_complete; then
  echo "Some ISTD folders are empty - extraction is incomplete"; exit 1
fi
echo "ISTD ready at $DATA_DIR/ISTD_Dataset"
