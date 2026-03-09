#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data"
MODELS_DIR="$ROOT_DIR/models"
LFW_ARCHIVE="$DATA_DIR/lfw_funneled.tgz"
LFW_URL="https://ndownloader.figshare.com/files/5976015"
ARCFACE_ZIP="$MODELS_DIR/buffalo_l.zip"
ARCFACE_URL="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
ARCFACE_MODEL="$MODELS_DIR/w600k_r50.onnx"

mkdir -p "$DATA_DIR" "$MODELS_DIR" "$ROOT_DIR/artifacts" "$ROOT_DIR/experiments" "$ROOT_DIR/results"

if [ ! -d "$ROOT_DIR/.venv" ]; then
  python3 -m venv "$ROOT_DIR/.venv"
fi

source "$ROOT_DIR/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/requirements.txt"

if [ ! -f "$LFW_ARCHIVE" ]; then
  echo "Downloading LFW..."
  curl -L "$LFW_URL" -o "$LFW_ARCHIVE"
fi

if [ ! -d "$DATA_DIR/lfw_funneled" ]; then
  echo "Extracting LFW..."
  tar -xzf "$LFW_ARCHIVE" -C "$DATA_DIR"
fi

if [ ! -f "$ARCFACE_ZIP" ]; then
  echo "Downloading ArcFace ONNX package..."
  curl -L "$ARCFACE_URL" -o "$ARCFACE_ZIP"
fi

if [ ! -f "$ARCFACE_MODEL" ]; then
  echo "Extracting ArcFace model..."
  unzip -o "$ARCFACE_ZIP" w600k_r50.onnx -d "$MODELS_DIR" >/dev/null
fi

echo
echo "Bootstrap complete."
echo "Next:"
echo "  source \"$ROOT_DIR/.venv/bin/activate\""
echo "  streamlit run \"$ROOT_DIR/app/streamlit_app.py\""
