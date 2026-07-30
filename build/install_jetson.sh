#!/usr/bin/env bash
# Native Jetson setup (JetPack 6.2 / CUDA 12.6 / Python 3.10).
#
# torch cannot live in uv.lock (it must come from the Jetson-only wheel index),
# so this script does: uv sync  ->  override torch/torchvision  ->  pin numpy<2.
#
# NOTE: for GPU you must be in the `video` group:
#   sudo usermod -aG video "$USER"   # then log out/in (or: newgrp video)
set -euo pipefail
cd "$(dirname "$0")/.."

JETSON_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126"
TORCH_VER="${TORCH_VER:-2.8.0}"
TV_VER="${TV_VER:-0.23.0}"

echo ">> uv sync (Python 3.10, no dev)"
uv sync --python 3.10 --no-dev

echo ">> install Jetson CUDA torch ${TORCH_VER} / torchvision ${TV_VER}"
uv pip install "torch==${TORCH_VER}" "torchvision==${TV_VER}" \
    --no-deps --reinstall --index-url "${JETSON_INDEX}"

echo ">> pin numpy<2 (Jetson torch is built against NumPy 1.x)"
uv pip install "numpy<2"

echo ">> verify"
uv run python -c "import torch; print('torch', torch.__version__, '| built cuda', torch.version.cuda, '| cuda avail', torch.cuda.is_available())"
echo ">> if 'cuda avail' is False: ensure you are in the 'video' group and re-login."
