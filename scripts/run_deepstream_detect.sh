#!/usr/bin/env bash
# Run the detection model under DeepStream and write a detections JSONL the
# host pipeline can replay.  DeepStream is not installed natively on this
# Jetson, so this runs in the container; see deepstream_test/ds_detect_dump.py
# for why inference and analysis are split.
#
#   scripts/run_deepstream_detect.sh <video> <out.jsonl> [from_s] [to_s]
set -euo pipefail

VIDEO=${1:?usage: run_deepstream_detect.sh <video> <out.jsonl> [from_s] [to_s]}
OUT=${2:?usage: run_deepstream_detect.sh <video> <out.jsonl> [from_s] [to_s]}
FROM=${3:-0}
TO=${4:-0}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${DS_IMAGE:-oad/deepstream:9.1}

mkdir -p "$(dirname "$ROOT/$OUT")"

# The nvidia runtime injects both nvgpu and openrm driver libraries and its own
# ld.so.conf makes the loader pick openrm first -- on which cuInit returns 100,
# because the Orin runs nvgpu.  Putting nvgpu first is what makes CUDA work.
exec docker run --rm --runtime nvidia \
  -v "$ROOT":/work -w /work \
  -e LD_LIBRARY_PATH=/opt/nvidia/l4t-gpu-libs/nvgpu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda-13.2/lib64:/opt/nvidia/deepstream/deepstream-9.1/lib \
  -e DS_VIDEO="/work/${VIDEO#/work/}" \
  -e DS_OUT="/work/${OUT#/work/}" \
  -e DS_FROM="$FROM" \
  -e DS_TO="$TO" \
  --entrypoint /bin/bash "$IMAGE" \
  -c 'python3 -X faulthandler /work/deepstream_test/ds_detect_dump.py'
