#!/usr/bin/env bash
# Jetson launcher for the Order-Accuracy pipeline.
#
#   ./run.sh                          # source from config/model.yaml
#   ./run.sh videos/kds_3.mp4         # override the video source
#   ./run.sh --cpu videos/clip.mp4    # force CPU (debugging)
#   ./run.sh --check                  # environment doctor, runs nothing
#   ./run.sh --exit-on-end clip.mp4   # quit when the video ends
#   ./run.sh --record clip.mp4        # also save the detection video to output/recordings/
#
# Any pipeline env var still works, e.g.:
#   LOG_LEVEL=INFO CONFIDENCE_THRESHOLD=0.4 ./run.sh clip.mp4
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; DIM=$'\033[2m'; RST=$'\033[0m'
die() { echo "${RED}error:${RST} $*" >&2; exit 1; }

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || die "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"

FORCE_CPU=0; CHECK_ONLY=0; SRC=""
while [ $# -gt 0 ]; do
  case "$1" in
    --cpu)         FORCE_CPU=1 ;;
    --check)       CHECK_ONLY=1 ;;
    --exit-on-end) export EXIT_ON_END=1 ;;
    --record)      export RECORD_VIDEO="${RECORD_VIDEO:-1}" ;;
    -h|--help)     sed -n '2,/^set -euo/{/^set -euo/!p}' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)            die "unknown flag: $1" ;;
    *)             SRC="$1" ;;
  esac
  shift
done

# .venv is gitignored; create it on first run rather than failing.
[ -x .venv/bin/python ] || { echo "${YLW}==>${RST} no .venv, running uv sync (several minutes)"; uv sync; }

# output/ and videos/ are gitignored, so a fresh clone lacks both. main.py
# writes output/yolo_detections.log at startup and dies without the directory.
mkdir -p output videos

echo "${DIM}--- environment ---${RST}"
.venv/bin/python - <<'PY'
import warnings; warnings.filterwarnings("ignore")
import sys, torch
v = sys.version_info
print(f"  python      {v.major}.{v.minor}.{v.micro}", end="")
print("  OK" if (3,10) <= (v.major,v.minor) < (3,13) else "  OUT OF RANGE (needs >=3.10,<3.13)")
print(f"  torch       {torch.__version__}")
if torch.cuda.is_available():
    print(f"  gpu         {torch.cuda.get_device_name(0)} sm_%d%d" % torch.cuda.get_device_capability(0))
else:
    print("  gpu         NONE - will run on CPU (~20x slower)")
PY

if [ -r /sys/devices/virtual/thermal/thermal_zone0/temp ]; then
  printf "  soc temp    %s C\n" "$(( $(cat /sys/devices/virtual/thermal/thermal_zone0/temp) / 1000 ))"
fi
if command -v nvpmodel >/dev/null; then
  MODE="$(nvpmodel -q 2>/dev/null | sed -n '1s/.*: //p' || true)"
  [ -n "$MODE" ] && { printf "  power mode  %s" "$MODE"
    [ "$MODE" = "MAXN_SUPER" ] || printf "   ${YLW}(sudo nvpmodel -m 0 for max perf)${RST}"; echo; }
fi

if [ "$FORCE_CPU" = "1" ]; then
  export CUDA_VISIBLE_DEVICES=""
  echo "  ${YLW}--cpu given: CUDA disabled for this run${RST}"
fi

if [ -n "$SRC" ]; then
  [ -f "$SRC" ] || die "video not found: $SRC"
  export VIDEO_SOURCE="$SRC"
fi

# Validate whichever source we ended up with, so a missing file fails here with
# a clear message instead of deep inside the capture thread.
.venv/bin/python - <<'PY' || exit 1
import os, sys, yaml, pathlib
src = os.environ.get("VIDEO_SOURCE") or yaml.safe_load(open("config/model.yaml"))["source"]
if str(src).isdigit():
    print(f"  source      camera index {src}")
elif str(src).startswith(("rtsp://", "http://", "https://")):
    print(f"  source      stream {src}")
elif pathlib.Path(src).is_file():
    print(f"  source      {src}")
else:
    sys.stderr.write(f"\n\033[31merror:\033[0m video source not found: {src}\n"
                     "       put the file in videos/ and pass it:  ./run.sh videos/<name>.mp4\n"
                     "       or edit 'source:' in config/model.yaml\n")
    sys.exit(1)
PY

[ "$CHECK_ONLY" = "1" ] && { echo "${GRN}==>${RST} check only, exiting."; exit 0; }

echo "${DIM}-------------------${RST}"
echo "${GRN}==>${RST} dashboard will be at http://localhost:8000"
[ -n "${RECORD_VIDEO:-}" ] && echo "${GRN}==>${RST} recording to output/recordings/ "\
  "(convert for sharing: ./bench/transcode.sh <file>.mkv)"
exec .venv/bin/python -m src.main
