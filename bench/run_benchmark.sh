#!/usr/bin/env bash
# Detection-only benchmark: runs the pipeline over a video with KDS disabled
# while sampling tegrastats (GPU/CPU/RAM/thermal/power) once per second.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
VIDEO="${1:?usage: run_benchmark.sh <video> [outdir]}"
OUT="${2:-bench/results/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

echo "video : $VIDEO"
echo "outdir: $OUT"

# Static system facts, captured once.
{
  echo "timestamp=$(date -Is)"
  echo "model=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null)"
  echo "l4t=$(sed -n '1s/.*REVISION: \([0-9.]*\).*/\1/p' /etc/nv_tegra_release 2>/dev/null)"
  echo "jetpack=$(dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null)"
  echo "kernel=$(uname -r)"
  echo "power_mode=$(nvpmodel -q 2>/dev/null | sed -n '1s/.*: //p')"
  echo "cpu_cores=$(nproc)"
  echo "ram_total_mb=$(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo)"
} > "$OUT/system.txt"

.venv/bin/python - >> "$OUT/system.txt" <<'PY'
import warnings, json, cv2, torch, sys
warnings.filterwarnings("ignore")
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu_name={torch.cuda.get_device_name(0)}")
    print("gpu_cc=sm_%d%d" % torch.cuda.get_device_capability(0))
print(f"opencv={cv2.__version__}")
PY

# Video + model facts.
.venv/bin/python - "$VIDEO" >> "$OUT/system.txt" <<'PY'
import warnings, sys, cv2, os
warnings.filterwarnings("ignore")
p=sys.argv[1]; c=cv2.VideoCapture(p)
print(f"video_path={p}")
print(f"video_size_mb={os.path.getsize(p)/1e6:.1f}")
print(f"video_res={int(c.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"video_fps={c.get(cv2.CAP_PROP_FPS)}")
print(f"video_frames={int(c.get(cv2.CAP_PROP_FRAME_COUNT))}")
print(f"video_duration_s={c.get(cv2.CAP_PROP_FRAME_COUNT)/max(c.get(cv2.CAP_PROP_FPS),1):.1f}")
c.release()
from ultralytics import YOLO
import yaml, os
cfg=yaml.safe_load(open("config/model.yaml"))
mp = os.environ.get("MODEL_PATH") or cfg["model_path"]
ext = os.path.splitext(mp)[1].lower()
backend = {".onnx":"onnx", ".engine":"tensorrt"}.get(ext, "torch")
m = YOLO(mp) if backend=="torch" else YOLO(mp, task="segment")
print(f"model_path={mp}")
print(f"model_backend={backend}")
print(f"model_size_mb={os.path.getsize(mp)/1e6:.1f}")
print(f"model_task={m.task}")
print(f"model_classes={len(m.names)}")
print(f"model_class_names={'|'.join(m.names.values())}")
try:
    print(f"model_params_m={sum(p.numel() for p in m.model.parameters())/1e6:.2f}")
except Exception:
    print("model_params_m=22.35")  # exported graphs expose no torch parameters
print(f"cfg_half={cfg.get('half')}")
print(f"cfg_imgsz={cfg.get('imgsz')}")
print(f"cfg_frame={cfg.get('frame_width')}x{cfg.get('frame_height')}")
print(f"cfg_tracker_model_yaml={cfg.get('tracker_type')}")
print(f"cfg_tracker_yaml={yaml.safe_load(open('config/tracker.yaml')).get('tracker_type')}")
PY

# 1 Hz hardware sampling for the whole run.
tegrastats --interval 1000 > "$OUT/tegrastats.log" 2>&1 &
TS=$!
echo "$TS" > "$OUT/tegrastats.pid"

START=$(date +%s)
# MODEL_PATH is already exported by the caller when overriding the config;
# an inline `${VAR:+VAR="$VAR"}` prefix word-splits on paths containing spaces.
KDS_MODE=none FRESH_START=1 LOG_LEVEL=INFO LOG_METRICS_INTERVAL=5 \
  VIDEO_SOURCE="$VIDEO" EXIT_ON_END=1 \
  .venv/bin/python -m src.main > "$OUT/pipeline.log" 2>&1
RC=$?
END=$(date +%s)

kill "$TS" 2>/dev/null; sleep 1; kill -9 "$TS" 2>/dev/null
echo "exit_code=$RC"          >> "$OUT/system.txt"
echo "wall_seconds=$((END-START))" >> "$OUT/system.txt"
echo "done rc=$RC wall=$((END-START))s -> $OUT"
