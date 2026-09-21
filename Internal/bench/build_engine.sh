#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SC=/tmp/claude-10003/-home-sam-benny-external-anubatechnologies-com/6de6c3a2-577b-4636-8e47-34c6a1dd5048/scratchpad
START=$(date +%s)
.venv/bin/python - > "$SC/engine_build.log" 2>&1 <<'PY'
import warnings; warnings.filterwarnings("ignore")
from ultralytics import YOLO
m = YOLO("rf_trained/weights (1).pt")
out = m.export(format="engine", imgsz=640, half=True, dynamic=False, simplify=True, workspace=4)
print("ENGINE:", out)
PY
echo "rc=$? elapsed=$(( $(date +%s)-START ))s" >> "$SC/engine_build.log"
