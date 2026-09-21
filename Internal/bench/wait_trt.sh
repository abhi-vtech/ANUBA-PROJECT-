#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="bench/results/camA_trt"
while ps -eo args | grep -q "[.]venv/bin/python -m src.main"; do sleep 60; done
sleep 5
.venv/bin/python bench/analyze.py "$OUT" > "$OUT/summary.txt" 2>&1
echo "analysis complete"
grep -E "^exit_code|^wall_seconds|^model_backend" "$OUT/system.txt" 2>/dev/null
