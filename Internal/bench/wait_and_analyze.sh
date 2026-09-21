#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="bench/results/camA_full"
# Wait for the pipeline process to disappear (matched via ps, not our own args).
while ps -eo args | grep -q "[.]venv/bin/python -m src.main"; do sleep 60; done
sleep 5
.venv/bin/python bench/analyze.py "$OUT" > "$OUT/summary.txt" 2>&1
echo "analysis complete"
grep -E "^exit_code|^wall_seconds" "$OUT/system.txt" 2>/dev/null
