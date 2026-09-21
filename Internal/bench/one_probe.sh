#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SC=/tmp/claude-10003/-home-sam-benny-external-anubatechnologies-com/6de6c3a2-577b-4636-8e47-34c6a1dd5048/scratchpad
V="Wienerschnitzel_Sacramento_CA_95818__camA__2026_09_06_11_to_12_PDT.mkv"
DASHBOARD_PORT=8100 OPTICAL_FLOW_ENABLED=false KDS_MODE=none FRESH_START=1 \
  LOG_LEVEL=INFO LOG_METRICS_INTERVAL=5 VIDEO_SOURCE="$V" \
  .venv/bin/python -m src.main > "$SC/flowoff.log" 2>&1 &
p=$!; sleep 85; kill $p 2>/dev/null; sleep 4; kill -9 $p 2>/dev/null
