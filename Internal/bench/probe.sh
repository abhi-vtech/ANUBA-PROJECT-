#!/usr/bin/env bash
# Short pipeline probes to compare per-run settings. Writes a small table.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
V="Wienerschnitzel_Sacramento_CA_95818__camA__2026_09_06_11_to_12_PDT.mkv"
SC=/tmp/claude-10003/-home-sam-benny-external-anubatechnologies-com/6de6c3a2-577b-4636-8e47-34c6a1dd5048/scratchpad
DUR=${DUR:-80}
run_probe () {
  local label="$1"; shift
  : > "$SC/probe_cur.log"
  env "$@" KDS_MODE=none FRESH_START=1 LOG_LEVEL=INFO LOG_METRICS_INTERVAL=5 \
      VIDEO_SOURCE="$V" .venv/bin/python -m src.main > "$SC/probe_cur.log" 2>&1 &
  local p=$!
  sleep "$DUR"
  kill "$p" 2>/dev/null; sleep 4; kill -9 "$p" 2>/dev/null; wait "$p" 2>/dev/null
  # take the 3rd-from-last sample so we are past warm-up and before teardown
  local f l d
  f=$(grep -o '"fps": [0-9.]*' "$SC/probe_cur.log" | tail -3 | head -1 | grep -oE '[0-9.]+$')
  l=$(grep -o '"loop_ms": [0-9.]*' "$SC/probe_cur.log" | tail -3 | head -1 | grep -oE '[0-9.]+$')
  d=$(grep -o '"detect_ms": [0-9.]*' "$SC/probe_cur.log" | tail -3 | head -1 | grep -oE '[0-9.]+$')
  printf "  %-34s fps=%-7s loop=%-8s detect=%s\n" "$label" "${f:-?}" "${l:-?}" "${d:-?}"
}
echo "=== pipeline probes (${DUR}s each, bytetrack active) ==="
run_probe "baseline (flow on, imgsz 640)"  NOOP=1
run_probe "flow OFF"                       OPTICAL_FLOW_ENABLED=false
run_probe "flow OFF + imgsz 512"           OPTICAL_FLOW_ENABLED=false IMGSZ=512
echo "=== done ==="
