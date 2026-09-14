#!/usr/bin/env bash
# Live progress bar for a running benchmark. Updates in place until the run ends.
#   ./bench/progress.sh            # watches bench/results/camA_full
#   ./bench/progress.sh <outdir>
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="${1:-bench/results/camA_full}"
LOG="$OUT/pipeline.log"
TOTAL=$(sed -n 's/^video_frames=//p' "$OUT/system.txt" 2>/dev/null); TOTAL=${TOTAL:-107811}
W=40
trap 'printf "\n"; exit 0' INT
while :; do
  LINE=$(grep -o '{"event": "metrics".*' "$LOG" 2>/dev/null | tail -1)
  if [ -n "$LINE" ]; then
    read -r DONE FPS ELAP DET < <(printf '%s' "$LINE" | python3 -c "
import sys,json; e=json.loads(sys.stdin.read())
print(e['frame_count'], e['fps'], e['elapsed_s'], e.get('detect_ms',0))" 2>/dev/null)
    if [ -n "$DONE" ]; then
      PCT=$(awk -v d="$DONE" -v t="$TOTAL" 'BEGIN{printf "%.1f", 100*d/t}')
      FILL=$(awk -v p="$PCT" -v w="$W" 'BEGIN{printf "%d", w*p/100}')
      ETA=$(awk -v d="$DONE" -v t="$TOTAL" -v f="$FPS" 'BEGIN{if(f>0) printf "%dm", (t-d)/f/60; else printf "?"}')
      BAR=$(printf "%${FILL}s" | tr ' ' '#')$(printf "%$((W-FILL))s" | tr ' ' '.')
      TJ=$(awk '/tj@/{}' /dev/null; grep -o 'tj@[0-9.]*C' "$OUT/tegrastats.log" 2>/dev/null | tail -1)
      printf "\r\033[K [%s] %5s%%  %7s/%s  %5s fps  det %sms  eta %-5s %s" \
             "$BAR" "$PCT" "$DONE" "$TOTAL" "$FPS" "$DET" "$ETA" "${TJ:-}"
    fi
  fi
  # stop when the pipeline process is gone
  ps -eo args | grep -q "[.]venv/bin/python -m src.main" || { printf "\n run finished\n"; break; }
  sleep 2
done
