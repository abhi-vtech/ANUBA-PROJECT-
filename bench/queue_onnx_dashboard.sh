#!/usr/bin/env bash
# Start the recorded ONNX dashboard run once the Jetson is free, so it neither
# slows down nor is slowed by other GPU work:
#   * no other user's KDS GPU job ("python -m kds") is running
#   * no order-accuracy pipeline is running
#   * dashboard port 8000 is free
# The Jetson must look free for 3 checks in a row, a minute apart.
#
#   nohup setsid ./bench/queue_onnx_dashboard.sh >/dev/null 2>&1 &
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec >>bench/queue_onnx_dashboard.log 2>&1
echo "[$(date -Is)] queued: waiting for the Jetson to be free"

busy() {
  pgrep -f "[p]ython -m kds" >/dev/null && return 0
  pgrep -f "[.]venv/bin/python -m src.main" >/dev/null && return 0
  ss -tln 2>/dev/null | grep -qE ":8000[[:space:]]" && return 0
  return 1
}

free_checks=0
last_note=0
while [ "$free_checks" -lt 3 ]; do
  if busy; then
    free_checks=0
    now=$(date +%s)
    if [ $((now - last_note)) -ge 1800 ]; then
      echo "[$(date -Is)] still busy: $(pgrep -fc '[p]ython -m kds') KDS job process(es), pipeline running: $(pgrep -fc '[.]venv/bin/python -m src.main')"
      last_note=$now
    fi
  else
    free_checks=$((free_checks + 1))
  fi
  sleep 60
done

echo "[$(date -Is)] the Jetson has been free for 3 minutes; launching the run"
./bench/run_onnx_dashboard.sh
echo "[$(date -Is)] queue finished"
touch bench/queue_onnx_dashboard.done
