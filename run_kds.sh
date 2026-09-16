#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run the Order-Accuracy pipeline with the kds-ocr KDS reader.
#
#   ./run_kds.sh                      the pair set below
#   ./run_kds.sh --live               the RTSP cameras instead
#   ./run_kds.sh --full               the whole recording, not just RUN_FOR_S
#
# Edit the two SOURCE lines below to change which pair is processed.
# Stop the run with Ctrl+C -- never kill -9, or the recording is left
# unplayable and an Xorg is orphaned.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# ── SOURCES ────────────────────────────────────────────────────────────────
# The KDS screen recording, and the kitchen camera for the SAME hour. Both
# filenames carry their wall-clock start (…_12_to_13_p0006_PDT = 12:00:06),
# which is how the two feeds are kept in step.
KDS_VIDEO="videos/Wienerschnitzel_Sacramento_CA_95818__kds__2026_09_06_11_to_12_PDT.mkv"
KITCHEN_VIDEO="videos/Wienerschnitzel_Sacramento_CA_95818__camA__2026_09_06_11_to_12_PDT.mkv"

# The other pair available on this box:
#   KDS_VIDEO="videos/Wienerschnitzel_Sacramento_CA_95818__kds__2026_09_13_12_to_13_p0007_PDT.mkv"
#   KITCHEN_VIDEO="videos/Wienerschnitzel_Sacramento_CA_95818__camA__2026_09_13_12_to_13_p0006_PDT.mkv"

# Live cameras, used by --live. The KDS URL carries credentials, so it comes
# from the environment and is never written here.
KDS_RTSP="${KDS_RTSP:-}"
KITCHEN_RTSP="${KITCHEN_RTSP:-}"

# ── SETTINGS ───────────────────────────────────────────────────────────────
PORT="${DASHBOARD_PORT:-8000}"
RUN_FOR="${RUN_FOR_S:-600}"          # seconds of VIDEO to process; 0 = all
REC_DIR="output/recordings"
REC="$REC_DIR/dashboard_$(date +%Y%m%d_%H%M%S).mkv"

LIVE=0
for arg in "$@"; do
    case "$arg" in
        --live) LIVE=1 ;;
        --full) RUN_FOR=0 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

if [ "$LIVE" = "1" ]; then
    if [ -z "$KDS_RTSP" ] || [ -z "$KITCHEN_RTSP" ]; then
        echo "--live needs KDS_RTSP and KITCHEN_RTSP in the environment." >&2
        exit 2
    fi
    export KDS_SOURCE="$KDS_RTSP"
    export VIDEO_SOURCE="$KITCHEN_RTSP"
    RUN_FOR=0
else
    for f in "$KDS_VIDEO" "$KITCHEN_VIDEO"; do
        [ -f "$f" ] || { echo "missing source: $f" >&2; exit 1; }
    done
    export KDS_SOURCE="$KDS_VIDEO"
    export VIDEO_SOURCE="$KITCHEN_VIDEO"
fi

# The dashboard cannot start if the port is still held -- usually a previous
# run, which keeps serving after its video ends, or a socket in TIME_WAIT.
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "Port $PORT is in use. Stop the previous run (Ctrl+C in its terminal)," >&2
    echo "or set DASHBOARD_PORT to something else." >&2
    exit 1
fi

export KDS_MODE=kdsocr
export RUN_FOR_S="$RUN_FOR"
export MAX_TICKETS=0                 # judge every ticket in the window
export DASHBOARD_PORT="$PORT"
export OPEN_BROWSER=0                # this box has no display; see the URLs below
export RECORD_DASHBOARD="$REC"       # the dashboard, as a browser window
export RECORD_WRONG_ONLY=1           # keep ONLY the orders judged WRONG

mkdir -p "$REC_DIR"

LAN_IP="$(ip -4 addr show 2>/dev/null | grep -oP 'inet \K[\d.]+' \
          | grep -v '^127\.' | grep -v '^172\.17\.' | head -1)"
cat <<BANNER

  Order Accuracy + kds-ocr
  ------------------------------------------------------------------
  KDS screen : $KDS_SOURCE
  Kitchen    : $VIDEO_SOURCE
  Window     : $([ "$RUN_FOR" = "0" ] && echo "whole recording" || echo "first ${RUN_FOR}s of video")

  OPEN THE DASHBOARD IN YOUR BROWSER:
      http://${LAN_IP:-<this-host>}:$PORT
      http://10.164.169.197:$PORT      (ZeroTier, slower -- the video feed
                                        is ~1 MB/s and can stall on a tunnel)

  Recording  : the dashboard window, trimmed at the end to
               output/wrong_orders/ -- WRONG orders only.
  Journeys   : output/ticket_journeys.jsonl (every judged ticket)

  Ctrl+C to stop. Do not kill -9: the recorder has to close its
  video and tear down Firefox and the hidden display.
  ------------------------------------------------------------------

BANNER

exec .venv/bin/python main.py
