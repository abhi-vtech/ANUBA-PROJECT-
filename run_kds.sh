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
#
# Runs on the Jetson/Linux box and on Windows (Git Bash). The three things
# that differ -- where the venv keeps its interpreter, how the OS is asked
# about sockets and addresses, and whether the dashboard can be recorded at
# all -- are each resolved at the point of use below.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) WINDOWS=1 ;;
    *)                    WINDOWS=0 ;;
esac

# The venv keeps its interpreter in bin/ on Linux and Scripts/ on Windows.
if   [ -x .venv/bin/python ];         then PYTHON=".venv/bin/python"
elif [ -x .venv/Scripts/python.exe ]; then PYTHON=".venv/Scripts/python.exe"
else
    echo "No interpreter in .venv (looked in bin/ and Scripts/). Run: uv sync" >&2
    exit 1
fi

# ── SOURCES ────────────────────────────────────────────────────────────────
# The KDS screen recording, and the kitchen camera for the SAME hour. Both
# filenames carry their wall-clock start (…_12_to_13_p0006_PDT = 12:00:06),
# which is how the two feeds are kept in step.
KDS_VIDEO="videos/Wienerschnitzel_Sacramento_CA_95818__kds__2026_09_06_11_to_12_PDT.mkv"
KITCHEN_VIDEO="videos/Wienerschnitzel_Sacramento_CA_95818__camA__2026_09_06_11_to_12_PDT.mkv"

# The other pair available on this box:
#   KDS_VIDEO="videos/Wienerschnitzel_Sacramento_CA_95818__kds__2026_09_13_12_to_13_p0007_PDT.mkv"
#   KITCHEN_VIDEO="videos/Wienerschnitzel_Sacramento_CA_95818__camA__2026_09_13_12_to_13_p0006_PDT.mkv"

# Recordings do not sit in the same place on every box: videos/ on the Jetson,
# beside the code on the Windows machine. Accept either, rather than making
# the SOURCE lines above box-specific.
resolve_source() {
    local want="$1" base candidate
    base="$(basename "$want")"
    for candidate in "$want" "$base" "videos/$base"; do
        if [ -f "$candidate" ]; then echo "$candidate"; return 0; fi
    done
    return 1
}

# Both feeds must come from the same hour, so a pair is discovered as a pair:
# a KDS recording, then the camA recording whose date and hour window match
# it. The p000N suffix is the per-camera start offset and differs between the
# two files of a legitimate pair, so it is not part of the key.
discover_pair() {
    local kds camA key
    for kds in $(ls -1 videos/*__kds__*.mkv *__kds__*.mkv 2>/dev/null | sort -u); do
        key="$(basename "$kds" | sed -n "s/.*__kds__\(.*_[0-9]*_to_[0-9]*\).*/\1/p")"
        [ -n "$key" ] || continue
        for camA in $(ls -1 videos/*__camA__*"$key"*.mkv *__camA__*"$key"*.mkv 2>/dev/null | sort -u); do
            echo "$kds"
            echo "$camA"
            return 0
        done
    done
    return 1
}

# Live cameras, used by --live. The KDS URL carries credentials, so it comes
# from the environment and is never written here.
KDS_RTSP="${KDS_RTSP:-}"
KITCHEN_RTSP="${KITCHEN_RTSP:-}"

# ── SETTINGS ───────────────────────────────────────────────────────────────
PORT="${DASHBOARD_PORT:-8000}"
RUN_FOR="${RUN_FOR_S:-600}"          # seconds of VIDEO to process; 0 = all
REC_DIR="output/recordings"
STAMP="$(date +%Y%m%d_%H%M%S)"   # names whichever recording this box can make

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
    kds_found=""
    kitchen_found=""
    if kds_found="$(resolve_source "$KDS_VIDEO")" \
       && kitchen_found="$(resolve_source "$KITCHEN_VIDEO")"; then
        :
    elif pair="$(discover_pair)"; then
        kds_found="$(echo "$pair" | sed -n 1p)"
        kitchen_found="$(echo "$pair" | sed -n 2p)"
        echo "The pair set in this script is not on this box; using the pair that is:" >&2
        echo "  KDS     : $kds_found" >&2
        echo "  Kitchen : $kitchen_found" >&2
        echo >&2
    else
        echo "missing source: $KDS_VIDEO" >&2
        echo "and no other kds/camA pair was found in videos/ or beside the code." >&2
        exit 1
    fi
    export KDS_SOURCE="$kds_found"
    export VIDEO_SOURCE="$kitchen_found"
fi

# The dashboard cannot start if the port is still held -- usually a previous
# run, which keeps serving after its video ends, or a socket in TIME_WAIT.
# Asked through Python because ss is Linux-only, and because this is the same
# bind the server itself will go on to attempt.
if ! "$PYTHON" -c "
import socket, sys
probe = socket.socket()
try:
    probe.bind(('0.0.0.0', $PORT))
except OSError:
    sys.exit(1)
finally:
    probe.close()
" 2>/dev/null; then
    echo "Port $PORT is in use. Stop the previous run (Ctrl+C in its terminal)," >&2
    echo "or set DASHBOARD_PORT to something else." >&2
    exit 1
fi

export KDS_MODE=kdsocr
export RUN_FOR_S="$RUN_FOR"
export MAX_TICKETS=0                 # judge every ticket in the window
export DASHBOARD_PORT="$PORT"
export OPEN_BROWSER="${OPEN_BROWSER:-0}"  # the Jetson has no display; see the URLs below
export RECORD_WRONG_ONLY=1           # keep ONLY the orders judged WRONG

# Recording the dashboard drives a hidden Xorg and Firefox and encodes with
# GStreamer (scripts/record_dashboard.py) -- Linux, and in practice the Jetson.
# Asking for it anywhere else buys a child process that dies on import and no
# recording at all, so a box without those records the annotated feed instead:
# the same boxes, masks, zones and trails, written straight from the pipeline
# by OpenCV. Either file is trimmed to the WRONG orders the same way -- the
# dashboard capture on the wall clock, the annotated feed on the video's own
# media time.
if [ "$WINDOWS" = "1" ] || [ -x /usr/lib/xorg/Xorg ]; then
    # The dashboard as the browser draws it -- ticket panel, checklist and
    # verdict, not just the detection overlay. The Jetson gets there through a
    # hidden Xorg and Firefox; Windows through Playwright and the installed
    # Chrome (scripts/record_dashboard_win.py). main.py picks the right one.
    export RECORD_DASHBOARD="$REC_DIR/dashboard_$STAMP.mkv"
    RECORDING_NOTE="the dashboard window, trimmed at the end to
               output/wrong_orders/ -- WRONG orders only."
else
    # `-` rather than `:-`, so RECORD_VIDEO= (explicitly empty) turns the
    # recording off entirely instead of falling back to the default path.
    # Encoding every frame costs about a quarter of the loop rate, which is
    # worth skipping on a run wanted only for its verdicts.
    export RECORD_VIDEO="${RECORD_VIDEO-$REC_DIR/annotated_$STAMP.mkv}"
    # nvv4l2h264enc is a Jetson part; asking for it here only buys a failed
    # probe before the OpenCV writer takes over anyway.
    export RECORD_ENCODER=opencv
    # 30 fps, the rate these cameras record at.  It has to EQUAL the source
    # rate rather than merely be a round number: the recorder writes one
    # frame per processed frame, so a playback rate that differs from the
    # source's stops a recorded second meaning a video second -- and the
    # wrong-order cuts are placed in the video's media time, so they would
    # land in the wrong place.  Overridable, but only for footage shot at a
    # different rate.
    export RECORD_FPS="${RECORD_FPS:-30}"
    if [ -n "$RECORD_VIDEO" ]; then
        RECORDING_NOTE="the annotated feed (no hidden display on this box),
               trimmed at the end to output/wrong_orders/ -- WRONG
               orders only."
    else
        RECORDING_NOTE="off (RECORD_VIDEO is empty) -- this run is for the
               verdicts alone; nothing is written to
               output/wrong_orders/."
    fi
fi

mkdir -p "$REC_DIR"

# The LAN address, asked for in the way that works here. It only fills in the
# URL printed below, so neither form may fail the run.
LAN_IP=""
if [ "$WINDOWS" = "0" ]; then
    LAN_IP="$( { ip -4 addr show 2>/dev/null || true; } | grep -oP 'inet \K[\d.]+' \
              | grep -v '^127\.' | grep -v '^172\.17\.' | head -1 || true)"
else
    LAN_IP="$("$PYTHON" -c "
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    sock.connect(('8.8.8.8', 80))   # sends nothing; this only picks a route
    print(sock.getsockname()[0])
except OSError:
    pass
finally:
    sock.close()
" 2>/dev/null || true)"
fi

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

  Recording  : $RECORDING_NOTE
  Journeys   : output/ticket_journeys.jsonl (every judged ticket)

  Ctrl+C to stop. Do not kill -9: the recorder has to close its
  video and tear down Firefox and the hidden display.
  ------------------------------------------------------------------

BANNER

exec "$PYTHON" main.py
