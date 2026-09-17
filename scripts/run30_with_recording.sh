#!/usr/bin/env bash
# One unattended pass: pipeline -> headless dashboard recording -> wrong-order
# clips -> per-ticket report.
#
# This does NOT replace run_kds.sh. It exists because run_kds.sh picks the
# Xorg/Firefox/GStreamer recorder whenever /usr/lib/xorg/Xorg is present, and on
# this box that recorder cannot run: no /etc/X11/xrdp/xorg.conf, no usable
# display, no `gi` in the venv, and no root to install any of it. So the same
# environment is exported here, minus RECORD_DASHBOARD, and the capture is done
# by scripts/record_dashboard_headless.py instead.
#
# IMPORTANT -- why this does not just wait on the pipeline pid:
# main.py KEEPS SERVING the dashboard after the video ends (run_kds.sh says so:
# "a previous run, which keeps serving after its video ends").  It never exits
# on its own.  A first attempt here waited on --while-pid and recorded for 94
# minutes against a 65-minute run, and the clip step would never have fired.
# So the end of work is detected from the pipeline's own marker line instead,
# and everything is then shut down in order.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

H="$HOME"
export TMPDIR="$H/.tmp"
export CUDA_CACHE_PATH="$H/.nv/ComputeCache"
export XDG_CACHE_HOME="$H/.cache"
export PLAYWRIGHT_BROWSERS_PATH="$H/.cache/ms-playwright"
export PATH="$H/.local/bin:$PATH"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" output/recordings "$H/logs"

PY=".venv/bin/python"
PORT="${DASHBOARD_PORT:-8000}"
STAMP="$(date +%Y%m%d_%H%M%S)"
REC="output/recordings/dashboard_$STAMP.mp4"

export KDS_SOURCE="Wienerschnitzel_Sacramento_CA_95818__kds__2026_09_06_11_to_12_PDT.mkv"
export VIDEO_SOURCE="Wienerschnitzel_Sacramento_CA_95818__camA__2026_09_06_11_to_12_PDT.mkv"
export KDS_MODE=kdsocr
export RUN_FOR_S="${RUN_FOR_S:-3600}"
export MAX_TICKETS=0
export DASHBOARD_PORT="$PORT"
export OPEN_BROWSER=0
# Archive the previous run's order history before starting. Without this the
# state machine rehydrates it: the dashboard opens showing old orders and old
# stats, and on_kds_ticket finds a PREVIOUS run's order for the same CHK ref
# and reuses it instead of opening a fresh one. Seen 2026-09-17: a run reported
# 15 orders / 13 failures at frame 45.
export FRESH_START=1

# Wall-clock estimate, only used to size the recorder's bitrate cap. Measured
# throughput is ~13.6 fps against a 30 fps source, i.e. ~0.45x realtime.
EXPECT_S="${EXPECT_S:-$(awk "BEGIN{printf \"%d\", $RUN_FOR_S * 30 / 13.6}")}"

echo "[$(date +%T)] starting pipeline (RUN_FOR_S=$RUN_FOR_S s of video, expect ~$((EXPECT_S/60)) min wall)"
$PY main.py > "$H/logs/pipeline.log" 2>&1 &
PIPE=$!
echo "[$(date +%T)] pipeline pid $PIPE"

for _ in $(seq 1 120); do
    if curl -sf -o /dev/null "http://127.0.0.1:$PORT/"; then break; fi
    if ! kill -0 "$PIPE" 2>/dev/null; then
        echo "[$(date +%T)] pipeline died before serving; see $H/logs/pipeline.log" >&2
        exit 1
    fi
    sleep 2
done

# NO_RECORD=1: verdicts and JSON only. The capture costs a Chromium, an ffmpeg
# and ~6 screenshots a second on a box whose GPU is already the constraint, and
# a run wanted only for its numbers does not need it.
if [ "${NO_RECORD:-0}" = "1" ]; then
    echo "[$(date +%T)] NO_RECORD=1 — dashboard is up at http://127.0.0.1:$PORT, not recording"
    RECPID=""
else
echo "[$(date +%T)] dashboard is up; starting headless recorder -> $REC"

$PY scripts/record_dashboard_headless.py \
    --url "http://127.0.0.1:$PORT/" --out "$REC" \
    --fps 6 --out-fps 30 --quality 95 --crf 20 \
    --max-bytes 1000000000 --expect-s "$EXPECT_S" \
    --settle-s 8 > "$H/logs/recorder.log" 2>&1 &
RECPID=$!
REC_START=$(date +%s)
echo "[$(date +%T)] recorder pid $RECPID"
fi

# The pipeline prints this the moment its capture loop ends. Poll for it rather
# than for process exit, because the process deliberately outlives the work.
echo "[$(date +%T)] waiting for the pipeline to finish its video..."
while kill -0 "$PIPE" 2>/dev/null; do
    if grep -q "In finally block — main loop ended" "$H/logs/pipeline.log" 2>/dev/null; then
        echo "[$(date +%T)] pipeline reached end of video"
        break
    fi
    sleep 10
done

# Let the last verdicts and the hotdog summary land before cutting the capture.
sleep 20

# SIGTERM, not SIGINT. Playwright's sync API swallows SIGINT, so the recorder
# ignored it twice: on 2026-09-16 it kept capturing a frozen dashboard for five
# hours, and on 2026-09-17 for nineteen minutes, with this script blocked on
# `wait` and the clips never cut.
#
# On SIGTERM the process dies, which closes ffmpeg's stdin; ffmpeg sees EOF and
# writes its trailer normally, so the file is still finalized properly. The
# sidecar is the one casualty -- it is written by the recorder's own exit path
# -- so it is reconstructed below from facts this script already knows.
if [ -z "${RECPID:-}" ]; then
    echo "[$(date +%T)] no recorder to stop"
else
echo "[$(date +%T)] stopping recorder"
kill -TERM "$RECPID" 2>/dev/null

# Never block indefinitely on the recorder again.
for _ in $(seq 1 60); do
    kill -0 "$RECPID" 2>/dev/null || break
    sleep 1
done
kill -0 "$RECPID" 2>/dev/null && { echo "[$(date +%T)] recorder ignored SIGTERM; SIGKILL"; kill -KILL "$RECPID" 2>/dev/null; }
for _ in $(seq 1 60); do pgrep -f "[f]fmpeg .*$STAMP" >/dev/null || break; sleep 1; done
echo "[$(date +%T)] recorder stopped"

# The clipper needs a sidecar to place cuts on the wall clock; without one it
# silently falls back to media time and every clip lands in the wrong place.
if [ ! -f "${REC%.mp4}.json" ] && [ -f "$REC" ]; then
    echo "[$(date +%T)] recorder left no sidecar; reconstructing it"
    REC_END=$(date +%s)
    "$PY" - "$REC" "$REC_START" "$REC_END" <<'PY'
import json, sys, time, pathlib
rec, t0, t1 = pathlib.Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
rec.with_suffix(".json").write_text(json.dumps({
    "file": rec.name,
    "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
    "wall_s": round(t1 - t0, 1),
    "fps": 6, "out_fps": 30, "size": [1920, 1080],
    "recorder": "playwright-chromium-headless",
    "note": "reconstructed by run30_with_recording.sh: the recorder was killed "
            "and never wrote its own sidecar.",
}, indent=2))
print("wrote", rec.with_suffix(".json").name)
PY
fi
fi

echo "[$(date +%T)] stopping pipeline"
kill -TERM "$PIPE" 2>/dev/null
sleep 5
kill -0 "$PIPE" 2>/dev/null && kill -KILL "$PIPE" 2>/dev/null

if [ -n "${RECPID:-}" ]; then
    echo "[$(date +%T)] cutting wrong-order clips"
    $PY scripts/clip_wrong_orders.py --recording "$REC" --timebase wall --keep-full 2>&1 | tail -30
fi

echo "[$(date +%T)] per-ticket report"
$PY scripts/ticket_report.py --json output/ticket_report.json 2>&1 | tail -60

echo "[$(date +%T)] DONE"
ls -la "$REC" "${REC%.mp4}.json" 2>/dev/null
ls -la output/wrong_orders/ 2>/dev/null | head -20
