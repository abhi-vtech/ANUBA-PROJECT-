#!/usr/bin/env bash
# Capture both RTSP feeds to hourly files, one folder per camera.
#
#   KDS_RTSP=... CAMA_RTSP=... ./scripts/capture_rtsp.sh          # until stopped
#   KDS_RTSP=... CAMA_RTSP=... HOURS=3 ./scripts/capture_rtsp.sh  # three hours
#
# Why capture instead of processing the feed live
# -----------------------------------------------
# A live source uses latest-wins in VideoCaptureThread, so at ~13 fps against a
# 30 fps camera more than half the frames are discarded. A FILE source uses a
# blocking queue and is therefore read at the pipeline's own pace, every frame
# processed. A camera cannot be asked to slow down -- blocking a live consumer
# just moves the drop into FFmpeg's buffer and grows latency without bound --
# so the only way to get the pipeline's pace AND every frame is to land the
# footage on disk first and process the file.
#
# `-c copy` means no decode and no re-encode: the bytes the camera sent are
# written as they arrive, at a few percent of one core, so this runs alongside a
# processing run without competing with it.
#
# Naming
# ------
# Both cameras get the IDENTICAL reference, which is what makes a pair a pair:
#
#   captures/camA/Wienerschnitzel_Sacramento_CA_95818__camA__2026_09_17_18_to_19_p2345_PDT.mkv
#   captures/kds/ Wienerschnitzel_Sacramento_CA_95818__kds__2026_09_17_18_to_19_p2345_PDT.mkv
#
# That is the convention src/kdsocr/clock.py already parses:
# YYYY_MM_DD_HH_to_HH, an optional _pMMSS start offset into the hour, then the
# timezone. The pipeline derives each video's wall clock from this, and the KDS
# feed is paced against the production video's start -- so a mismatched pair
# would desynchronise the tickets from the food. Giving both files one reference
# removes that failure mode: the downloaded sets differ (p0006 vs p0007) because
# two independent recorders started a second apart; these two start together.
#
# Hours are labelled in PACIFIC time, because the parser reads the label as PDT
# and the store is in Sacramento. This box runs UTC, so the label is NOT the
# local hour here.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

: "${KDS_RTSP:?set KDS_RTSP}"
: "${CAMA_RTSP:?set CAMA_RTSP}"

STORE="${STORE_PREFIX:-Wienerschnitzel_Sacramento_CA_95818}"
ROOT="${CAPTURE_ROOT:-captures}"
HOURS="${HOURS:-0}"          # 0 = run until stopped
FFMPEG="${FFMPEG:-$HOME/.local/bin/ffmpeg}"

mkdir -p "$ROOT/camA" "$ROOT/kds"

seg=0
while :; do
    [ "$HOURS" -gt 0 ] && [ "$seg" -ge "$HOURS" ] && break

    # Pacific wall clock decides the label; both cameras share it.
    NOW_H=$(TZ=America/Los_Angeles date +%H)
    NOW_YMD=$(TZ=America/Los_Angeles date +%Y_%m_%d)
    NEXT_H=$(printf '%02d' $(( (10#$NOW_H + 1) % 24 )))
    OFF_M=$(TZ=America/Los_Angeles date +%M)
    OFF_S=$(TZ=America/Los_Angeles date +%S)
    TZL=$(TZ=America/Los_Angeles date +%Z)

    # Seconds left in this clock hour: segments land on hour boundaries, so a
    # file never straddles two hours and its label stays true.
    REMAIN=$(( 3600 - (10#$OFF_M * 60 + 10#$OFF_S) ))
    [ "$REMAIN" -lt 60 ] && { sleep "$REMAIN"; continue; }

    if [ "$OFF_M$OFF_S" = "0000" ]; then
        REF="${NOW_YMD}_${NOW_H}_to_${NEXT_H}_${TZL}"
    else
        REF="${NOW_YMD}_${NOW_H}_to_${NEXT_H}_p${OFF_M}${OFF_S}_${TZL}"
    fi

    CAMA_OUT="$ROOT/camA/${STORE}__camA__${REF}.mkv"
    KDS_OUT="$ROOT/kds/${STORE}__kds__${REF}.mkv"

    echo "[$(date +%T)] segment $((seg+1)): ${REF}  (${REMAIN}s)"
    echo "            camA -> $CAMA_OUT"
    echo "            kds  -> $KDS_OUT"

    # Started together, bounded together, so the pair covers the same minutes.
    "$FFMPEG" -hide_banner -loglevel error -rtsp_transport tcp -i "$CAMA_RTSP" \
        -t "$REMAIN" -c copy -f matroska -y "$CAMA_OUT" &
    A=$!
    "$FFMPEG" -hide_banner -loglevel error -rtsp_transport tcp -i "$KDS_RTSP" \
        -t "$REMAIN" -c copy -f matroska -y "$KDS_OUT" &
    K=$!

    wait $A; rc_a=$?
    wait $K; rc_k=$?
    echo "[$(date +%T)] segment done (camA rc=$rc_a, kds rc=$rc_k)"
    ls -la "$CAMA_OUT" "$KDS_OUT" 2>/dev/null | awk '{printf "            %.0f MB  %s\n", $5/1048576, $9}'
    seg=$((seg+1))
done

echo "[$(date +%T)] capture finished: $seg segment(s) in $ROOT/"
