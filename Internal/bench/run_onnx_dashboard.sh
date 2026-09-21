#!/usr/bin/env bash
# One full recorded run, safe to start detached:
#   nohup setsid ./bench/run_onnx_dashboard.sh [video] >/dev/null 2>&1 &
#
#   * ONNX model through ONNX Runtime's TensorRT backend (FP16)
#   * hardware video decode (nvv4l2decoder) and hardware H.264 encode
#   * the dashboard's Analysis window (src/feed_analysis.py)
#   * a recording of the dashboard exactly as a browser shows it
#     (scripts/record_dashboard.py), plus the annotated camera feed
#
# Results: bench/results/<label>_dashboard/  (benchmark telemetry + summary)
#          output/recordings/dashboard_<label>.mp4               (wall-clock speed)
#          output/recordings/dashboard_<label>_camera_speed.mp4  (same video, camera speed)
#          output/recordings/detections_<label>.mp4              (annotated feed)
cd "$(dirname "${BASH_SOURCE[0]}")/.."
VIDEO="${1:-Wienerschnitzel_Sacramento_CA_95818__camA__2026_09_06_11_to_12_PDT.mkv}"
LABEL="${LABEL:-camA_onnx}"
OUT="bench/results/${LABEL}_dashboard"
DEST="output/recordings"
FEED_MKV="$OUT/detections_${LABEL}.mkv"
DASH_MKV="$DEST/dashboard_${LABEL}.mkv"
FF=$(bench/.tools/bin/python -c "import imageio_ffmpeg as f; print(f.get_ffmpeg_exe())")
exec >>"bench/run_${LABEL}_dashboard.log" 2>&1
echo "[$(date -Is)] start: video=$VIDEO label=$LABEL"

running() { ps -eo args | grep -q "[.]venv/bin/python -m src.main"; }
if running; then echo "ABORT: another pipeline is running"; exit 1; fi
for i in $(seq 1 30); do ss -tln 2>/dev/null | grep -qE ":8000[[:space:]]" || break; sleep 10; done
rm -rf "$OUT"; mkdir -p "$OUT" "$DEST"
rm -f "$DASH_MKV" "${DASH_MKV%.mkv}.json" "${DASH_MKV%.mkv}.mp4" "${DASH_MKV%.mkv}_camera_speed.mp4"

MODEL_PATH="rf_trained/weights (1).onnx" ONNX_PROVIDER=tensorrt INGEST=gstreamer RECORD_VIDEO="$FEED_MKV" \
  ./bench/run_benchmark.sh "$VIDEO" "$OUT" &
BENCH=$!

PID=""
for i in $(seq 1 180); do
  PID=$(pgrep -f "[.]venv/bin/python -m src.main" | head -1)
  [ -n "$PID" ] && break
  sleep 1
done
if [ -z "$PID" ]; then echo "pipeline did not start"; wait "$BENCH"; exit 1; fi
echo "[$(date -Is)] pipeline pid $PID; starting the dashboard recorder"
.venv/bin/python scripts/record_dashboard.py --while-pid "$PID" --out "$DASH_MKV" --bitrate "${DASH_BITRATE:-8000000}" &
REC=$!

wait "$BENCH"
echo "[$(date -Is)] pipeline finished: $(grep -E '^exit_code|^wall_seconds' "$OUT/system.txt" | tr '\n' ' ')"
wait "$REC"
echo "[$(date -Is)] dashboard recorder finished"
grep "\[RECORD\]" "$OUT/pipeline.log"

.venv/bin/python bench/analyze.py "$OUT" > "$OUT/summary.txt" 2>&1 && echo "benchmark summary written"
cp -f output/feed_analysis.json "$OUT/feed_analysis.json" 2>/dev/null && echo "feed analysis copied"

# Both recordings are already H.264 from the hardware encoder: MKV -> MP4 is a remux.
for f in "$FEED_MKV" "$DASH_MKV"; do
  [ -s "$f" ] || continue
  "$FF" -hide_banner -loglevel error -y -i "$f" -c copy -movflags +faststart "${f%.mkv}.mp4" && echo "mp4: ${f%.mkv}.mp4"
done

# The dashboard capture follows the wall clock; this copy plays at camera speed.
DUR=$(sed -n 's/^video_duration_s=//p' "$OUT/system.txt")
WALL=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['wall_s'])" "${DASH_MKV%.mkv}.json" 2>/dev/null)
if [ -n "$DUR" ] && [ -n "$WALL" ] && [ -s "$DASH_MKV" ]; then
  SCALE=$(python3 -c "print(round($DUR / $WALL, 6))")
  "$FF" -hide_banner -loglevel error -y -itsscale "$SCALE" -i "$DASH_MKV" -c copy -movflags +faststart \
    "${DASH_MKV%.mkv}_camera_speed.mp4" && echo "camera-speed copy written (timestamps x$SCALE)"
fi

# Keep the annotated feed next to the dashboard recording.
for f in "$FEED_MKV" "${FEED_MKV%.mkv}.mp4"; do
  { [ -f "$f" ] && [ ! -L "$f" ]; } || continue
  mv -f "$f" "$DEST/" && ln -sfn "../../../$DEST/$(basename "$f")" "$f"
done
ls -lh "$DEST" | grep -E "$LABEL"
echo "[$(date -Is)] done"
touch "bench/run_${LABEL}_dashboard.done"
