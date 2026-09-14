#!/usr/bin/env bash
# One full recorded benchmark, end to end, safe to run detached:
#   run -> analyse -> H.264 transcode -> verify frames -> store in output/recordings/
#   ./bench/record_run.sh <label> <model_path> [video]
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LABEL="${1:?usage: record_run.sh <label> <model_path> [video]}"
MODEL="${2:?usage: record_run.sh <label> <model_path> [video]}"
VIDEO="${3:-Wienerschnitzel_Sacramento_CA_95818__camA__2026_09_06_11_to_12_PDT.mkv}"
OUT="bench/results/camA_${LABEL}_rec"
NAME="detections_camA_${LABEL}"
REC="$OUT/$NAME.mkv"; MP4="$OUT/$NAME.mp4"
DEST="output/recordings"
exec >>"bench/record_${LABEL}.log" 2>&1
echo "[$(date -Is)] start: label=$LABEL model=$MODEL"
alive() { ps -eo args | grep -qE "$1"; }
if alive "[.]venv/bin/python -m src.main"; then echo "ABORT: another pipeline is running"; exit 1; fi
for i in $(seq 1 30); do ss -tln 2>/dev/null | grep -qE ":8000[[:space:]]" || break; sleep 10; done
rm -rf "$OUT"
MODEL_PATH="$MODEL" RECORD_VIDEO="$REC" ./bench/run_benchmark.sh "$VIDEO" "$OUT"
echo "[$(date -Is)] run finished: $(grep -E '^exit_code|^wall_seconds|^model_backend' "$OUT/system.txt" | tr '\n' ' ')"
grep "\[RECORD\]" "$OUT/pipeline.log"
.venv/bin/python bench/analyze.py "$OUT" > "$OUT/summary.txt" 2>&1 && echo "analysis written"
./bench/transcode.sh "$REC" "$MP4" fast 23 | tail -1
WANT=$(grep -oE '\[RECORD\] wrote [0-9]+' "$OUT/pipeline.log" | grep -oE '[0-9]+$' | tail -1)
GOT=$(.venv/bin/python -c "import cv2,sys;c=cv2.VideoCapture(sys.argv[1]);print(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) if c.isOpened() else 0)" "$MP4" 2>/dev/null || echo 0)
echo "frames: recorder=${WANT:-?} mp4=$GOT"
mkdir -p "$DEST"
for f in "$REC" "$MP4"; do
  { [ -f "$f" ] && [ ! -L "$f" ]; } || continue
  mv -f "$f" "$DEST/" && ln -sfn "../../../$DEST/$(basename "$f")" "$f"
  echo "stored $(basename "$f") -> $DEST/"
done
ls -lh "$DEST"
echo "[$(date -Is)] done"
touch "bench/record_${LABEL}.done"
