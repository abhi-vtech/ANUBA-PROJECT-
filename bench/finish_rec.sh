#!/usr/bin/env bash
# Finish the recorded TensorRT run even if the Claude Code session that started
# it ends: wait for the pipeline and the tracked watcher, redo any step they did
# not complete, then store both videos in output/recordings/ and symlink them
# back into the results folder.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="bench/results/camA_trt_rec"
NAME="detections_camA_trt"
REC="$OUT/$NAME.mkv"; MP4="$OUT/$NAME.mp4"
DEST="output/recordings"
exec >>bench/finish_rec.log 2>&1
echo "[$(date -Is)] finisher started"
alive() { ps -eo args | grep -q "$1"; }
while alive "[.]venv/bin/python -m src.main"; do sleep 60; done
echo "[$(date -Is)] pipeline exited"
sleep 15
# Let the tracked watcher do analysis + transcode while it is still around.
while alive "[w]ait_rec.sh" || alive "[f]fmpeg.*$NAME"; do sleep 30; done
echo "[$(date -Is)] watcher and transcode idle"
WANT=$(grep -oE '\[RECORD\] wrote [0-9]+' "$OUT/pipeline.log" | grep -oE '[0-9]+$' | tail -1)
[ -s "$OUT/summary.txt" ] || { echo "analysis missing -> running it"; .venv/bin/python bench/analyze.py "$OUT" > "$OUT/summary.txt" 2>&1; }
mp4_frames() { .venv/bin/python -c "import cv2,sys;c=cv2.VideoCapture(sys.argv[1]);print(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) if c.isOpened() else 0)" "$1" 2>/dev/null || echo 0; }
GOT=0; [ -s "$MP4" ] && GOT=$(mp4_frames "$MP4")
if [ -z "$WANT" ] || [ "$GOT" != "$WANT" ]; then
  echo "mp4 missing or incomplete (have $GOT, recorder wrote ${WANT:-?}) -> transcoding"
  ./bench/transcode.sh "$REC" "$MP4" fast 23 | tail -1
  GOT=$(mp4_frames "$MP4")
fi
echo "mp4 frames=$GOT, recorder frames=${WANT:-?}"
mkdir -p "$DEST"
for f in "$REC" "$MP4"; do
  { [ -f "$f" ] && [ ! -L "$f" ]; } || continue
  mv -f "$f" "$DEST/" && ln -sfn "../../../$DEST/$(basename "$f")" "$f"
  echo "stored $(basename "$f") -> $DEST/"
done
ls -lh "$DEST"
echo "[$(date -Is)] finisher done"
touch bench/finish_rec.done
