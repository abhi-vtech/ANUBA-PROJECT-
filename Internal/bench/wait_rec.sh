#!/usr/bin/env bash
# Wait for the recorded TensorRT run, then analyse it and transcode the video.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="bench/results/camA_trt_rec"
REC="$OUT/detections_camA_trt.mkv"
MP4="$OUT/detections_camA_trt.mp4"
running() { ps -eo args | grep -q "[.]venv/bin/python -m src.main"; }
# The launch happens in parallel with this watcher, so first wait for it to appear.
for i in $(seq 1 100); do running && break; sleep 3; done
running || { echo "pipeline never started"; exit 1; }
while running; do sleep 60; done
sleep 5
echo "pipeline finished $(date -Is)"
grep -E "^exit_code|^wall_seconds" "$OUT/system.txt"
grep -E "\[RECORD\]" "$OUT/pipeline.log"
.venv/bin/python bench/analyze.py "$OUT" > "$OUT/summary.txt" 2>&1 && echo "analysis written"
# Transcode only after the run, so it never competes with the measured pipeline.
./bench/transcode.sh "$REC" "$MP4" fast 23 2>&1 | tail -1
.venv/bin/python -c "
import cv2
c=cv2.VideoCapture('$MP4')
print(f'mp4 check: readable={c.isOpened()} frames={int(c.get(cv2.CAP_PROP_FRAME_COUNT))} fps={c.get(cv2.CAP_PROP_FPS)}')"
ls -lh "$REC" "$MP4" | awk '{print $5, $9}'
