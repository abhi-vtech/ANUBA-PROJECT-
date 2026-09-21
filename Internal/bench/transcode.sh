#!/usr/bin/env bash
# Transcode the pipeline's mp4v recording to H.264 MP4 for sharing and playback.
#   ./bench/transcode.sh <in.mkv> [out.mp4] [preset] [crf]
# Runs on all CPU cores -- do not run it while a benchmark is being measured.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
IN="${1:?usage: transcode.sh <in.mkv> [out.mp4] [preset] [crf]}"
OUT="${2:-${IN%.*}.mp4}"
PRESET="${3:-fast}"     # ultrafast..veryslow: slower = smaller file, same quality
CRF="${4:-23}"          # 18 visually lossless .. 28 small; 23 is x264's default
FFMPEG=$(bench/.tools/bin/python -c "import imageio_ffmpeg as f; print(f.get_ffmpeg_exe())")
echo "in     : $IN ($(du -h "$IN" | cut -f1))"
echo "out    : $OUT"
echo "codec  : libx264 preset=$PRESET crf=$CRF, yuv420p, faststart"
START=$(date +%s)
"$FFMPEG" -hide_banner -loglevel error -stats -y -i "$IN" \
  -c:v libx264 -preset "$PRESET" -crf "$CRF" -pix_fmt yuv420p \
  -movflags +faststart -threads 0 "$OUT"
echo "done in $(( $(date +%s) - START ))s -> $OUT ($(du -h "$OUT" | cut -f1))"
