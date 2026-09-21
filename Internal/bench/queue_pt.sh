#!/usr/bin/env bash
# Run the .pt recorded benchmark only after the TensorRT recording is fully stored.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec >>bench/queue_pt.log 2>&1
echo "[$(date -Is)] queued: .pt run waits for the TensorRT recording to be stored"
while [ ! -f bench/finish_rec.done ]; do sleep 60; done
echo "[$(date -Is)] TensorRT recording stored; waiting for the system to go idle"
while ps -eo args | grep -qE "[.]venv/bin/python -m src.main|[f]fmpeg.*detections|[w]ait_rec.sh|[f]inish_rec.sh"; do sleep 30; done
sleep 60
echo "[$(date -Is)] launching .pt run"
./bench/record_run.sh pt "rf_trained/weights (1).pt"
echo "[$(date -Is)] queue finished"
