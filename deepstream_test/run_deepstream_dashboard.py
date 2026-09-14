#!/usr/bin/env python3
"""Full-hour DeepStream run with the same live dashboard/analysis window as
the ONNX Runtime pipeline.

Architecture: the DeepStream container (no FastAPI/websocket stack inside
it) writes a live JSON snapshot every 0.25s to a file on the mounted volume.
This host process runs the *real* src/dashboard.py FastAPI app (the one the
ONNX Runtime pipeline already uses) plus the real src/system_monitor.py
tegrastats sampler, and bridges the two: a watcher thread tails the JSON
file DeepStream is writing and calls dashboard.set_analysis()/set_system()
directly, exactly the way src/main.py does today.

Usage:
    .venv/bin/python deepstream_test/run_deepstream_dashboard.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # OrderAccuracy/Internal
DS_DIR = ROOT / "deepstream_test"
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402
from src import dashboard  # noqa: E402
from src.system_monitor import SystemMonitor  # noqa: E402

RUN_TAG = "camA_full"
LIVE_PATH = DS_DIR / f"live_{RUN_TAG}.json"
LOG_PATH = DS_DIR / f"run_{RUN_TAG}.log"
VIDEO_DURATION_S = 3593.7
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", 8000))

DOCKER_CMD = [
    "docker", "run", "--rm", "--runtime", "nvidia", "--network", "none",
    "-e", "LD_LIBRARY_PATH=/opt/nvidia/l4t-gpu-libs/nvgpu:/usr/local/nvidia/lib:"
          "/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda-13.2/lib64:"
          "/opt/nvidia/deepstream/deepstream-9.1/lib",
    "-e", f"DS_CLIP=/work/deepstream_test/camA_full.h264",
    "-e", f"DS_RUN_TAG={RUN_TAG}",
    "-e", f"DS_VIDEO_DURATION_S={VIDEO_DURATION_S}",
    "-v", f"{ROOT}:/work",
    "--entrypoint", "/bin/bash", "oad/deepstream:9.1",
    "-c", "cd /work/deepstream_test && python3 -X faulthandler run_deepstream.py",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def watch_live_file(stop: threading.Event) -> None:
    """Poll the JSON DeepStream is writing and push it into the dashboard's
    globals, exactly like main.py's dashboard.set_analysis()/set_system()."""
    last_mtime = 0.0
    system_monitor = SystemMonitor().start()
    if not system_monitor.available:
        log("WARNING: tegrastats not found on host -- system panel will be sparse")
    while not stop.is_set():
        try:
            mtime = LIVE_PATH.stat().st_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                payload = json.loads(LIVE_PATH.read_text())
                dashboard.set_analysis(payload.get("analysis", {}))
                sys_snap = payload.get("system", {})
                sys_snap["hw"] = system_monitor.snapshot()
                dashboard.set_system(sys_snap)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        time.sleep(0.25)
    system_monitor.stop()


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    log(f"starting dashboard on http://0.0.0.0:{DASHBOARD_PORT}")
    dashboard_thread = threading.Thread(
        target=lambda: uvicorn.run(dashboard.app, host="0.0.0.0", port=DASHBOARD_PORT, log_level="warning"),
        daemon=True,
    )
    dashboard_thread.start()
    time.sleep(2.0)

    stop_watch = threading.Event()
    watcher = threading.Thread(target=watch_live_file, args=(stop_watch,), daemon=True)
    watcher.start()

    log(f"launching DeepStream container for the full hour (video_duration_s={VIDEO_DURATION_S})")
    log(f"docker log -> {LOG_PATH}")
    with open(LOG_PATH, "w") as logf:
        proc = subprocess.Popen(DOCKER_CMD, stdout=logf, stderr=subprocess.STDOUT, cwd=str(ROOT))
        start = time.time()
        try:
            ret = proc.wait()
        except KeyboardInterrupt:
            log("interrupted -- terminating docker container")
            proc.terminate()
            ret = proc.wait(timeout=15)
    elapsed = time.time() - start
    log(f"DeepStream container exited code={ret} after {elapsed:.1f}s wall")

    stop_watch.set()
    watcher.join(timeout=3)

    summary_path = DS_DIR / f"summary_{RUN_TAG}.json"
    if summary_path.exists():
        log(f"summary written to {summary_path}")
    else:
        log(f"WARNING: {summary_path} was not written -- check {LOG_PATH}")

    log("dashboard stays up serving the final state -- Ctrl+C to stop, or leave it running")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
