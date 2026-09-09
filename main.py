"""Run the Order-Accuracy pipeline.

    python main.py

No arguments, no flags.  Everything is read from ``config/model.yaml``:
which video to process, which model to load, and how the KDS is fed.

The dashboard opens by itself once the server is actually listening, which
takes a few seconds while the YOLO weights load.  Set OPEN_BROWSER=0 to stop
it opening, or DASHBOARD_PORT=<n> to serve somewhere other than 8000.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PORT = int(os.environ.get("DASHBOARD_PORT", "8000"))
URL = "http://localhost:%d" % PORT


def _port_is_taken() -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("0.0.0.0", PORT))
        return False
    except OSError:
        return True
    finally:
        probe.close()


def _open_browser_when_ready(timeout_s: float = 300.0) -> None:
    """Open the dashboard once the port answers.

    Opening it immediately would land on a connection error, because the
    server only binds after the process has started up.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=0.5):
                pass
        except OSError:
            time.sleep(0.5)
            continue
        print("Dashboard is up: %s" % URL, flush=True)
        try:
            webbrowser.open(URL)
        except Exception:
            print("Could not open a browser; open %s yourself." % URL, flush=True)
        return
    print("Server did not come up within %.0fs." % timeout_s, flush=True)


def _preflight() -> list:
    """Check the files the run needs before starting anything.

    A missing source video is otherwise completely silent: OpenCV opens
    nothing, no frame ever arrives, and the pipeline spins with the dashboard
    stuck on its loading frame -- which looks exactly like a model that will
    not load.
    """
    problems = []
    try:
        import yaml

        with open(os.path.join(ROOT, "config", "model.yaml"), encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception as exc:                     # pragma: no cover - config is read again downstream
        return ["could not read config/model.yaml: %s" % exc]

    source = os.environ.get("VIDEO_SOURCE", cfg.get("source"))
    # A bare number is a webcam index and a URL is a stream; only check paths.
    if isinstance(source, str) and not source.isdigit() and "://" not in source:
        path = source if os.path.isabs(source) else os.path.join(ROOT, source)
        if not os.path.exists(path):
            problems.append(
                "source video not found: %s\n"
                "      (set `source:` in config/model.yaml, or VIDEO_SOURCE=<path>)"
                % source
            )

    model = cfg.get("model_path")
    if isinstance(model, str):
        path = model if os.path.isabs(model) else os.path.join(ROOT, model)
        if not os.path.exists(path):
            problems.append("model weights not found: %s" % model)

    return problems


def main() -> int:
    problems = _preflight()
    if problems:
        print("Cannot start:")
        for problem in problems:
            print("  - %s" % problem)
        return 1

    if _port_is_taken():
        print(
            "Port %d is already in use, so this run has nowhere to serve.\n"
            "Another pipeline is probably still going -- what you see at %s\n"
            "would be that older process, not this one.\n"
            "\n"
            "Find and stop it:\n"
            "    netstat -ano | findstr :%d\n"
            "    taskkill /PID <pid> /T /F\n"
            "\n"
            "Or serve elsewhere:  set DASHBOARD_PORT=8001 && python main.py"
            % (PORT, URL, PORT)
        )
        return 1

    print("=" * 66)
    print("Order-Accuracy pipeline")
    print("  dashboard : %s  (opens automatically once ready)" % URL)
    print("  config    : config/model.yaml")
    print("  stop with : Ctrl+C")
    print("=" * 66, flush=True)

    # Ingredient attribution requires a completed well trip.  TRIP_DEBUG=1
    # surfaces the two lines that show it working:
    #   'Trajectory pending pick' -- hand entered an ingredient well
    #   'TRAJECTORY CONFIRM'      -- hand came back to a hotdog; item credited
    # Off by default: one of those fires every frame while a pick is in
    # transit, which floods the console and slows the loop.
    if os.environ.get("TRIP_DEBUG") == "1":
        logging.getLogger("src.temporal").setLevel(logging.DEBUG)
        print("  TRIP_DEBUG on: ingredient trip logging enabled", flush=True)

    if os.environ.get("OPEN_BROWSER", "1") != "0":
        threading.Thread(target=_open_browser_when_ready, daemon=True).start()
    else:
        print("  OPEN_BROWSER=0: open %s yourself" % URL, flush=True)

    from src.main import main as run_pipeline

    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
