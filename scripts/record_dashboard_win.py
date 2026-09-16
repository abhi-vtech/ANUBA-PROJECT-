#!/usr/bin/env python3
"""Record the dashboard exactly as a browser shows it, on Windows.

    python scripts/record_dashboard_win.py --while-pid <pipeline pid>
    python scripts/record_dashboard_win.py --duration-s 60          # a fixed length

The Jetson recorder (``record_dashboard.py``) drives a hidden Xorg, Firefox in
kiosk mode and GStreamer into the board's hardware H.264 encoder.  None of that
exists on Windows -- no Xorg, no ``gi``, no ``nvv4l2h264enc`` -- so this does
the same job with the pieces that are here: Playwright driving the installed
Chrome, and frames taken as screenshots.

Same contract as the Jetson recorder on purpose -- same flags, same sidecar --
so ``src/main.py`` and ``scripts/clip_wrong_orders.py`` do not have to know
which one ran:

    <out>        the recording, H.264-free MKV at OUT_FPS
    <out>.json   {"started": <ISO>, "wall_s": <float>}  -- the wall clock the
                 journeys' opened_wall/verdict_wall are cut against

Screenshots rather than Chromium's screencast, which Playwright exposes as
``record_video_dir``.  On this dashboard the screencast yields about one second
of video per run however long the run is -- measured at 1.04s for 20s of wall
time, headless and headed alike -- because it emits a frame only when the
compositor repaints, and the page's motion is an MJPEG ``<img>`` that does not
drive one.  A screenshot always renders, so the capture rate is ours to choose.

The file is written frame by frame as the run goes, so stopping the recorder
badly costs the last moment rather than the whole recording -- which is what a
screencast, only written when the browser context closes, would cost.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

#: Playback rate of the finished file.  The cameras record at 30 and the
#: annotated recorder writes 30; the clips are reviewed alongside both.
OUT_FPS = 30

#: How often a screenshot is actually taken.  A 1920x1080 JPEG costs ~50ms, so
#: 10/s leaves the browser idle half the time and barely touches the pipeline.
#: Every output frame between two screenshots repeats the earlier one, which
#: costs almost nothing to encode and keeps video time equal to wall time.
DEFAULT_CAPTURE_FPS = 10.0

_stop = False


def log(msg: str) -> None:
    print("[dash-rec] " + msg, flush=True)


def _pid_alive(pid: int) -> bool:
    """Whether the pipeline is still running.

    No signal 0 on Windows, so ask the OS directly; an exit code that is not
    STILL_ACTIVE means it has gone.
    """
    import ctypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    k = ctypes.windll.kernel32
    handle = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not k.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        k.CloseHandle(handle)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/")
    ap.add_argument("--out", required=True, help="the recording to write")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--while-pid", type=int, default=0,
                    help="record until this process exits")
    ap.add_argument("--duration-s", type=float, default=0.0)
    ap.add_argument("--settle-s", type=float, default=3.0,
                    help="let the page render before the clock starts")
    ap.add_argument("--capture-fps", type=float, default=DEFAULT_CAPTURE_FPS)
    args = ap.parse_args(argv)

    try:
        width, height = (int(v) for v in args.size.lower().split("x"))
    except ValueError:
        log("--size must look like 1920x1080"); return 2
    if not args.while_pid and not args.duration_s:
        log("need --while-pid or --duration-s, or this would never stop")
        return 2

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log("playwright is not installed: uv pip install playwright")
        return 1

    def _on_signal(_sig, _frm):
        global _stop
        _stop = True
    for _name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        _sig = getattr(signal, _name, None)
        if _sig is not None:
            try:
                signal.signal(_sig, _on_signal)
            except (ValueError, OSError):
                pass

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # The pipeline asks this process to stop by creating this file.  It cannot
    # ask with a console event: those reach every process in the group, the
    # Playwright driver included, and a dead driver takes the browser with it.
    stop_file = out.with_suffix(".stop")
    try:
        stop_file.unlink()
    except OSError:
        pass

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(OUT_FPS), (width, height))
    if not writer.isOpened():
        log("could not open %s for writing" % out); return 1

    interval = 1.0 / max(1.0, args.capture_fps)
    started_at = None
    frames = 0
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel="chrome", headless=True)
            page = browser.new_page(viewport={"width": width, "height": height})
            log("opening %s at %dx%d" % (args.url, width, height))
            try:
                page.goto(args.url, wait_until="domcontentloaded", timeout=120_000)
            except Exception as exc:
                log("could not open the dashboard: %s" % str(exc)[:160])
                browser.close(); return 1

            # Let the panels and the first video frame arrive before the clock
            # starts, so the sidecar's start is a moment the recording shows.
            time.sleep(max(0.0, args.settle_s))

            t0 = time.time()
            started_at = dt.datetime.now()
            log("recording from %s at %.0f fps capture -> %d fps file"
                % (started_at.isoformat(timespec="seconds"), args.capture_fps, OUT_FPS))

            deadline = (t0 + args.duration_s) if args.duration_s else None
            last_note = t0
            while not _stop:
                if deadline and time.time() >= deadline:
                    log("reached --duration-s"); break
                if stop_file.exists():
                    log("asked to stop by the pipeline"); break
                if args.while_pid and not _pid_alive(args.while_pid):
                    log("pipeline %d has gone" % args.while_pid); break

                tick = time.time()
                try:
                    buf = page.screenshot(type="jpeg", quality=75)
                except Exception as exc:
                    log("screenshot failed (%s); stopping" % str(exc)[:90])
                    break
                img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                if img.shape[1] != width or img.shape[0] != height:
                    img = cv2.resize(img, (width, height))

                # Hold each screenshot until the file has caught up with the
                # wall clock.  Written this way round -- against elapsed time
                # rather than a fixed count per shot -- the video stays the
                # same length as the run even when a screenshot came late.
                target = int((tick - t0) * OUT_FPS)
                while frames < target:
                    writer.write(img)
                    frames += 1

                if tick - last_note >= 300:
                    log("%.0f min recorded" % ((tick - t0) / 60.0))
                    last_note = tick

                slack = interval - (time.time() - tick)
                if slack > 0:
                    time.sleep(slack)

            # Pad out to the moment we stopped, so the file covers every second
            # the sidecar claims it does.
            if started_at is not None:
                final = int((time.time() - t0) * OUT_FPS)
                if frames and final > frames:
                    last = img if img is not None else None
                    while last is not None and frames < final:
                        writer.write(last)
                        frames += 1
            browser.close()
    finally:
        writer.release()
        try:
            stop_file.unlink()
        except OSError:
            pass

    if not frames or started_at is None:
        log("no frames were captured"); return 1

    wall_s = frames / float(OUT_FPS)
    out.with_suffix(".json").write_text(json.dumps({
        "started": started_at.isoformat(),
        "wall_s": round(wall_s, 3),
        "url": args.url,
        "size": "%dx%d" % (width, height),
        "fps": OUT_FPS,
        "capture_fps": args.capture_fps,
        "recorder": "playwright-chrome-screenshots",
    }, indent=2), encoding="utf-8")
    log("wrote %s (%.0fs, %d frames, %.1f MB) and its sidecar"
        % (out.name, wall_s, frames, out.stat().st_size / 1048576.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
