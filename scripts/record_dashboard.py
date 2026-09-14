#!/usr/bin/env python3
"""Record the dashboard exactly as a browser shows it.

    uv run python scripts/record_dashboard.py --while-pid <pipeline pid>
    uv run python scripts/record_dashboard.py --duration-s 60          # a fixed length

The Jetson is used over SSH with no screen, so this:

1. starts a hidden X display -- the real Xorg binary with xrdp's in-memory
   `xrdpdev` driver, since the setuid Xorg wrapper only allows console users;
2. opens Firefox in kiosk mode on the dashboard;
3. captures the display with GStreamer `ximagesrc` straight into the Jetson's
   hardware H.264 encoder (nvv4l2h264enc), writing an MKV.

The capture follows the wall clock.  When the pipeline runs slower than real
time the recording is longer than the camera footage; a camera-speed copy can
be made afterwards without re-encoding (see the sidecar .json).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

Gst.init(None)

ROOT = Path(__file__).resolve().parents[1]
XORG = "/usr/lib/xorg/Xorg"
XRDP_CONF = "/etc/X11/xrdp/xorg.conf"

FIREFOX_PREFS = {
    "browser.shell.checkDefaultBrowser": False,
    "browser.startup.homepage_override.mstone": "ignore",
    "startup.homepage_welcome_url": "",
    "browser.aboutwelcome.enabled": False,
    "trailhead.firstrun.didSeeAboutWelcome": True,
    "datareporting.policy.dataSubmissionEnabled": False,
    "datareporting.policy.firstRunURL": "",
    "toolkit.telemetry.reportingpolicy.firstRun": False,
    "browser.sessionstore.resume_from_crash": False,
    "browser.translations.automaticallyPopup": False,
    "app.update.auto": False,
    "layout.css.devPixelsPerPx": "1.0",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _not_self(path: str) -> str:
    """A pkill/pgrep pattern for `path` that cannot match the pkill command itself."""
    escaped = re.escape(path)
    return f"[{escaped[0]}]{escaped[1:]}"


def screen_size(display: str) -> tuple:
    out = subprocess.run(["xwininfo", "-root"], env={**os.environ, "DISPLAY": display},
                         capture_output=True, text=True).stdout
    w, h = re.search(r"Width:\s+(\d+)", out), re.search(r"Height:\s+(\d+)", out)
    return (int(w.group(1)), int(h.group(1))) if w and h else (0, 0)


def start_display(display: str, width: int, height: int, log_dir: Path) -> subprocess.Popen:
    number = display.lstrip(":")
    socket_path = Path(f"/tmp/.X11-unix/X{number}")
    if socket_path.exists():
        raise SystemExit(f"display {display} is already in use ({socket_path}); pass --display with another number")
    env = {**os.environ, "XRDP_START_WIDTH": str(width), "XRDP_START_HEIGHT": str(height)}
    proc = subprocess.Popen(
        [XORG, display, "-config", XRDP_CONF, "-noreset", "-nolisten", "tcp",
         "-logfile", str(log_dir / f"xorg{number}.log")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env, start_new_session=True,
    )
    for _ in range(80):
        if socket_path.exists():
            break
        if proc.poll() is not None:
            raise SystemExit(f"Xorg exited with code {proc.returncode}; see {log_dir}/xorg{number}.log")
        time.sleep(0.25)
    else:
        proc.terminate()
        raise SystemExit("Xorg did not create its display socket within 20 s")
    time.sleep(1.0)
    if screen_size(display) != (width, height):
        subprocess.run(["xrandr", "-s", f"{width}x{height}"], env={**os.environ, "DISPLAY": display},
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)
    return proc


def wait_for_dashboard(url: str, timeout_s: float, need_analysis: bool) -> bool:
    api = url.rstrip("/") + "/api/analysis"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if need_analysis:
                with urllib.request.urlopen(api, timeout=3) as resp:
                    if json.load(resp).get("analysis"):
                        return True
            else:
                with urllib.request.urlopen(url, timeout=3) as resp:
                    if resp.status == 200:
                        return True
        except Exception:
            pass
        time.sleep(2)
    return False


def start_firefox(display: str, url: str, width: int, height: int, profile: Path, log_path: Path) -> subprocess.Popen:
    profile.mkdir(parents=True, exist_ok=True)
    (profile / "user.js").write_text(
        "".join(f'user_pref("{k}", {json.dumps(v)});\n' for k, v in FIREFOX_PREFS.items())
    )
    for lock in ("lock", ".parentlock"):
        try:
            (profile / lock).unlink()
        except FileNotFoundError:
            pass
    env = {**os.environ, "DISPLAY": display, "MOZ_ENABLE_WAYLAND": "0"}
    return subprocess.Popen(
        ["firefox", "--kiosk", "--new-instance", "--no-remote", "--profile", str(profile),
         "--width", str(width), "--height", str(height), url],
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True,
    )


class ScreenRecorder:
    def __init__(self, display: str, out: Path, fps: int, bitrate: int):
        location = str(out).replace("\\", "\\\\").replace('"', '\\"')
        self.pipeline = Gst.parse_launch(
            f"ximagesrc display-name={display} use-damage=false show-pointer=false ! "
            f"video/x-raw,framerate={fps}/1 ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
            f"nvv4l2h264enc bitrate={bitrate} iframeinterval={fps * 2} ! h264parse ! matroskamux ! "
            f'filesink location="{location}"'
        )
        self.bus = self.pipeline.get_bus()

    def start(self) -> None:
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("screen capture pipeline did not start")

    def error(self):
        msg = self.bus.pop_filtered(Gst.MessageType.ERROR)
        if msg is None:
            return None
        err, debug = msg.parse_error()
        return f"{err.message} ({debug})"

    def stop(self) -> None:
        # EOS lets matroskamux write its index, so the file is complete.
        self.pipeline.send_event(Gst.Event.new_eos())
        self.bus.timed_pop_filtered(30 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
        self.pipeline.set_state(Gst.State.NULL)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stop_process_group(proc, pattern: str = None) -> None:
    if proc is not None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    if pattern:
        subprocess.run(["pkill", "-f", pattern], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    if proc is not None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if pattern:
        subprocess.run(["pkill", "-9", "-f", pattern], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000/")
    ap.add_argument("--out", help="output .mkv (default: output/recordings/dashboard_<time>.mkv)")
    ap.add_argument("--display", default=":99")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--bitrate", type=int, default=8_000_000)
    ap.add_argument("--while-pid", type=int, help="record until this process exits")
    ap.add_argument("--duration-s", type=float, help="record for this many seconds")
    ap.add_argument("--wait-s", type=float, default=1800, help="how long to wait for the dashboard to have data")
    ap.add_argument("--settle-s", type=float, default=12, help="page load time before capture starts")
    ap.add_argument("--profile", default=str(ROOT / "bench" / "firefox_dashboard_profile"))
    args = ap.parse_args(argv)

    width, height = (int(v) for v in args.size.lower().split("x"))
    out = Path(args.out) if args.out else ROOT / "output" / "recordings" / f"dashboard_{time.strftime('%Y%m%d_%H%M%S')}.mkv"
    out.parent.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "bench" / "logs"  # Xorg and Firefox logs, kept out of output/
    log_dir.mkdir(parents=True, exist_ok=True)
    profile = Path(args.profile)

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    xorg = firefox = recorder = None
    started = ended = None
    try:
        xorg = start_display(args.display, width, height, log_dir)
        actual = screen_size(args.display)
        log(f"hidden display {args.display} running at {actual[0]}x{actual[1]}")
        log(f"waiting for dashboard data at {args.url}")
        if not wait_for_dashboard(args.url, args.wait_s, need_analysis=True):
            log("the dashboard never published analysis data; nothing recorded")
            return 1
        firefox = start_firefox(args.display, args.url, width, height, profile, log_dir / "firefox_dashboard.log")
        time.sleep(args.settle_s)
        recorder = ScreenRecorder(args.display, out, args.fps, args.bitrate)
        recorder.start()
        started = time.time()
        log(f"recording {actual[0]}x{actual[1]} @ {args.fps} fps -> {out}")
        while not stop.is_set():
            if args.while_pid and not pid_alive(args.while_pid):
                log(f"process {args.while_pid} exited")
                break
            if args.duration_s and time.time() - started >= args.duration_s:
                break
            err = recorder.error()
            if err:
                log(f"capture error: {err}")
                break
            time.sleep(1.0)
    finally:
        if recorder is not None:
            ended = time.time()
            recorder.stop()
            log(f"recording closed after {ended - started:.0f} s -> {out}")
            out.with_suffix(".json").write_text(json.dumps({
                "file": out.name,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
                "wall_s": round(ended - started, 1),
                "fps": args.fps,
                "size": list(screen_size(args.display)) if xorg is not None else [width, height],
                "url": args.url,
            }, indent=2))
        stop_process_group(firefox, _not_self(str(profile)))
        if xorg is not None:
            stop_process_group(xorg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
