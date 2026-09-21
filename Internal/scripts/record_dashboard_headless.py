#!/usr/bin/env python3
"""Record the dashboard exactly as a browser shows it -- without an X server.

    .venv/bin/python scripts/record_dashboard_headless.py --while-pid <pipeline pid>
    .venv/bin/python scripts/record_dashboard_headless.py --duration-s 60

Why this exists alongside record_dashboard.py
---------------------------------------------
record_dashboard.py needs a hidden Xorg (xrdp's `xrdpdev` driver), Firefox, and
GStreamer's Python bindings.  On this Jetson none of the three is usable:
`/etc/X11/xrdp/xorg.conf` does not exist, the only live display belongs to
another user, and `gi` is not importable from the project venv.  Installing any
of that needs root, which this account does not have.

Playwright's Chromium needs none of it -- it renders headlessly, off-screen --
so this captures the same page the same way and encodes with ffmpeg.  It is the
SAME artifact by the project's rule: the browser page itself, nothing drawn on
top, never annotated camera frames.

Output is `.mp4` (fragmented), and it writes the same `<recording>.json` sidecar
that scripts/clip_wrong_orders.py reads, so the wrong-order cuts land on the
wall clock exactly as they do for the Xorg recorder.

Capture rate: screenshotting this page measured 8.5 fps (JPEG q85) on this box,
so the 5 fps default has headroom and the loop paces on absolute deadlines
rather than drifting.  If the measured rate still ends up off target the sidecar
says so and a warning is logged, because a video whose seconds are not wall
seconds would silently misplace every clip.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def ffmpeg_bin() -> str:
    local = Path.home() / ".local" / "bin" / "ffmpeg"
    return str(local) if local.exists() else "ffmpeg"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000/")
    ap.add_argument("--out", help="output .mp4 (default: output/recordings/dashboard_<time>.mp4)")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=int, default=8,
                    help="CAPTURE rate. Screenshotting this page measured 8.5 fps, "
                         "so 8 is the practical ceiling; the file is still --out-fps.")
    ap.add_argument("--out-fps", type=int, default=30,
                    help="frame rate of the written file; ffmpeg duplicates frames "
                         "to reach it, which costs almost no bitrate on a static page")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality fed to the encoder")
    ap.add_argument("--crf", type=int, default=20, help="x264 quality (lower = better)")
    ap.add_argument("--max-bytes", type=float, default=1_000_000_000,
                    help="size ceiling; the bitrate cap is derived from this and "
                         "the expected duration so the file lands under it")
    ap.add_argument("--expect-s", type=float, default=0.0,
                    help="expected run length in seconds, used with --max-bytes to "
                         "pick the bitrate cap; 0 disables the cap")
    ap.add_argument("--while-pid", type=int, help="record until this process exits")
    ap.add_argument("--duration-s", type=float, help="record for this many seconds")
    ap.add_argument("--settle-s", type=float, default=10, help="page load time before capture starts")
    args = ap.parse_args(argv)

    width, height = (int(v) for v in args.size.lower().split("x"))
    out = Path(args.out) if args.out else ROOT / "output" / "recordings" / f"dashboard_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    frame_interval = 1.0 / args.fps
    frames = 0

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--hide-scrollbars"],
        )
        page = browser.new_page(viewport={"width": width, "height": height})
        log(f"opening {args.url}")
        page.goto(args.url, wait_until="domcontentloaded", timeout=120_000)
        time.sleep(args.settle_s)

        # Constant-rate input: every JPEG is one frame of 1/fps, so the video's
        # seconds are wall seconds as long as the loop keeps its deadlines.
        # `-r out_fps` on the OUTPUT side duplicates frames up to 30 without
        # touching duration, so the wall-clock mapping the wrong-order clips
        # depend on survives, and duplicates cost almost nothing in h264.
        cmd = [
            ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "image2pipe", "-vcodec", "mjpeg", "-framerate", str(args.fps), "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(args.crf),
            "-r", str(args.out_fps),
            "-pix_fmt", "yuv420p",
        ]
        # A CRF encode has no size ceiling, and this file has to stay under
        # --max-bytes.  Capping the rate turns CRF into "this quality, unless it
        # would blow the budget", which is what is wanted: a mostly-static
        # dashboard spends almost nothing until something moves.
        if args.expect_s > 0 and args.max_bytes > 0:
            cap_kbps = max(200, int((args.max_bytes * 8 / args.expect_s) / 1000 * 0.92))
            cmd += ["-maxrate", "%dk" % cap_kbps, "-bufsize", "%dk" % (cap_kbps * 2)]
            log(f"size budget {args.max_bytes/1e9:.2f} GB over ~{args.expect_s:.0f} s "
                f"-> bitrate cap {cap_kbps} kbps")
        cmd += [
            # Fragmented, for the same reason the GStreamer recorder fragments:
            # an unclean exit still leaves everything up to the last fragment
            # playable, instead of a file with no moov atom and no content.
            "-movflags", "+frag_keyframe+empty_moov",
            str(out),
        ]
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        log(f"recording -> {out} (capture {args.fps} fps, file {args.out_fps} fps, "
            f"jpeg q{args.quality}, crf {args.crf})")

        started = time.time()
        next_tick = time.monotonic()
        overruns = 0
        try:
            while True:
                if args.while_pid and not pid_alive(args.while_pid):
                    log(f"process {args.while_pid} exited")
                    break
                if args.duration_s and time.time() - started >= args.duration_s:
                    break
                try:
                    enc.stdin.write(page.screenshot(type="jpeg", quality=args.quality))
                except (BrokenPipeError, OSError):
                    log("encoder closed the pipe")
                    break
                frames += 1
                next_tick += frame_interval
                slack = next_tick - time.monotonic()
                if slack > 0:
                    time.sleep(slack)
                else:
                    # Behind schedule: take the next frame immediately and reset
                    # the deadline, so one slow screenshot cannot compound.
                    overruns += 1
                    next_tick = time.monotonic()
        except KeyboardInterrupt:
            log("interrupted")
        finally:
            ended = time.time()
            try:
                enc.stdin.close()
            except OSError:
                pass
            enc.wait(timeout=120)
            browser.close()

    wall_s = ended - started
    measured = frames / wall_s if wall_s else 0.0
    drift = abs(measured - args.fps) / args.fps if args.fps else 0.0
    log(f"recording closed after {wall_s:.0f} s -> {out} "
        f"({frames} frames, {measured:.2f} fps measured, {overruns} overruns)")
    if drift > 0.05:
        log(f"WARNING: captured {measured:.2f} fps against a {args.fps} fps target "
            f"({drift * 100:.0f}% off). Video seconds are not wall seconds, so "
            f"wrong-order clips cut from this file will be displaced.")

    size_b = out.stat().st_size if out.exists() else 0
    log(f"file size {size_b/1e6:.0f} MB")
    if args.max_bytes and size_b > args.max_bytes:
        log(f"WARNING: {size_b/1e9:.2f} GB exceeds the {args.max_bytes/1e9:.2f} GB budget")

    out.with_suffix(".json").write_text(json.dumps({
        "file": out.name,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "wall_s": round(wall_s, 1),
        # `fps` stays the CAPTURE rate because that is what wall_s/frames
        # describes; the container runs at out_fps via duplicated frames.
        "fps": args.fps,
        "out_fps": args.out_fps,
        "measured_fps": round(measured, 3),
        "frames": frames,
        "bytes": size_b,
        "size": [width, height],
        "url": args.url,
        "recorder": "playwright-chromium-headless",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
