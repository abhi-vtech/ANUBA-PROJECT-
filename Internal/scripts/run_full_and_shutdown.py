"""Process the full hour of KDS + kitchen video unattended, then shut down.

    python scripts/run_full_and_shutdown.py

Three jobs:

1. **Keep the machine awake** for the whole run.  Windows is told the work is
   in progress via ``SetThreadExecutionState``, which is what a media player or
   installer uses.  That is deliberately *not* a mouse jiggler -- it keeps the
   machine from sleeping without stealing the cursor, so the laptop stays
   usable while the run is going.

2. **Run the dual pipeline to completion** (``run_kds_dual.py --exit-on-end``),
   streaming its output to ``output/full_run_<stamp>.log`` and printing a
   progress line every minute.

3. **Shut down afterwards** -- but only after an abort window, and only if the
   run actually succeeded.  A crashed or interrupted run leaves the machine on
   so the logs can be read.

Aborting
--------
* ``Ctrl+C`` at any time stops the run and cancels the shutdown.
* Once the shutdown is scheduled you still have ``--shutdown-delay`` seconds
  (default 120).  Cancel it from any terminal with::

      shutdown /a

Options
-------
``--no-shutdown``        run to completion, then just stop (no shutdown)
``--shutdown-delay N``   seconds of abort window (default 120)
``--idle-detect-stride`` passed through to the pipeline
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# SetThreadExecutionState flags (winbase.h)
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


def on_battery() -> bool:
    """True when the machine is running on battery.

    A multi-hour GPU run on battery will flatten the laptop long before it
    finishes, so this is worth refusing rather than discovering at 4am.
    """
    if os.name != "nt":
        return False
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Battery).BatteryStatus"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()
        # 1 = discharging; 2 = on AC; 3 = fully charged on AC. No battery = desktop.
        return bool(out) and out[0].strip() == "1"
    except Exception:
        return False


def _standby_timeout_dc() -> str:
    """Current battery standby timeout, in minutes, as text ('' if unknown)."""
    try:
        out = subprocess.run(
            ["powercfg", "/query", "SCHEME_CURRENT", "SUB_SLEEP", "STANDBYIDLE"],
            capture_output=True, text=True, timeout=20,
        ).stdout
        for line in out.splitlines():
            if "Current DC Power Setting Index" in line:
                return str(int(line.split(":")[1].strip(), 16) // 60)
    except Exception:
        pass
    return ""


class KeepAwake:
    """Stop Windows sleeping for the length of the run.

    Two mechanisms, because one is not enough on a modern laptop:

    * ``SetThreadExecutionState`` tells Windows work is in progress.  On a
      **Modern Standby (S0ix)** machine -- which most recent laptops are --
      this does NOT stop the idle timer putting the system into standby, so on
      its own it is not sufficient.  It is still set, for S3 machines.
    * The power-scheme **idle timeouts** are set to 0 for the duration and
      restored on exit.  This is what actually holds a Modern Standby machine
      awake.

    Neither is a cursor jiggler: the machine stays usable while the run goes.
    """

    def __init__(self, keep_display_on: bool = False):
        self.keep_display_on = keep_display_on
        self.active = False
        self._saved_dc = ""

    def __enter__(self):
        if os.name != "nt":
            print("keep-awake: not Windows, skipping")
            return self

        flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        if self.keep_display_on:
            flags |= ES_DISPLAY_REQUIRED
        self.active = ctypes.windll.kernel32.SetThreadExecutionState(flags) != 0

        # The part that actually works under Modern Standby.
        self._saved_dc = _standby_timeout_dc()
        for scope in ("standby-timeout-ac", "standby-timeout-dc",
                      "hibernate-timeout-ac", "hibernate-timeout-dc"):
            subprocess.run(["powercfg", "/change", scope, "0"],
                           capture_output=True, timeout=20)
        print(
            "keep-awake: execution-state %s, sleep timeouts disabled "
            "(battery standby was %s min, restored on exit)"
            % ("set" if self.active else "FAILED", self._saved_dc or "?")
        )
        return self

    def __exit__(self, *exc):
        if os.name != "nt":
            return False
        if self.active:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        # Put the battery timeout back so the laptop is not left unable to sleep.
        if self._saved_dc:
            subprocess.run(["powercfg", "/change", "standby-timeout-dc", self._saved_dc],
                           capture_output=True, timeout=20)
        print("keep-awake: released (battery standby restored to %s min)"
              % (self._saved_dc or "?"))
        return False


def find_default(folder: Path):
    suffixes = (".mkv", ".mp4", ".avi", ".mov", ".m4v")
    if not folder.is_dir():
        return None
    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in suffixes)
    return videos[0] if videos else None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the full dual-video pipeline unattended, then shut down.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Cancel a scheduled shutdown with:  shutdown /a",
    )
    parser.add_argument("--kds", default=None, help="KDS screen video (default: videos/KDS_Feed/)")
    parser.add_argument("--production", default=None, help="kitchen video (default: videos/KDS/)")
    parser.add_argument(
        "--no-shutdown",
        action="store_true",
        help="process everything but leave the machine running.",
    )
    parser.add_argument(
        "--shutdown-delay",
        type=int,
        default=120,
        help="seconds between the run finishing and the shutdown (default 120).",
    )
    parser.add_argument(
        "--idle-detect-stride",
        type=int,
        default=None,
        help="detection stride while the KDS has no ticket (passed through).",
    )
    parser.add_argument(
        "--allow-battery",
        action="store_true",
        help="run even if the laptop is on battery (it will very likely go flat "
             "or sleep before a multi-hour job finishes).",
    )
    parser.add_argument(
        "--keep-display-on",
        action="store_true",
        help="also stop the screen blanking (default: screen may sleep).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    kds = Path(args.kds) if args.kds else find_default(ROOT / "videos" / "KDS_Feed")
    production = (
        Path(args.production) if args.production else find_default(ROOT / "videos" / "KDS")
    )
    for label, path in (("KDS", kds), ("production", production)):
        if path is None or not path.exists():
            print("error: no %s video found (%s)" % (label, path), file=sys.stderr)
            return 2

    if on_battery() and not args.allow_battery:
        # A full run is hours of GPU work.  On battery the machine will either
        # go flat or hit the battery idle timer -- a held wake request does not
        # stop Modern Standby.  Refuse rather than fail silently overnight.
        print(
            "error: this laptop is on BATTERY. A full run is a couple of hours "
            "of GPU work, so the battery will go flat and Windows will sleep on "
            "the battery idle timer. Plug the charger in, or pass --allow-battery "
            "to override.",
            file=sys.stderr,
        )
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "output"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / ("full_run_%s.log" % stamp)

    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "run_kds_dual.py"),
        "--kds",
        str(kds),
        "--production",
        str(production),
        "--exit-on-end",
        "--log-level",
        "INFO",
    ]
    if args.idle_detect_stride is not None:
        cmd += ["--idle-detect-stride", str(args.idle_detect_stride)]

    print("=" * 70)
    print("FULL DUAL-VIDEO RUN")
    print("=" * 70)
    print("KDS        : %s" % kds.name)
    print("production : %s" % production.name)
    print("log        : %s" % log_path)
    print("dashboard  : http://localhost:8000")
    print(
        "after      : %s"
        % ("no shutdown (--no-shutdown)" if args.no_shutdown
           else "shutdown, %ds after the run ends" % args.shutdown_delay)
    )
    print("abort      : Ctrl+C now, or `shutdown /a` once it is scheduled")
    print("=" * 70)
    print()

    started = time.time()
    code = 1
    try:
        with KeepAwake(args.keep_display_on):
            code = _run(cmd, log_path, started)
    except KeyboardInterrupt:
        print("\ninterrupted - no shutdown will be scheduled")
        return 130

    elapsed = timedelta(seconds=int(time.time() - started))
    print("\nrun finished in %s with exit code %d" % (elapsed, code))
    _summarise(log_path)

    if args.no_shutdown:
        print("\n--no-shutdown given; leaving the machine running.")
        return code
    if code != 0:
        # A failed run is exactly when the logs matter most, so stay on.
        print("\nRun did NOT finish cleanly (exit %d) - NOT shutting down." % code)
        print("Read %s before re-running." % log_path)
        return code

    return _schedule_shutdown(args.shutdown_delay)


def _run(cmd, log_path: Path, started: float) -> int:
    """Run the pipeline, tee its output to a log, print a heartbeat."""
    frame_re = re.compile(r'"frame_count":\s*(\d+)')
    fps_re = re.compile(r'"fps":\s*([\d.]+)')
    last_beat = 0.0
    last_frame = 0
    last_fps = 0.0

    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )
        try:
            for line in process.stdout:
                log.write(line)
                match = frame_re.search(line)
                if match:
                    last_frame = int(match.group(1))
                fps_match = fps_re.search(line)
                if fps_match:
                    last_fps = float(fps_match.group(1))

                # Surface the events that matter, plus a minute heartbeat.
                if any(
                    key in line
                    for key in (
                        "Payment = PAID",
                        "created (OG-",
                        "WRONG ORDER",
                        "ORDER COMPLETED",
                        "Video ended",
                        "Traceback",
                        "already in use",
                    )
                ):
                    print("  " + line.rstrip()[-160:])

                now = time.time()
                if now - last_beat >= 60:
                    last_beat = now
                    print(
                        "  [%s] frame %d, %.1f fps" % (
                            timedelta(seconds=int(now - started)),
                            last_frame,
                            last_fps,
                        )
                    )
        except KeyboardInterrupt:
            process.terminate()
            raise
        finally:
            log.flush()
        return process.wait()


def _summarise(log_path: Path) -> None:
    """Print what the run concluded, straight from its log."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    created = len(re.findall(r"created \(OG-", text))
    wrong = len(re.findall(r"WRONG ORDER", text))
    correct = len(re.findall(r"ORDER COMPLETED", text))
    kept = re.search(r"Failure clips kept: (\d+), discarded \(order correct\): (\d+)", text)
    print("  tickets created : %d" % created)
    print("  correct         : %d" % correct)
    print("  wrong           : %d" % wrong)
    if kept:
        print("  clips kept      : %s   discarded: %s" % (kept.group(1), kept.group(2)))
    print("  failure clips   : %s" % (ROOT / "output" / "failures"))


def _schedule_shutdown(delay: int) -> int:
    if os.name != "nt":
        print("shutdown is only wired up for Windows; leaving the machine on.")
        return 0
    when = datetime.now() + timedelta(seconds=delay)
    print("\n" + "!" * 70)
    print("SHUTTING DOWN at %s  (in %d seconds)" % (when.strftime("%H:%M:%S"), delay))
    print("Cancel it with:   shutdown /a")
    print("!" * 70)
    result = subprocess.run(
        ["shutdown", "/s", "/t", str(delay), "/c", "Order-Accuracy run finished"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("could not schedule the shutdown: %s" % result.stderr.strip())
        return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
