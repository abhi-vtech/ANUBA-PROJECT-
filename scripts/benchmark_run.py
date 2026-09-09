"""Benchmark the pipeline over a full-length video and write a report.

    python scripts/benchmark_run.py --video <path>

Answers one question: can this machine process an hour of footage
continuously, and at what cost.  Written to size the Jetson deployment, so it
records what changes on different hardware -- throughput, per-stage latency,
CPU, RAM and GPU -- rather than anything about order accuracy.

Samples the pipeline's own ``src.metrics`` output for timing, and nvidia-smi
plus the process tree for resources, then writes a Markdown report to
``output/benchmark_<stamp>.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

METRIC_RE = re.compile(r'\{"event": "metrics".*?\}')


def _percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * pct / 100.0))
    return ordered[idx]


def _gpu_sample():
    """Return (util_pct, mem_used_mb, total_mb) or None when unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=8,
        ).decode().strip().splitlines()[0]
        util, used, total = [int(p.strip()) for p in out.split(",")]
        return util, used, total
    except Exception:
        return None


def _proc_sample(pid):
    """CPU percent and working set for the process tree, via PowerShell."""
    script = (
        "$ids=@(%d); $all=@(); "
        "function walk($i){ $all+=$i; Get-CimInstance Win32_Process -Filter \"ParentProcessId=$i\" | "
        "ForEach-Object { walk $_.ProcessId } } "
        "foreach($i in $ids){ walk $i } "
        "$ps = $all | ForEach-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }; "
        "if($ps){ '{0},{1}' -f (($ps | Measure-Object WorkingSet64 -Sum).Sum/1MB), "
        "(($ps | Measure-Object CPU -Sum).Sum) }"
    ) % pid
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", script],
            stderr=subprocess.DEVNULL, timeout=20,
        ).decode().strip()
        if not out:
            return None
        mem_mb, cpu_s = out.split(",")
        return float(mem_mb), float(cpu_s)
    except Exception:
        return None


class Sampler(threading.Thread):
    """Polls resources while the pipeline runs."""

    def __init__(self, pid, interval=15.0):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.stop_flag = threading.Event()
        self.gpu_util = []
        self.gpu_mem = []
        self.gpu_total = 0
        self.ram_mb = []
        self.cpu_seconds = []

    def run(self):
        while not self.stop_flag.is_set():
            gpu = _gpu_sample()
            if gpu:
                self.gpu_util.append(gpu[0])
                self.gpu_mem.append(gpu[1])
                self.gpu_total = gpu[2]
            proc = _proc_sample(self.pid)
            if proc:
                self.ram_mb.append(proc[0])
                self.cpu_seconds.append(proc[1])
            self.stop_flag.wait(self.interval)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--port", default="8001")
    parser.add_argument("--sample-interval", type=float, default=15.0)
    args = parser.parse_args(argv)

    import cv2
    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    video_seconds = total_frames / src_fps

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(ROOT, "output", "benchmark_%s.log" % stamp)
    report_path = os.path.join(ROOT, "output", "benchmark_%s.md" % stamp)
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)

    env = os.environ.copy()
    env["VIDEO_SOURCE"] = args.video
    env["DASHBOARD_PORT"] = args.port
    env["OPEN_BROWSER"] = "0"

    print("=" * 66)
    print("BENCHMARK")
    print("  video   : %s" % os.path.basename(args.video))
    print("  length  : %d frames, %.1f min at %.0f fps" %
          (total_frames, video_seconds / 60, src_fps))
    print("  log     : %s" % os.path.relpath(log_path, ROOT))
    print("  report  : %s" % os.path.relpath(report_path, ROOT))
    print("=" * 66, flush=True)

    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-u", "main.py"],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        sampler = Sampler(proc.pid, args.sample_interval)
        sampler.start()

        last_report = 0.0
        while proc.poll() is None:
            time.sleep(5)
            now = time.time()
            if now - last_report >= 60:
                last_report = now
                frame = _last_frame(log_path)
                elapsed = now - started
                rate = frame / elapsed if elapsed > 0 else 0
                remaining = (total_frames - frame) / rate if rate > 0 else 0
                print("  [%s] frame %d/%d (%.0f%%)  %.1f fps  ETA %s" % (
                    str(timedelta(seconds=int(elapsed))), frame, total_frames,
                    100.0 * frame / max(1, total_frames), rate,
                    str(timedelta(seconds=int(remaining)))), flush=True)

        sampler.stop_flag.set()
        sampler.join(timeout=5)

    wall = time.time() - started
    _write_report(report_path, args, total_frames, src_fps, video_seconds,
                  wall, log_path, sampler, proc.returncode)
    print("\nreport written: %s" % os.path.relpath(report_path, ROOT))
    print(_summary_line(total_frames, wall, video_seconds))
    return 0


def _last_frame(log_path):
    frame = 0
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                m = METRIC_RE.search(line)
                if m:
                    try:
                        frame = json.loads(m.group(0)).get("frame_count", frame)
                    except ValueError:
                        pass
    except OSError:
        pass
    return frame


def _read_metrics(log_path):
    rows = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                m = METRIC_RE.search(line)
                if m:
                    try:
                        rows.append(json.loads(m.group(0)))
                    except ValueError:
                        pass
    except OSError:
        pass
    return rows


def _summary_line(frames, wall, video_seconds):
    rate = frames / wall if wall else 0
    return ("processed %d frames in %s -> %.2f fps, %.2fx real time"
            % (frames, str(timedelta(seconds=int(wall))), rate,
               video_seconds / wall if wall else 0))


def _write_report(path, args, total_frames, src_fps, video_seconds,
                  wall, log_path, sampler, exit_code):
    rows = _read_metrics(log_path)
    frames_done = rows[-1].get("frame_count", 0) if rows else 0
    completed = frames_done >= total_frames * 0.99

    loop = [r["loop_ms"] for r in rows if "loop_ms" in r]
    detect = [r["detect_ms"] for r in rows if "detect_ms" in r]
    flow = [r["flow_ms"] for r in rows if "flow_ms" in r]

    # Instantaneous throughput between consecutive metric samples, which shows
    # whether the rate held steady or degraded over the hour.
    inst = []
    for a, b in zip(rows, rows[1:]):
        df = b.get("frame_count", 0) - a.get("frame_count", 0)
        dt = b.get("elapsed_s", 0) - a.get("elapsed_s", 0)
        if df > 0 and dt > 0:
            inst.append(df / dt)
    first_third = inst[: max(1, len(inst) // 3)]
    last_third = inst[-max(1, len(inst) // 3):]

    rate = frames_done / wall if wall else 0
    lines = []
    add = lines.append
    add("# Pipeline benchmark")
    add("")
    add("Generated %s" % datetime.now().strftime("%Y-%m-%d %H:%M"))
    add("")
    add("## Result")
    add("")
    add("| | |")
    add("|---|---|")
    add("| Completed | %s |" % ("yes" if completed else
                                "NO - stopped at frame %d of %d" % (frames_done, total_frames)))
    add("| Exit code | %d |" % exit_code)
    add("| Video length | %.1f min (%d frames @ %.0f fps) |" %
        (video_seconds / 60, total_frames, src_fps))
    add("| Wall-clock time | %s |" % str(timedelta(seconds=int(wall))))
    add("| Frames processed | %d |" % frames_done)
    add("| Sustained throughput | **%.2f fps** |" % rate)
    add("| Speed vs real time | **%.2fx** |" % (video_seconds / wall if wall else 0))
    add("| Time to process 1 h of footage | %s |" %
        (str(timedelta(seconds=int(3600 * src_fps / rate))) if rate else "n/a"))
    add("")

    add("## Throughput stability")
    add("")
    if inst:
        add("| | fps |")
        add("|---|---|")
        add("| Minimum | %.2f |" % min(inst))
        add("| p25 | %.2f |" % _percentile(inst, 25))
        add("| Median | %.2f |" % _percentile(inst, 50))
        add("| p75 | %.2f |" % _percentile(inst, 75))
        add("| Maximum | %.2f |" % max(inst))
        add("")
        avg_first = sum(first_third) / len(first_third)
        avg_last = sum(last_third) / len(last_third)
        drift = (avg_last - avg_first) / avg_first * 100 if avg_first else 0
        add("First third averaged %.2f fps, last third %.2f fps (%+.1f%%)."
            % (avg_first, avg_last, drift))
        add("")
        if abs(drift) < 10:
            add("Throughput held steady across the run: no leak or slow degradation.")
        elif drift < 0:
            add("**Throughput degraded over the run** - worth investigating before "
                "committing to unattended operation.")
        else:
            add("Throughput improved over the run, most likely a lighter scene later on.")
    else:
        add("No metric samples captured.")
    add("")

    add("## Per-stage latency (ms)")
    add("")
    add("| Stage | median | p95 | max |")
    add("|---|---|---|---|")
    for name, series in (("Full loop", loop), ("Detection (YOLO)", detect), ("Optical flow", flow)):
        if series:
            add("| %s | %.1f | %.1f | %.1f |" %
                (name, _percentile(series, 50), _percentile(series, 95), max(series)))
    add("")
    if detect and loop:
        share = 100.0 * _percentile(detect, 50) / _percentile(loop, 50)
        add("Detection is %.0f%% of loop time; the remainder is tracking, "
            "attribution and dashboard encoding." % share)
    add("")

    add("## Resources")
    add("")
    if sampler.gpu_util:
        add("| | mean | peak |")
        add("|---|---|---|")
        add("| GPU utilisation | %.0f%% | %d%% |" %
            (sum(sampler.gpu_util) / len(sampler.gpu_util), max(sampler.gpu_util)))
        add("| GPU memory | %d MiB | %d MiB (of %d MiB) |" %
            (sum(sampler.gpu_mem) / len(sampler.gpu_mem), max(sampler.gpu_mem), sampler.gpu_total))
    else:
        add("GPU sampling unavailable.")
    add("")
    if sampler.ram_mb:
        add("| | mean | peak |")
        add("|---|---|---|")
        add("| Process RAM | %d MB | %d MB |" %
            (sum(sampler.ram_mb) / len(sampler.ram_mb), max(sampler.ram_mb)))
        growth = sampler.ram_mb[-1] - sampler.ram_mb[0]
        add("")
        add("RAM moved %+d MB from first sample to last%s." %
            (growth, " - no leak over the run" if abs(growth) < 300
             else " - **worth checking for a leak**"))
    if sampler.cpu_seconds and wall:
        cpu_pct = 100.0 * (sampler.cpu_seconds[-1] - sampler.cpu_seconds[0]) / wall
        add("")
        add("Average CPU load across the run: %.0f%% of one core-equivalent "
            "(sum over all threads)." % cpu_pct)
    add("")

    add("## Reading this for Jetson")
    add("")
    if rate:
        add("- This machine sustains **%.2f fps**, i.e. **%.2fx real time**." %
            (rate, video_seconds / wall))
        if video_seconds / wall >= 1.0:
            add("- It keeps up with a live 30 fps camera with headroom.")
        else:
            add("- It does **not** keep up with a live 30 fps camera here: an hour "
                "of footage takes %s to process." %
                str(timedelta(seconds=int(3600 * src_fps / rate))))
        add("- Detection dominates the loop, so Jetson throughput will track its "
            "GPU inference speed for this model more than anything else.")
        add("- GPU memory peaked at %d MiB, which is the figure to compare against "
            "the Jetson module's shared memory budget." %
            (max(sampler.gpu_mem) if sampler.gpu_mem else 0))
        add("- TensorRT export is the obvious next lever, and is normally worth "
            "2-4x over PyTorch inference on Jetson.")
    add("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
