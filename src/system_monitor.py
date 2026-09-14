"""Background system stats for the dashboard's Analysis window.

Reads `tegrastats` (ships with JetPack, needs no sudo) once a second: RAM, CPU
load per core, GPU load, temperatures and power rails.  On a machine without
tegrastats the snapshot is simply empty.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import threading
from typing import Optional

_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
_SWAP = re.compile(r"SWAP (\d+)/(\d+)MB")
_CPU = re.compile(r"CPU \[([^\]]+)\]")
_CORE = re.compile(r"(\d+)%@")
_GPU = re.compile(r"GR3D_FREQ (\d+)%")
_TEMP = re.compile(r"\b(cpu|gpu|tj|soc\d?)@(-?[\d.]+)C")
_RAIL = re.compile(r"\b(VDD_[A-Z0-9_]+) (\d+)mW")


def parse_tegrastats(line: str) -> Optional[dict]:
    """One tegrastats line -> dict, or None if it is not a stats line."""
    ram = _RAM.search(line)
    if not ram:
        return None
    out: dict = {"ram_used_mb": int(ram.group(1)), "ram_total_mb": int(ram.group(2))}
    swap = _SWAP.search(line)
    if swap:
        out["swap_used_mb"] = int(swap.group(1))
    cpu = _CPU.search(line)
    if cpu:
        cores = []
        for part in cpu.group(1).split(","):
            m = _CORE.search(part)
            cores.append(int(m.group(1)) if m else 0)  # "off" cores count as idle
        if cores:
            out["cpu_cores"] = cores
            out["cpu_avg_pct"] = round(sum(cores) / len(cores), 1)
            out["cpu_max_pct"] = max(cores)
    gpu = _GPU.search(line)
    if gpu:
        out["gpu_pct"] = int(gpu.group(1))
    temps = {name: float(value) for name, value in _TEMP.findall(line)}
    if temps:
        out["temps_c"] = temps
        out["temp_c"] = temps.get("tj", max(temps.values()))
    rails = {name: int(mw) for name, mw in _RAIL.findall(line)}
    if rails:
        out["rails_mw"] = rails
        out["power_w"] = round(rails.get("VDD_IN", sum(rails.values())) / 1000.0, 2)
    return out


class SystemMonitor:
    def __init__(self, interval_ms: int = 1000):
        self.interval_ms = interval_ms
        self.available = shutil.which("tegrastats") is not None
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: dict = {}

    def start(self) -> "SystemMonitor":
        if not self.available or self._proc is not None:
            return self
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._run, name="system-monitor", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            snap = parse_tegrastats(line)
            if snap:
                with self._lock:
                    self._latest = snap

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._latest)

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None
