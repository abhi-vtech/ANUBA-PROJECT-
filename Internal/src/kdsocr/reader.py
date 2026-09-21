"""Run kds-ocr as a child process.

kds-ocr is a separate project (``kds-ocr/`` inside this one, git remote
``anuba-technologies/kds-ocr``) with its own reference data and its own
``reference/`` lookups resolved relative to its repo root -- so it is invoked
as a process from that directory rather than imported.  That also keeps its
EasyOCR/torch stack off our import path and means a crash in the reader cannot
take the detection loop down with it.

It writes recipes to a JSONL file as they happen; `EmissionTailer` reads that
file.  The two processes share nothing else.
"""
from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

#: The kds-ocr checkout. It lives inside the project (moved there
#: 2026-09-15), so a relative path resolves from the pipeline's
#: working directory; override with KDSOCR_REPO.
DEFAULT_REPO = "kds-ocr"


@dataclass
class ReaderConfig:
    """How to launch kds-ocr. Every field has a working default for this box."""

    #: Checkout of anuba-technologies/kds-ocr.
    repo: str = DEFAULT_REPO
    #: Recording(s) to read, in chronological order. Mutually exclusive with rtsp.
    videos: List[str] = field(default_factory=list)
    #: Live KDS feed. Never logged -- it carries credentials.
    rtsp: str = ""
    #: Bound a live run so it finalizes. 0 = unbounded.
    live_seconds: float = 0.0
    #: Where kds-ocr writes its run files.
    out_dir: str = "output/kdsocr"
    #: The stream we actually consume.
    recipes_path: str = "output/kdsocr/recipes.jsonl"
    #: EasyOCR on the GPU. Effectively mandatory: torch on this Orin is
    #: ~40x slower on the CPU for this workload.
    gpu: bool = True
    #: Replay a recording at 1x so emissions arrive at the pace the kitchen
    #: worked. Without it a one-hour video finishes in minutes and every
    #: ticket lands before the production video has caught up.
    realtime: bool = True
    #: Skip into the recording (seconds) -- both feeds must use the same offset.
    start_at: float = 0.0
    sample_interval: Optional[float] = None
    store_id: str = "6258"
    #: REPLAY: consume an existing kds-ocr recipe stream instead of producing a
    #: new one.  The KDS side is then byte-for-byte the same on every run, which
    #: is what makes two vision runs comparable -- re-reading the screen each
    #: time re-derives the tickets and moves the baseline underneath whatever
    #: was being measured.  It also skips ~18 minutes of EasyOCR per run.
    #:
    #: The file must be a kds-ocr `recipes.jsonl`.  The goldset is NOT a
    #: substitute: it stores raw ticket records (order_ref, items, item_count)
    #: with no ingredient expansion, no `total_dogs` and no emission timing, and
    #: it covers different business dates than any one production video -- so it
    #: cannot pace against the kitchen footage at all.  It exists to SCORE the
    #: OCR offline (kds-ocr/scripts/evaluate_gold.py), not to feed the pipeline.
    replay: bool = False
    #: Interpreter for the child. Ours by default: it has easyocr and a working
    #: CUDA torch, so no separate environment is needed.
    python: str = ""
    #: Child stdout/stderr goes here; it is verbose and would drown our log.
    log_path: str = "output/kdsocr/reader.log"

    def resolved_repo(self) -> Path:
        return Path(os.path.expanduser(self.repo)).resolve()

    def command(self) -> List[str]:
        if not self.videos and not self.rtsp:
            raise ValueError("kds-ocr needs either videos= or rtsp=")
        if self.videos and self.rtsp:
            raise ValueError("kds-ocr takes videos= or rtsp=, not both")
        py = self.python or sys.executable
        cmd = [py, "-m", "kds.cli", "--no-db",
               "--out", str(Path(self.out_dir).resolve()),
               "--recipes-out", str(Path(self.recipes_path).resolve()),
               "--store-id", str(self.store_id)]
        if self.rtsp:
            cmd += ["--rtsp", self.rtsp]
            if self.live_seconds > 0:
                cmd += ["--live-seconds", str(self.live_seconds)]
        else:
            cmd += ["--videos"] + [str(Path(v).resolve()) for v in self.videos]
        if self.gpu:
            cmd.append("--gpu")
        if self.realtime:
            cmd.append("--realtime")
        if self.start_at:
            cmd += ["--start-at", str(self.start_at)]
        if self.sample_interval is not None:
            cmd += ["--sample-interval", str(self.sample_interval)]
        return cmd

    def describe(self) -> str:
        """The command with the RTSP credentials masked."""
        parts = []
        mask_next = False
        for part in self.command():
            if mask_next:
                parts.append("<rtsp-url-hidden>")
                mask_next = False
                continue
            mask_next = part == "--rtsp"
            parts.append(shlex.quote(part))
        return " ".join(parts)


class KdsOcrReader:
    """Owns the kds-ocr child process."""

    def __init__(self, config: ReaderConfig):
        self.config = config
        self.proc: Optional[subprocess.Popen] = None
        self._log_fh = None
        self._started_at = 0.0

    def start(self) -> None:
        if self.config.replay:
            # Nothing to launch: the recipe stream already exists and the
            # tailer reads it from the start. Crucially do NOT rotate it --
            # the rotation below exists to stop a stale file replaying, which
            # is precisely what is wanted here.
            recipes = Path(self.config.recipes_path)
            if not recipes.is_file():
                raise SystemExit(
                    "KDS replay asked for but %s does not exist. Point "
                    "KDS_RECIPES at a kds-ocr recipes.jsonl." % recipes
                )
            n = sum(1 for _ in recipes.open())
            logger.info("kds-ocr REPLAY: reading %d emission(s) from %s; "
                        "no OCR process started", n, recipes)
            return

        repo = self.config.resolved_repo()
        if not (repo / "kds" / "cli.py").is_file():
            raise SystemExit(
                "kds-ocr not found at %s (no kds/cli.py). Clone "
                "https://github.com/anuba-technologies/kds-ocr.git there, or set "
                "KDSOCR_REPO." % repo
            )
        for path in (self.config.out_dir, os.path.dirname(self.config.recipes_path),
                     os.path.dirname(self.config.log_path)):
            if path:
                os.makedirs(path, exist_ok=True)
        # A stale file from a previous run would replay as this run's tickets.
        recipes = Path(self.config.recipes_path)
        if recipes.exists():
            stamp = time.strftime("%Y%m%d_%H%M%S")
            rotated = recipes.with_suffix(".jsonl.%s" % stamp)
            recipes.rename(rotated)
            logger.info("rotated previous recipe stream to %s", rotated.name)

        self._log_fh = open(self.config.log_path, "a")
        cmd = self.config.command()
        logger.info("starting kds-ocr: %s", self.config.describe())
        # Its own process group, so stopping it cannot signal our loop.
        self.proc = subprocess.Popen(
            cmd, cwd=str(repo), stdout=self._log_fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._started_at = time.time()
        logger.info("kds-ocr running as pid %d, log at %s",
                    self.proc.pid, self.config.log_path)

    @property
    def started(self) -> bool:
        return self.proc is not None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def exit_code(self) -> Optional[int]:
        return None if self.proc is None else self.proc.poll()

    def stop(self, timeout: float = 10.0) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            logger.info("stopping kds-ocr (pid %d)", self.proc.pid)
            try:
                self.proc.send_signal(signal.SIGINT)   # let it finalize
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("kds-ocr did not stop on SIGINT; killing")
                self.proc.kill()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            except (OSError, ValueError):
                pass
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            finally:
                self._log_fh = None
