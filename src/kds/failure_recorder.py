"""Record the dashboard view for every ticket, keep it only when the order fails.

A wrong order is worth watching back; a correct one is not.  But whether an
order is wrong is only known when its ticket disappears, long after the
interesting part happened -- so every ticket is recorded from the moment it is
created, and the clip is **deleted** if the order turns out CORRECT.

Recording goes straight to a per-ticket file rather than a RAM buffer.  A
five-minute ticket at 1280x720 would be several GB in memory; on disk it is a
few MB, and the discard path is a file delete.

Output for a failed order::

    output/failures/CHK248_20260907_191533.mp4    the dashboard view
    output/failures/CHK248_20260907_191533.json   expected vs detected, reason
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_HEADER_H = 34
_DARK = (24, 22, 20)
_WHITE = (240, 240, 240)
_GREEN = (94, 197, 34)
_RED = (68, 68, 239)


@dataclass
class _Clip:
    ticket_id: str
    writer: cv2.VideoWriter
    path: Path
    size: Tuple[int, int]
    started_at: float
    frames: int = 0


class FailureRecorder:
    """Records one clip per live ticket; keeps only the failures."""

    def __init__(
        self,
        output_dir: str = "output/failures",
        fps: float = 10.0,
        max_seconds: float = 900.0,
        max_concurrent: int = 6,
        enabled: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.fps = max(1.0, float(fps))
        self.max_seconds = float(max_seconds)
        self.max_concurrent = int(max_concurrent)
        self.enabled = bool(enabled)
        self._clips: Dict[str, _Clip] = {}
        # In-progress clips live INSIDE the output directory, not in the system
        # temp dir: keeping a failure means renaming the file, and a rename
        # cannot cross drives (output/ is often on a different disk from %TEMP%).
        self._tmp_dir = self.output_dir / ".pending"
        if self.enabled:
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self._tmp_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                logger.exception("Could not create recording directories")
                self.enabled = False
        self.kept = 0
        self.discarded = 0

    # ------------------------------------------------------------------ start

    def start(self, ticket_id: str, frame: np.ndarray) -> None:
        """Begin recording a ticket.  Safe to call more than once."""
        if not self.enabled or not ticket_id or ticket_id in self._clips:
            return
        if frame is None or frame.size == 0:
            return
        if len(self._clips) >= self.max_concurrent:
            logger.warning(
                "Not recording %s: already recording %d tickets",
                ticket_id,
                len(self._clips),
            )
            return

        height, width = frame.shape[:2]
        safe = "".join(c for c in ticket_id if c.isalnum() or c in "-_")
        path = self._tmp_dir / ("%s_%d.mp4" % (safe, int(time.time() * 1000)))
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
        )
        if not writer.isOpened():
            logger.error("Could not open a video writer for ticket %s", ticket_id)
            return
        self._clips[ticket_id] = _Clip(
            ticket_id=ticket_id,
            writer=writer,
            path=path,
            size=(width, height),
            started_at=time.monotonic(),
        )
        logger.info("Recording ticket %s -> %s", ticket_id, path.name)

    # ------------------------------------------------------------------ write

    def write(self, frame: np.ndarray) -> None:
        """Append one dashboard frame to every ticket currently recording."""
        if not self.enabled or not self._clips or frame is None or frame.size == 0:
            return
        height, width = frame.shape[:2]
        for ticket_id, clip in list(self._clips.items()):
            if time.monotonic() - clip.started_at > self.max_seconds:
                logger.warning(
                    "Ticket %s exceeded the %.0fs recording cap; stopping its clip",
                    ticket_id,
                    self.max_seconds,
                )
                self.finish(ticket_id, correct=False, detail="recording cap reached")
                continue
            payload = frame
            if (width, height) != clip.size:
                payload = cv2.resize(frame, clip.size)
            clip.writer.write(payload)
            clip.frames += 1

    # ----------------------------------------------------------------- finish

    def finish(
        self,
        ticket_id: str,
        correct: bool,
        detail: str = "",
        result: Optional[dict] = None,
    ) -> Optional[Path]:
        """Close a ticket's clip.  Keeps it only if the order was wrong."""
        clip = self._clips.pop(ticket_id, None)
        if clip is None:
            return None
        try:
            clip.writer.release()
        except Exception:  # pragma: no cover
            logger.debug("writer release failed", exc_info=True)

        if correct:
            # Nothing to review -- throw it away rather than fill the disk.
            self.discarded += 1
            _unlink(clip.path)
            logger.info(
                "Ticket %s CORRECT - recording discarded (%d frames)",
                ticket_id,
                clip.frames,
            )
            return None

        stamp = time.strftime("%Y%m%d_%H%M%S")
        safe = "".join(c for c in ticket_id if c.isalnum() or c in "-_")
        final = self.output_dir / ("%s_%s.mp4" % (safe, stamp))
        try:
            os.replace(str(clip.path), str(final))
        except OSError:
            # Same-filesystem rename is the normal path; fall back to a copy so
            # an unusual layout still preserves the evidence.
            try:
                shutil.move(str(clip.path), str(final))
            except (OSError, shutil.Error):
                logger.exception("Could not keep the failure clip for %s", ticket_id)
                _unlink(clip.path)
                return None

        sidecar = final.with_suffix(".json")
        try:
            with open(sidecar, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "ticket_id": ticket_id,
                        "recorded_at": stamp,
                        "video": final.name,
                        "frames": clip.frames,
                        "duration_s": round(clip.frames / self.fps, 2),
                        "detail": detail,
                        "result": result or {},
                    },
                    handle,
                    indent=2,
                )
        except OSError:
            logger.warning("Could not write the sidecar for %s", ticket_id)

        self.kept += 1
        logger.warning(
            "WRONG ORDER %s - recording kept: %s (%d frames)",
            ticket_id,
            final,
            clip.frames,
        )
        return final

    def close(self) -> None:
        """Keep every still-open clip: an unfinished order is unverified."""
        for ticket_id in list(self._clips):
            self.finish(ticket_id, correct=False, detail="pipeline stopped")

    @property
    def active(self) -> int:
        return len(self._clips)

    def is_recording(self, ticket_id: str) -> bool:
        return ticket_id in self._clips


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        logger.debug("could not delete %s", path)


# ---------------------------------------------------------------------------
# Dashboard composition
# ---------------------------------------------------------------------------

def compose_dashboard_frame(
    production: Optional[np.ndarray],
    kds: Optional[np.ndarray],
    status: str = "",
    correct: Optional[bool] = None,
    target_height: int = 720,
) -> Optional[np.ndarray]:
    """Build the frame that gets recorded: production feed beside the KDS feed.

    This is the pair a reviewer actually needs -- what the kitchen did, next to
    the ticket that asked for it -- with a status strip naming the ticket and
    its progress.
    """
    panels = []
    for image in (production, kds):
        if image is None or image.size == 0:
            continue
        scale = target_height / float(image.shape[0])
        panels.append(
            cv2.resize(image, (max(1, int(image.shape[1] * scale)), target_height))
        )
    if not panels:
        return None

    body = panels[0] if len(panels) == 1 else np.hstack(panels)
    canvas = np.zeros((target_height + _HEADER_H, body.shape[1], 3), np.uint8)
    canvas[:] = _DARK
    canvas[_HEADER_H:, :] = body

    color = _WHITE if correct is None else (_GREEN if correct else _RED)
    cv2.putText(
        canvas,
        status[:200],
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )
    return canvas
