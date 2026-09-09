"""Event timeline for debugging (section 20).

Keeps a bounded in-memory ring for the dashboard and, optionally, appends every
event as one JSON line to disk so a run can be reconstructed afterwards.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


def format_clock(timestamp: Optional[float] = None) -> str:
    return time.strftime("%H:%M:%S", time.localtime(timestamp or time.time()))


class EventTimeline:
    def __init__(self, path: Optional[str] = None, maxlen: int = 400):
        self.path = path
        self._events: Deque[Dict] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        if path:
            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                logger.warning("Could not create timeline directory for %s", path)

    def add(
        self,
        kind: str,
        message: str,
        ticket_id: str = "",
        timestamp: Optional[float] = None,
        **extra,
    ) -> Dict:
        record = {
            "time": format_clock(),
            "monotonic": timestamp if timestamp is not None else time.monotonic(),
            "wall": time.time(),
            "kind": kind,
            "ticket_id": ticket_id,
            "message": message,
        }
        if extra:
            record.update(extra)
        with self._lock:
            self._events.appendleft(record)
            if self.path:
                try:
                    with open(self.path, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                except OSError:
                    logger.warning("Could not append to timeline %s", self.path)
        logger.info("[%s] %s %s", record["time"], kind, message)
        return record

    def recent(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            return list(self._events)[:limit]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
