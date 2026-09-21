"""Replay frames from a JSONL file. No GPU, no model, no video decode.

This is the source that makes the analysis testable.  Any run of any other
source can be recorded with `--record`, then replayed here to check that a
change to the rules produces the intended difference in events -- at thousands
of frames per second, deterministically, in CI.

One JSON object per line:

    {"index": 0, "t": 0.0, "width": 1280, "height": 720,
     "objects": [{"track_id": 4, "label": "hand", "bbox": [10,20,60,80],
                  "confidence": 0.91}]}
"""
from __future__ import annotations

import json
from typing import Iterator, Optional

from src.core.contract import Frame, TrackedObject


class ReplaySource:
    def __init__(self, path: str, limit: Optional[int] = None):
        self.path = path
        self.limit = limit
        self._fh = None

    def __iter__(self) -> Iterator[Frame]:
        self._fh = open(self.path)
        try:
            for n, line in enumerate(self._fh):
                if self.limit is not None and n >= self.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield Frame(
                    index=int(row.get("index", n)),
                    t=float(row.get("t", 0.0)),
                    width=int(row.get("width", 1280)),
                    height=int(row.get("height", 720)),
                    objects=[
                        TrackedObject(
                            track_id=int(o.get("track_id", -1)),
                            label=str(o["label"]),
                            bbox=tuple(float(v) for v in o["bbox"]),
                            confidence=float(o.get("confidence", 1.0)),
                            polygon=o.get("polygon"),
                        )
                        for o in row.get("objects", [])
                    ],
                )
        finally:
            self.close()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
