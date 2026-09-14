"""The runtime-neutral contract every pipeline speaks.

This module is the seam between *how frames are produced* and *what we conclude
from them*.  Everything above it (Ultralytics on .pt/.onnx/.engine, DeepStream
nvinfer, a recorded JSONL replay) is a source; everything below it (zones,
lifecycle, order validation) is analysis.  Analysis imports only this module,
so it cannot accidentally acquire a dependency on torch, cv2 or pyservicemaker.

The rule that keeps it honest: nothing here may import a runtime.  If a new
field cannot be filled in by *every* source, it does not belong in the
contract -- put it in `Frame.extra` instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple

BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

#: Track id used when the runtime produced a detection but no tracking identity.
UNTRACKED: int = -1


@dataclass(frozen=True)
class TrackedObject:
    """One detected object in one frame, as every runtime can describe it.

    `label` is the canonical class *name*, never an index.  Sources are
    responsible for mapping their own class ids through their label file, so
    analysis code never has to know that "hot-dog" is index 6 in one runtime
    and index 4 in another -- the class-index coupling that makes
    `[class-attrs-6]` in config_infer.txt so fragile stops at this boundary.
    """

    track_id: int
    label: str
    bbox: BBox
    confidence: float = 1.0
    polygon: Optional[List[Tuple[float, float]]] = None

    @property
    def is_tracked(self) -> bool:
        return self.track_id != UNTRACKED

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True)
class Frame:
    """One frame's worth of objects, plus the clock the analysis should use.

    `t` is *media* time in seconds, not wall time: PTS for a file or a
    DeepStream buffer, frame_index/fps for a source that has no timestamps.
    Every dwell and debounce threshold in the analysis is expressed against
    this clock, which is why a 3.7x faster runtime does not change any
    lifecycle outcome -- the same video always yields the same t.
    """

    index: int
    t: float
    width: int
    height: int
    objects: Sequence[TrackedObject] = ()
    extra: Dict[str, Any] = field(default_factory=dict)

    def by_label(self, *labels: str) -> List[TrackedObject]:
        wanted = set(labels)
        return [o for o in self.objects if o.label in wanted]


class FrameSource(Protocol):
    """Anything that can produce `Frame`s.

    Sources that own their event loop (DeepStream) implement this by pushing
    into a queue that `__iter__` drains; sources that are pulled (Ultralytics,
    replay) yield directly.  `close()` must be idempotent.
    """

    def __iter__(self) -> Iterator[Frame]: ...

    def close(self) -> None: ...


@dataclass
class Event:
    """Something the analysis concluded. The only output type."""

    t: float
    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"t": round(self.t, 3), "event": self.kind, **self.payload}
