"""Line a kds-ocr emission up with the production video.

A LIVE run needs none of this: both feeds are the present moment.

A RECORDED run does.  kds-ocr replays its hour at 1x wall-clock, while our
production loop runs at whatever rate the detector manages -- so without
pacing, an hour of tickets can arrive long before the production video reaches
the food those tickets describe, and every order would be judged against an
empty kitchen.

kds-ocr stamps each emission with ``screen_at``: the moment the ticket was on
screen, in the RECORDING's own clock (its filename start plus the position in
the file).  Converting that to "seconds into the recording" gives a number
directly comparable to the production loop's media time, which is what
`KdsOcrClient.set_master_time` is fed.

The filename convention is kds-ocr's own (``..._12_to_13_p0007_PDT.mkv`` =
12:00:07 PDT); it is re-implemented here rather than imported because that
project is a separate process, not a dependency on our import path.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_FNAME_RE = re.compile(
    r"(?P<y>\d{4})_(?P<mo>\d{2})_(?P<d>\d{2})_(?P<h>\d{2})_to_\d{2}"
    r"(?:_p(?P<off>\d{4}))?_(?P<tz>[A-Z]{2,4})"
)

_TZ = {"PDT": "America/Los_Angeles", "PST": "America/Los_Angeles"}


def video_start_from_filename(name: str) -> Optional[datetime]:
    """``..._2026_09_13_12_to_13_p0007_PDT.mkv`` -> 2026-09-13 12:00:07 PDT."""
    m = _FNAME_RE.search(str(name))
    if not m:
        return None
    tz = ZoneInfo(_TZ.get(m.group("tz"), "America/Los_Angeles"))
    base = datetime(int(m.group("y")), int(m.group("mo")), int(m.group("d")),
                    int(m.group("h")), tzinfo=tz)
    off = m.group("off")
    if off:
        base += timedelta(minutes=int(off[:2]), seconds=int(off[2:]))
    return base


def parse_screen_at(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def offset_seconds(screen_at, video_start: Optional[datetime]) -> Optional[float]:
    """How far into the recording this emission belongs, or None if unknowable."""
    ts = parse_screen_at(screen_at)
    if ts is None or video_start is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=video_start.tzinfo)
    return (ts - video_start).total_seconds()
