"""One frame showing both feeds and the ticket state -- what gets recorded.

The dashboard is a web page, so it cannot be written to a video file directly.
This composes the same three things the page shows into a single frame:

    [ annotated kitchen camera ]  [ KDS screen ]
    [ ticket status bar                        ]

That frame is what `RECORD_DASHBOARD` writes for the whole run, so the
recording is reviewable on its own later -- the verdict, the food, and the
ticket that asked for it, in one picture, without needing the dashboard
running.

Kept deliberately cheap: a resize, a copy into a canvas and some text. It runs
on the pinned main loop, and `draw_annotations` already costs that loop more
than it should (see the GPU-offload notes).
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BANNER_H = 64
PAD = 8
BG = (18, 24, 33)          # matches the dashboard's dark ground
FG = (229, 231, 235)
MUTED = (156, 163, 175)
ACCENT = (249, 115, 22)    # the dashboard's ticket orange
OK = (129, 199, 132)
BAD = (113, 113, 239)      # BGR: red

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img, text, x, y, scale=0.45, color=FG, thick=1):
    cv2.putText(img, text, (x, y), _FONT, scale, color, thick, cv2.LINE_AA)


def _placeholder(w: int, h: int, text: str) -> np.ndarray:
    panel = np.full((h, w, 3), BG, np.uint8)
    _label(panel, text, max(8, w // 2 - 110), h // 2, 0.5, MUTED)
    return panel


def _fit(frame: Optional[np.ndarray], h: int, fallback_w: int,
         missing: str) -> np.ndarray:
    """Scale a frame to height `h`, keeping its aspect ratio."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return _placeholder(fallback_w, h, missing)
    fh, fw = frame.shape[:2]
    if fh == h:
        return frame
    w = max(1, int(round(fw * (h / float(fh)))))
    interp = cv2.INTER_AREA if h < fh else cv2.INTER_LINEAR
    return cv2.resize(frame, (w, h), interpolation=interp)


def compose_dashboard_frame(
    production: Optional[np.ndarray],
    kds: Optional[np.ndarray] = None,
    status: str = "",
    lines: Optional[Sequence[str]] = None,
    verdict: Optional[bool] = None,
) -> Optional[np.ndarray]:
    """Kitchen camera + KDS screen + a status bar, as one frame.

    Returns None only when there is nothing at all to draw, so the caller can
    skip the write rather than record a blank.
    """
    if production is None and kds is None:
        return None

    # The kitchen camera sets the height; the KDS screen (1280x1024) is
    # letterboxed to match, so neither is cropped.
    left = _fit(production, 720 if production is None else production.shape[0],
                1280, "no kitchen frame")
    h = left.shape[0]
    right = _fit(kds, h, 900, "no KDS frame (KDS_PREVIEW=0)")

    width = left.shape[1] + right.shape[1] + PAD
    canvas = np.full((h + BANNER_H, width, 3), BG, np.uint8)
    canvas[0:h, 0:left.shape[1]] = left
    canvas[0:h, left.shape[1] + PAD:width] = right

    # Which half is which -- a reviewer opening the file months later should
    # not have to guess.
    _label(canvas, "KITCHEN", 12, 22, 0.5, MUTED)
    _label(canvas, "KDS SCREEN", left.shape[1] + PAD + 12, 22, 0.5, MUTED)

    bar_y = h
    cv2.rectangle(canvas, (0, bar_y), (width, h + BANNER_H), BG, -1)
    cv2.line(canvas, (0, bar_y), (width, bar_y), (55, 65, 81), 1)

    colour = ACCENT if verdict is None else (OK if verdict else BAD)
    _label(canvas, status or "no active ticket", 12, bar_y + 26, 0.62, colour, 2)
    if lines:
        _label(canvas, "   |   ".join(lines)[:220], 12, bar_y + 50, 0.46, MUTED)
    return canvas


def status_for(order, ticket_id: Optional[str] = None) -> tuple:
    """`(headline, detail_lines)` describing the order being built.

    Reads only what `src.domain.schemas.Order` guarantees, so this module does
    not drag the state machine in.
    """
    if order is None or not getattr(order, "ticket_id", ""):
        return ("no active ticket", [])
    required = dict(getattr(order, "required_counts", {}) or {})
    picked = dict(getattr(order, "picked_counts", {}) or {})
    dogs_need = int(required.get("hot-dog", getattr(order, "hotdog_count", 0)) or 0)
    dogs_got = int(picked.get("hot-dog", 0) or 0)

    head = "TICKET %s   %d/%d hotdogs" % (
        ticket_id or order.ticket_id, dogs_got, dogs_need)

    # Toppings only: the hotdog count is already in the headline.
    detail: List[str] = []
    for item in sorted(required):
        if item == "hot-dog":
            continue
        detail.append("%s %d/%d" % (item, picked.get(item, 0), required[item]))
    return (head, detail)
