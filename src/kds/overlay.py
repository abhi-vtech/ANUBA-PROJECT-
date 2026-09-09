"""Annotate the KDS screen frame for the dashboard feed.

Draws what the reader currently believes about each ticket card directly on the
KDS video, so the operator can see the system's interpretation next to the real
screen: which card is which ticket, whether it is paid, and how many of its
hotdogs have been made.

It also carries the verdict.  An order is judged when its ticket DISAPPEARS
from the KDS, at which point there is no longer a card to draw on -- so the
CORRECT / WRONG result is shown as a banner over the feed for a few seconds
after it is decided.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from src.kds.schemas import LifecycleState

# BGR
_GREEN = (94, 197, 34)
_AMBER = (0, 176, 240)
_GREY = (150, 150, 150)
_RED = (68, 68, 239)
_BLUE = (246, 130, 59)
_WHITE = (245, 245, 245)
_DARK = (26, 24, 22)

_STATE_COLOR: Dict[str, tuple] = {
    LifecycleState.ACTIVE.value: _GREEN,
    LifecycleState.IN_PROGRESS.value: _GREEN,
    LifecycleState.WAITING_FOR_COMPLETION.value: _GREEN,
    LifecycleState.QUEUED.value: _BLUE,
    LifecycleState.PAID.value: _BLUE,
    LifecycleState.COMPLETED.value: _GREEN,
    LifecycleState.WRONG.value: _RED,
}


#: How long a verdict banner stays on the feed after the ticket disappears.
VERDICT_HOLD_S = 8.0


def _label(image: np.ndarray, text: str, x: int, y: int, color, scale=0.45) -> None:
    """Draw text on a filled chip so it stays readable over any card colour."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    cv2.rectangle(image, (x, y - th - 6), (x + tw + 10, y + 4), _DARK, -1)
    cv2.rectangle(image, (x, y - th - 6), (x + tw + 10, y + 4), color, 1)
    cv2.putText(image, text, (x + 5, y - 2), font, scale, color, 1, cv2.LINE_AA)


def annotate_kds_frame(
    frame: np.ndarray,
    cards: Sequence,
    manager,
    stability,
    verdict: Optional[dict] = None,
) -> np.ndarray:
    """Return a copy of ``frame`` with per-card ticket state drawn on it.

    ``cards`` are the :class:`~src.kds.ticket_detector.DetectedCard` objects for
    this frame.  Cards with no confirmed ticket yet are outlined in grey, so an
    unrecognised card is visibly different from an ignored unpaid one.

    ``verdict`` is the most recent finalised order, as produced by
    :meth:`KdsMonitor.last_verdict`; it is drawn as a banner while fresh.
    """
    if frame is None or frame.size == 0:
        return frame
    canvas = frame.copy()

    for card in cards:
        x1, y1, x2, y2 = card.bbox
        ticket_id = stability.ticket_for_slot(card.slot_id)
        group = manager.get(ticket_id) if ticket_id else None

        if group is not None:
            color = _STATE_COLOR.get(group.state.value, _BLUE)
            state = group.state.value
            position = manager.position_of(ticket_id)
            if position == 0:
                state = "ACTIVE - " + state
            elif position > 0:
                state = "#%d in queue" % (position + 1)
            caption = "%d/%d hotdogs" % (group.detected_total, group.expected_total)
        elif ticket_id:
            # Seen and identified, but not paid -> deliberately not tracked.
            color = _AMBER
            state = "NOT PAID - ignored"
            caption = ""
        else:
            color = _GREY
            state = "reading..."
            caption = ""

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        _label(canvas, ticket_id or "card %d" % card.slot_id, x1 + 3, y1 + 20, color, 0.5)
        _label(canvas, state, x1 + 3, y1 + 42, color)
        if caption:
            _label(canvas, caption, x1 + 3, y1 + 62, color)

    _draw_header(canvas, manager)
    _draw_verdict(canvas, verdict)
    return canvas


def _draw_verdict(canvas: np.ndarray, verdict: Optional[dict]) -> None:
    """Banner announcing the result of the ticket that just disappeared."""
    if not verdict:
        return
    age = time.monotonic() - verdict.get("shown_at", 0.0)
    if age > VERDICT_HOLD_S:
        return

    correct = bool(verdict.get("correct"))
    color = _GREEN if correct else _RED
    headline = "%s  ORDER %s" % (
        "OK" if correct else "!",
        "CORRECT" if correct else "WRONG",
    )
    lines = [
        "%s   TICKET %s" % (headline, verdict.get("ticket_id", "?")),
        verdict.get("detail", ""),
    ]
    lines = [line for line in lines if line]

    height, width = canvas.shape[:2]
    box_h = 26 * len(lines) + 16
    top = height - box_h - 10
    overlay = canvas.copy()
    cv2.rectangle(overlay, (10, top), (width - 10, height - 10), _DARK, -1)
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
    cv2.rectangle(canvas, (10, top), (width - 10, height - 10), color, 2)
    for i, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (24, top + 30 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62 if i == 0 else 0.5,
            color if i == 0 else _WHITE,
            2 if i == 0 else 1,
            cv2.LINE_AA,
        )


def _draw_header(canvas: np.ndarray, manager) -> None:
    """A one-line FIFO summary across the top of the KDS feed."""
    height, width = canvas.shape[:2]
    queue = manager.queue
    active = manager.active_group
    text = "KDS FEED   queue:%d  done:%d  wrong:%d" % (
        len(queue),
        len(manager.completed),
        len(manager.warnings),
    )
    if active is not None:
        text += "   ACTIVE %s  %d/%d" % (
            active.ticket_id,
            active.detected_total,
            active.expected_total,
        )
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, 26), _DARK, -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
    cv2.putText(
        canvas, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1, cv2.LINE_AA
    )
