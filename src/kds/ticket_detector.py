"""Locate ticket cards on the KDS screen.

The KDS renders each order as a pale card on a dark desktop, so cards are found
by masking the card-body colour band, closing the text rows into a solid blob,
and keeping contours whose size and aspect look like a ticket.

Detected boxes are then matched to persistent *slots*.  A slot is a stable
handle on "the card in this position", which is what lets the rest of the
pipeline keep tracking a ticket through a frame where OCR failed entirely --
the card is still visibly there, so the ticket has not disappeared.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.kds.colors import ColorBands

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class CardSlot:
    """A persistent screen position that holds a ticket card."""

    slot_id: int
    bbox: BBox
    first_seen: float
    last_seen: float
    # The ticket ID currently bound to this slot, once OCR confirms one.
    ticket_id: Optional[str] = None
    seen_count: int = 1

    def update(self, bbox: BBox, timestamp: float) -> None:
        # Light smoothing: card boxes jitter a few px as the text repaints.
        x1, y1, x2, y2 = self.bbox
        nx1, ny1, nx2, ny2 = bbox
        alpha = 0.4
        self.bbox = (
            int(x1 + alpha * (nx1 - x1)),
            int(y1 + alpha * (ny1 - y1)),
            int(x2 + alpha * (nx2 - x2)),
            int(y2 + alpha * (ny2 - y2)),
        )
        self.last_seen = timestamp
        self.seen_count += 1


@dataclass
class DetectedCard:
    """One card found in one frame, with its crop."""

    slot_id: int
    bbox: BBox
    image: np.ndarray
    ticket_id: Optional[str] = None


class TicketCardDetector:
    """Finds ticket cards and keeps stable slot identities across frames."""

    def __init__(self, bands: Optional[ColorBands] = None, config: Optional[dict] = None):
        self.bands = bands or ColorBands(config)
        cfg = (self.bands.config.get("card_detection", {}) or {})
        self.min_area_fraction = float(cfg.get("min_area_fraction", 0.008))
        self.max_area_fraction = float(cfg.get("max_area_fraction", 0.60))
        self.min_aspect = float(cfg.get("min_aspect_ratio", 0.55))
        self.max_aspect = float(cfg.get("max_aspect_ratio", 4.0))
        self.close_kernel = int(cfg.get("close_kernel", 15))
        self.min_width_px = int(cfg.get("min_width_px", 90))
        self.min_height_px = int(cfg.get("min_height_px", 110))
        self.slot_iou_threshold = float(cfg.get("slot_iou_threshold", 0.45))
        self.screen_v_max = int(cfg.get("screen_v_max", 110))
        self.min_screen_fraction = float(cfg.get("min_screen_fraction", 0.08))
        self.min_rectangularity = float(cfg.get("min_rectangularity", 0.70))
        self.split_merged_columns = bool(cfg.get("split_merged_columns", True))
        self.expected_card_width_px = int(cfg.get("expected_card_width_px", 245))
        self.max_card_width_px = int(cfg.get("max_card_width_px", 420))
        self.gutter_v_max = int(cfg.get("gutter_v_max", 90))
        self.ignore_bottom_fraction = float(cfg.get("ignore_bottom_fraction", 0.045))

        self._slots: Dict[int, CardSlot] = {}
        self._next_slot_id = 1

    # ------------------------------------------------------------------ find

    def screen_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Filled mask of the KDS desktop (the dark navy area behind the cards).

        Cards are pale, and so are the wall behind the monitor and the monitor's
        own bezel.  Masking on paleness alone therefore fuses card + wall into
        one blob.  Restricting the search to the dark desktop region removes the
        surroundings entirely, which is what makes card detection work on
        phone-recorded footage.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dark = cv2.inRange(
            hsv,
            np.array([0, 0, 0], np.uint8),
            np.array([179, 255, self.screen_v_max], np.uint8),
        )
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < frame.shape[0] * frame.shape[1] * self.min_screen_fraction:
            return None
        filled = np.zeros(frame.shape[:2], np.uint8)
        # Filling the desktop contour re-includes the cards, which are holes in it.
        cv2.drawContours(filled, [cv2.convexHull(largest)], -1, 255, thickness=cv2.FILLED)
        return filled

    def find_card_boxes(self, frame: np.ndarray) -> List[BBox]:
        """Raw card rectangles in one frame, top-to-bottom then left-to-right."""
        if frame is None or frame.size == 0:
            return []
        height, width = frame.shape[:2]
        frame_area = float(height * width)
        # The KDS status bar along the bottom is pale and would look like a card.
        usable_height = int(height * (1.0 - self.ignore_bottom_fraction))

        mask = self.bands.card_body_mask(frame)
        mask[usable_height:, :] = 0
        screen = self.screen_mask(frame)
        if screen is not None:
            mask = cv2.bitwise_and(mask, screen)
        kernel = np.ones((self.close_kernel, self.close_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw: List[Tuple[BBox, float]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < self.min_width_px or h < self.min_height_px:
                continue
            rectangularity = cv2.contourArea(contour) / float(max(1, w * h))
            raw.append(((x, y, x + w, y + h), rectangularity))

        boxes: List[BBox] = []
        for box, rectangularity in raw:
            for piece in self._split_columns(frame, box):
                px1, py1, px2, py2 = piece
                pw, ph = px2 - px1, py2 - py1
                if pw < self.min_width_px or ph < self.min_height_px:
                    continue
                area_fraction = (pw * ph) / frame_area
                if not (self.min_area_fraction <= area_fraction <= self.max_area_fraction):
                    continue
                aspect = ph / float(pw)
                if not (self.min_aspect <= aspect <= self.max_aspect):
                    continue
                # A ticket card is a filled rectangle; reject ragged blobs.
                # Splitting only ever removes area, so the parent's ratio is a
                # valid lower bound for each piece.
                if len(raw) and rectangularity < self.min_rectangularity:
                    continue
                boxes.append(piece)

        # Drop boxes fully contained in a larger one (card border + card body).
        boxes = _drop_contained(boxes)
        boxes.sort(key=lambda b: (b[1], b[0]))
        return boxes

    def _split_columns(self, frame: np.ndarray, box: BBox) -> List[BBox]:
        """Split a blob that fused several side-by-side cards.

        The KDS draws cards edge to edge with only a thin dark border, so one
        colour mask blob routinely covers a whole row of tickets.  Cards are
        separated on the dark vertical gutters between them; a blob narrow
        enough to be a single card is returned unchanged.
        """
        x1, y1, x2, y2 = box
        width = x2 - x1
        if not self.split_merged_columns or width <= self.max_card_width_px:
            return [box]

        strip = frame[y1:y2, x1:x2]
        grey = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        # A gutter column is dark for essentially its whole height.
        dark_ratio = (grey < self.gutter_v_max).mean(axis=0)
        is_gutter = dark_ratio > 0.80

        # Collapse runs of gutter columns to their centres.
        cuts: List[int] = []
        run_start: Optional[int] = None
        for i, gutter in enumerate(is_gutter):
            if gutter and run_start is None:
                run_start = i
            elif not gutter and run_start is not None:
                cuts.append((run_start + i) // 2)
                run_start = None
        if run_start is not None:
            cuts.append((run_start + len(is_gutter)) // 2)

        edges = [0] + [c for c in cuts if 0 < c < width] + [width]
        pieces: List[BBox] = []
        for left, right in zip(edges[:-1], edges[1:]):
            if right - left < self.min_width_px:
                continue
            pieces.append((x1 + left, y1, x1 + right, y2))

        if not pieces:
            return [box]
        # If the gutters did not actually break the blob up, fall back to an
        # even split on the expected card pitch rather than returning a
        # multi-card box that would parse as one nonsensical ticket.
        if len(pieces) == 1 and width > self.max_card_width_px:
            count = max(2, int(round(width / float(self.expected_card_width_px))))
            step = width / float(count)
            return [
                (x1 + int(i * step), y1, x1 + int((i + 1) * step), y2)
                for i in range(count)
            ]
        return pieces

    def detect(self, frame: np.ndarray, timestamp: float) -> List[DetectedCard]:
        """Find cards and assign each to a persistent slot."""
        boxes = self.find_card_boxes(frame)
        results: List[DetectedCard] = []
        claimed: set = set()

        for box in boxes:
            slot = self._match_slot(box, claimed)
            if slot is None:
                slot = CardSlot(
                    slot_id=self._next_slot_id,
                    bbox=box,
                    first_seen=timestamp,
                    last_seen=timestamp,
                )
                self._slots[slot.slot_id] = slot
                self._next_slot_id += 1
            else:
                slot.update(box, timestamp)
            claimed.add(slot.slot_id)

            x1, y1, x2, y2 = slot.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            results.append(
                DetectedCard(
                    slot_id=slot.slot_id,
                    bbox=(x1, y1, x2, y2),
                    image=frame[y1:y2, x1:x2].copy(),
                    ticket_id=slot.ticket_id,
                )
            )
        return results

    def _match_slot(self, box: BBox, claimed: set) -> Optional[CardSlot]:
        best: Optional[CardSlot] = None
        best_iou = self.slot_iou_threshold
        for slot in self._slots.values():
            if slot.slot_id in claimed:
                continue
            score = iou(slot.bbox, box)
            if score >= best_iou:
                best_iou = score
                best = slot
        return best

    # ----------------------------------------------------------------- slots

    def bind_ticket(self, slot_id: int, ticket_id: str) -> None:
        """Remember which ticket a slot holds, so identity survives OCR gaps."""
        slot = self._slots.get(slot_id)
        if slot is not None:
            slot.ticket_id = ticket_id

    def slot_for_ticket(self, ticket_id: str) -> Optional[CardSlot]:
        for slot in self._slots.values():
            if slot.ticket_id == ticket_id:
                return slot
        return None

    def prune(self, now: float, max_age_s: float) -> List[CardSlot]:
        """Forget slots not seen for ``max_age_s``; returns the removed slots."""
        stale = [s for s in self._slots.values() if now - s.last_seen > max_age_s]
        for slot in stale:
            self._slots.pop(slot.slot_id, None)
        return stale

    @property
    def slots(self) -> Dict[int, CardSlot]:
        return dict(self._slots)

    def reset(self) -> None:
        self._slots.clear()
        self._next_slot_id = 1


def _drop_contained(boxes: List[BBox]) -> List[BBox]:
    keep: List[BBox] = []
    for i, box in enumerate(boxes):
        contained = False
        for j, other in enumerate(boxes):
            if i == j:
                continue
            if (
                box[0] >= other[0] - 2
                and box[1] >= other[1] - 2
                and box[2] <= other[2] + 2
                and box[3] <= other[3] + 2
                and _area(box) < _area(other)
            ):
                contained = True
                break
        if not contained:
            keep.append(box)
    return keep


def _area(box: BBox) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])
