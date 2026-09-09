"""Turn one ticket-card image into a :class:`TicketSnapshot`.

Implements sections 1, 2, 7, 10 and 22:

* ticket ID and order type from the header,
* payment status from the green total bar (``Subtotal`` vs ``*** Paid ***``),
* hotdog items from **yellow** bars only (RULE 2),
* **grey** text attached to the closest preceding yellow bar (RULE 3),
* unknown shortcuts reported, never mapped (RULE 4).

Colour is never the sole input: rows are also filtered by their position in the
card, by their text content, and by whether the shortcut resolves against the
configured vocabulary.  The snapshot produced here is a single observation and
is meant to be fed to :mod:`src.kds.stability`, not acted on directly.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.kds.colors import ColorBands
from src.kds.ocr_engine import OcrBox, OcrEngine
from src.kds.schemas import (
    ADDON_ROLES,
    IGNORED_ITEM_ROLES,
    AddOn,
    HotdogGroup,
    PaymentStatus,
    RowColor,
    TicketLine,
    TicketSnapshot,
)
from src.kds.shortcut_map import ShortcutMapper, normalize, split_quantity

logger = logging.getLogger(__name__)

# "CHK 424", "CHK424", "CHK  424", and the common OCR drift CHl(/CH1) 424.
_TICKET_ID_RE = re.compile(r"\bCH[KL1I]\s*[.:]?\s*(\d{1,6})\b", re.IGNORECASE)
# Any bare 3-6 digit run, used only as a last resort on the header row.
_BARE_ID_RE = re.compile(r"\b(\d{3,6})\b")
# "*** Paid *** 14.12" -- OCR frequently returns "Pald", "Pai d", "Paíd".
_PAID_RE = re.compile(r"\bP\s*[Aa4][ILl1]\s*[Dd]\b|\bPAID\b", re.IGNORECASE)
_SUBTOTAL_RE = re.compile(r"\bSUB\s*T[O0]T[A4]L\b|\bSUBTOTAL\b", re.IGNORECASE)
_ORDER_TYPE_RE = re.compile(r"\b(DINE\s*IN|DRIVE\s*THRU|TO\s*GO|CARRY\s*OUT)\b", re.IGNORECASE)
# Footer: "6259 Harjot", "UWS: 2", "Count 3".
_FOOTER_RE = re.compile(r"\bUWS\b|\bCOUNT\b", re.IGNORECASE)
# A row that is nothing but a quantity, e.g. the "2" of a wrapped "2 Ketchup".
_BARE_QTY_RE = re.compile(r"^\s*[-\s]*(\d{1,3})\s*$")
# Leading list bullet the KDS draws before each item line.
_BULLET_RE = re.compile(r"^\s*[-–—•]\s*")


@dataclass
class ParsedRow:
    """One logical row: merged OCR boxes plus a resolved colour role."""

    text: str
    bbox: Tuple[int, int, int, int]
    color: RowColor
    confidence: float
    fractions: dict

    @property
    def y_center(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0


class TicketParser:
    def __init__(
        self,
        ocr: OcrEngine,
        bands: Optional[ColorBands] = None,
        mapper: Optional[ShortcutMapper] = None,
        config: Optional[dict] = None,
    ):
        self.ocr = ocr
        self.bands = bands or ColorBands(config)
        self.mapper = mapper or ShortcutMapper()
        cfg = self.bands.config
        ocr_cfg = cfg.get("ocr", {}) or {}
        self.upscale_to_height = int(ocr_cfg.get("upscale_to_height", 900))
        self.min_confidence = float(ocr_cfg.get("min_confidence", 0.35))
        rows_cfg = cfg.get("rows", {}) or {}
        self.merge_tolerance_frac = float(rows_cfg.get("merge_tolerance_frac", 0.02))
        # A card this yellow is entirely highlighted, so the yellow-bar rule
        # cannot discriminate; see _hotdog_rows_for_highlighted_card.
        self.whole_card_yellow_threshold = float(
            (cfg.get("card_detection", {}) or {}).get("whole_card_yellow_threshold", 0.45)
        )

    # ------------------------------------------------------------------ OCR

    def _read_rows(self, card: np.ndarray) -> List[ParsedRow]:
        height = card.shape[0]
        scale = 1.0
        if self.upscale_to_height and height < self.upscale_to_height:
            scale = self.upscale_to_height / float(height)
            image = cv2.resize(card, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            image = card

        boxes = [b for b in self.ocr.read(image) if b.confidence >= self.min_confidence]
        if scale != 1.0:
            boxes = [b.scaled(scale, scale) for b in boxes]
        boxes.sort(key=lambda b: (b.cy, b.bbox[0]))

        rows = self._group_into_rows(boxes, card)
        rows = self._merge_continuations(rows, card)
        return rows

    def _group_into_rows(self, boxes: Sequence[OcrBox], card: np.ndarray) -> List[ParsedRow]:
        """Group OCR boxes that sit on the same horizontal line."""
        rows: List[List[OcrBox]] = []
        for box in boxes:
            placed = False
            for row in rows:
                ref = row[0]
                overlap = min(box.bbox[3], ref.bbox[3]) - max(box.bbox[1], ref.bbox[1])
                if overlap > 0.5 * min(box.height, ref.height):
                    row.append(box)
                    placed = True
                    break
            if not placed:
                rows.append([box])

        parsed: List[ParsedRow] = []
        for row in rows:
            row.sort(key=lambda b: b.bbox[0])
            x1 = min(b.bbox[0] for b in row)
            y1 = min(b.bbox[1] for b in row)
            x2 = max(b.bbox[2] for b in row)
            y2 = max(b.bbox[3] for b in row)
            text = " ".join(b.text for b in row).strip()
            confidence = sum(b.confidence for b in row) / len(row)
            color, fractions = self.bands.classify_row(card, (x1, y1, x2, y2))
            parsed.append(ParsedRow(text, (x1, y1, x2, y2), color, confidence, fractions))
        parsed.sort(key=lambda r: r.y_center)
        return parsed

    def _merge_continuations(self, rows: List[ParsedRow], card: np.ndarray) -> List[ParsedRow]:
        """Join a row that is a visual continuation of the one above it.

        The KDS wraps long lines: ``"2 Org" / "Chicgo"`` and ``"2" /
        "Ketchup"`` are each one logical item.

        Merging is deliberately conservative, because consecutive add-ons look
        exactly like a wrapped line: ``"Ketchup" / "Relish" / "Tomato"`` are
        three separate add-ons, not one.  A row is therefore only joined to the
        one above when either

        * the previous row is nothing but a quantity (``"2"`` + ``"Ketchup"``), or
        * the row does **not** resolve to anything on its own while the joined
          text does (``"2 Org"`` + ``"Chicgo"``).

        Anything that already resolves stands alone.
        """
        if not rows:
            return rows
        merged: List[ParsedRow] = [rows[0]]
        for row in rows[1:]:
            prev = merged[-1]
            gap = row.bbox[1] - prev.bbox[3]
            prev_h = max(1, prev.bbox[3] - prev.bbox[1])
            prev_is_bare_qty = bool(_BARE_QTY_RE.match(_BULLET_RE.sub("", prev.text)))
            starts_new_qty = bool(
                re.match(r"^\s*[-–—•]?\s*\d{1,3}\s*[xX]?\s+\S", row.text)
            )
            same_color = row.color is prev.color
            close = gap < 0.9 * prev_h

            joined = (prev.text + " " + row.text).strip()
            wraps = (
                not self._resolves(row.text)
                and not self._resolves(prev.text)
                and self._resolves(joined)
            )
            should_merge = prev_is_bare_qty or (wraps and not starts_new_qty)

            if same_color and close and should_merge:
                # Only merge item-like rows; never fuse the header or the
                # total bar into a product line.
                if not _is_structural(prev.text) and not _is_structural(row.text):
                    x1 = min(prev.bbox[0], row.bbox[0])
                    y1 = min(prev.bbox[1], row.bbox[1])
                    x2 = max(prev.bbox[2], row.bbox[2])
                    y2 = max(prev.bbox[3], row.bbox[3])
                    color, fractions = self.bands.classify_row(card, (x1, y1, x2, y2))
                    merged[-1] = ParsedRow(
                        text=joined,
                        bbox=(x1, y1, x2, y2),
                        color=color,
                        confidence=min(prev.confidence, row.confidence),
                        fractions=fractions,
                    )
                    continue
            merged.append(row)
        return merged

    def _resolves(self, text: str) -> bool:
        """True if ``text`` means something on its own.

        A line that already resolves to a supported hotdog, a known add-on, or
        a known non-hotdog product is a complete line and must never be fused
        into its neighbour.
        """
        cleaned = _BULLET_RE.sub("", text).strip()
        if not cleaned or _is_structural(cleaned):
            return False
        if _BARE_QTY_RE.match(cleaned):
            return False
        if self.mapper.is_non_hotdog_line(cleaned):
            return True
        if self.mapper.resolve_shortcut(cleaned, log_unknown=False).known:
            return True
        return self.mapper.resolve_addon(cleaned, log_unknown=False).known

    # --------------------------------------------------------------- parsing

    def parse(
        self,
        card: np.ndarray,
        timestamp: float,
        frame_index: int = -1,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        fallback_ticket_id: Optional[str] = None,
    ) -> TicketSnapshot:
        """Parse one ticket card image into a snapshot."""
        pink_fraction = self.bands.pink_fraction(card)
        rows = self._read_rows(card)

        if not rows:
            # The card is visibly there but nothing could be read.  That is an
            # unreadable observation, NOT a disappearance -- the distinction is
            # what preserves ticket identity through an OCR dropout (test 10).
            return TicketSnapshot(
                ticket_id=fallback_ticket_id or "",
                payment=PaymentStatus.UNKNOWN,
                bbox=bbox,
                pink_fraction=pink_fraction,
                timestamp=timestamp,
                frame_index=frame_index,
                readable=False,
            )

        ticket_id = self._extract_ticket_id(rows) or (fallback_ticket_id or "")
        order_type = self._extract_order_type(rows)
        payment, total_text = self._extract_payment(rows)
        hotdogs, other_lines, unknown = self._extract_items(rows, card)

        return TicketSnapshot(
            ticket_id=ticket_id,
            payment=payment,
            hotdogs=hotdogs,
            other_lines=other_lines,
            unknown_shortcuts=unknown,
            bbox=bbox,
            order_type=order_type,
            total_text=total_text,
            pink_fraction=pink_fraction,
            timestamp=timestamp,
            frame_index=frame_index,
            readable=True,
        )

    # ------------------------------------------------------------- extractors

    @staticmethod
    def _extract_ticket_id(rows: Sequence[ParsedRow]) -> str:
        for row in rows:
            match = _TICKET_ID_RE.search(row.text)
            if match:
                return "CHK " + match.group(1)
        # Fall back to a bare number on the topmost row only; the footer also
        # holds digits (server number, UWS) and must never be mistaken for an ID.
        if rows:
            header = rows[0]
            if not _FOOTER_RE.search(header.text):
                match = _BARE_ID_RE.search(header.text)
                if match:
                    return "CHK " + match.group(1)
        return ""

    @staticmethod
    def _extract_order_type(rows: Sequence[ParsedRow]) -> str:
        for row in rows:
            match = _ORDER_TYPE_RE.search(row.text)
            if match:
                return " ".join(match.group(1).split()).title()
        return ""

    @staticmethod
    def _extract_payment(rows: Sequence[ParsedRow]) -> Tuple[PaymentStatus, str]:
        """Read the green total bar.

        ``*** Paid *** 14.12`` means PAID; ``Subtotal 10.80`` means the ticket
        is NOT paid.  Content decides, not colour -- both states use the same
        green-on-black bar.  Anything unreadable stays UNKNOWN, which the
        payment gate treats exactly like NOT PAID (RULE 1).
        """
        candidates = [r for r in rows if r.color is RowColor.PAID_GREEN] or list(rows)
        # Prefer rows that actually look like a total bar.
        for row in candidates:
            if _SUBTOTAL_RE.search(row.text):
                return PaymentStatus.NOT_PAID, row.text
        for row in candidates:
            if _PAID_RE.search(row.text):
                return PaymentStatus.PAID, row.text
        # Re-check every row: the green bar sometimes merges with "Count".
        for row in rows:
            if _SUBTOTAL_RE.search(row.text):
                return PaymentStatus.NOT_PAID, row.text
        for row in rows:
            if _PAID_RE.search(row.text):
                return PaymentStatus.PAID, row.text
        return PaymentStatus.UNKNOWN, ""

    def _extract_items(
        self, rows: Sequence[ParsedRow], card: np.ndarray
    ) -> Tuple[List[HotdogGroup], List[TicketLine], List[str]]:
        """Build the hotdog -> add-on tree.

        Grey (or orange) text belongs to the closest preceding **product** line
        and keeps belonging to it until another product line starts (section 7).

        Crucially, the preceding product is not necessarily a hotdog: a real
        ticket reads ``1 AB C/C`` (yellow) / ``1 Dlx Ch Brg`` (magenta) /
        ``No Must`` (grey), where ``No Must`` modifies the *burger*.  Attaching
        it to the last hotdog instead would invent an order requirement that
        the customer never made, so add-ons under a non-hotdog product are
        dropped rather than reassigned.
        """
        yellow_fraction = self.bands.fraction(card, "yellow")
        highlighted = yellow_fraction >= self.whole_card_yellow_threshold

        hotdogs: List[HotdogGroup] = []
        other: List[TicketLine] = []
        unknown: List[str] = []
        current: Optional[HotdogGroup] = None
        # True when the most recent product line was NOT a hotdog, so any
        # add-on that follows belongs to that product and not to `current`.
        parent_is_non_hotdog = False

        for row in rows:
            text = _BULLET_RE.sub("", row.text).strip()
            if not text or _is_structural(text):
                continue

            quantity, body = split_quantity(text)
            line = TicketLine(
                text=text,
                color=row.color,
                bbox=row.bbox,
                confidence=row.confidence,
                quantity=quantity,
                body=body,
            )

            if row.color in ADDON_ROLES:
                if current is not None and not parent_is_non_hotdog:
                    current.addons.append(self.mapper.resolve_addon(text).to_addon())
                else:
                    # Either no product precedes it, or the product it modifies
                    # is not a hotdog.  Either way it is never promoted to a
                    # standalone order item.
                    logger.debug("Add-on row not attached to a hotdog: %r", text)
                    other.append(line)
                continue

            if row.color in IGNORED_ITEM_ROLES:
                # Cyan drinks, blue tenders, magenta burgers: real products,
                # but not hotdogs, so they never enter the order group.
                other.append(line)
                parent_is_non_hotdog = True
                continue

            is_item_row = row.color is RowColor.YELLOW or (
                highlighted and row.color in (RowColor.PLAIN, RowColor.PINK)
            )
            if not is_item_row:
                # An unhighlighted product line (fries, corn dog).  It is still
                # a product, so it takes ownership of any add-on below it.
                other.append(line)
                parent_is_non_hotdog = True
                continue

            if self.mapper.is_non_hotdog_line(text):
                other.append(line)
                parent_is_non_hotdog = True
                continue

            match = self.mapper.resolve_shortcut(text)
            if not match.known and highlighted:
                # On a fully highlighted card the yellow bar carries no
                # information, so an unresolvable line is far more likely to be
                # a non-hotdog product than a mis-read hotdog.  Record it as a
                # plain line rather than raising a false UNKNOWN_SHORTCUT.
                other.append(line)
                parent_is_non_hotdog = True
                continue

            group = match.to_group()
            if group.is_unknown:
                unknown.append(text)
                logger.warning("UNKNOWN SHORTCUT on ticket card: %r", text)
            hotdogs.append(group)
            current = group
            parent_is_non_hotdog = False

        return hotdogs, other, unknown


def _is_structural(text: str) -> bool:
    """True for header / footer / total rows, which are never order items."""
    if _TICKET_ID_RE.search(text) or _ORDER_TYPE_RE.search(text):
        return True
    if _FOOTER_RE.search(text) or _SUBTOTAL_RE.search(text) or _PAID_RE.search(text):
        return True
    stripped = normalize(text)
    if not stripped:
        return True
    # A row of only digits / punctuation (the "ooo" priority dots, "0:38").
    if re.fullmatch(r"[\d\s:./o]*", text.strip(), flags=re.IGNORECASE):
        return True
    return False
