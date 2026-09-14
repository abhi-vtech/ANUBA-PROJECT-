"""KDS colour semantics (section 22).

Colour is loaded entirely from ``config/kds_visual.yaml`` so the bands can be
re-tuned for a different KDS theme or camera without a code change.

Nothing here decides anything on its own: the parser combines the colour role
returned by :meth:`ColorBands.classify_row` with the ticket bounding box, the
OCR text, relative row position, temporal persistence and ticket identity
before acting.  ``light_pink`` in particular only ever produces a *fraction*;
pink means the order is overdue, and is informational only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from src.kds.schemas import RowColor
from src.domain.paths import resource

logger = logging.getLogger(__name__)


@dataclass
class Band:
    """One HSV band, optionally a union of two ranges (for wrap-around hues)."""

    name: str
    lower: np.ndarray
    upper: np.ndarray
    lower2: Optional[np.ndarray] = None
    upper2: Optional[np.ndarray] = None
    min_fraction: float = 0.3

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        mask = cv2.inRange(hsv, self.lower, self.upper)
        if self.lower2 is not None and self.upper2 is not None:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, self.lower2, self.upper2))
        return mask

    def fraction(self, hsv: np.ndarray) -> float:
        """Share of pixels inside this band, 0..1."""
        if hsv is None or hsv.size == 0:
            return 0.0
        mask = self.mask(hsv)
        return float(np.count_nonzero(mask)) / float(mask.size)


def _arr(values) -> np.ndarray:
    return np.array([int(v) for v in values], dtype=np.uint8)


class ColorBands:
    """All KDS colour bands, plus row/card classification helpers."""

    #: Saturated highlight bars.  These are scored competitively rather than by
    #: fixed priority: several of the bands touch at the edges (orange 8-22 vs
    #: yellow 20-38), so the band with the strongest relative response wins.
    HIGHLIGHT_BANDS = ("yellow", "orange", "blue", "magenta", "cyan")

    #: Checked only after every highlight band has failed.  paid_green is text
    #: rather than fill, and grey is defined by the *absence* of saturation, so
    #: both would otherwise steal rows from the highlight bands.
    FALLBACK_BANDS = ("paid_green", "grey")

    def __init__(self, config: Optional[dict] = None, config_path: Optional[str] = None):
        if config is None:
            config = load_visual_config(config_path)
        self.config = config
        raw_colors = config.get("colors", {}) or {}
        self.bands: Dict[str, Band] = {}
        for name, spec in raw_colors.items():
            if "hsv_lower" not in spec:
                continue
            self.bands[name] = Band(
                name=name,
                lower=_arr(spec["hsv_lower"]),
                upper=_arr(spec["hsv_upper"]),
                lower2=_arr(spec["hsv_lower_2"]) if "hsv_lower_2" in spec else None,
                upper2=_arr(spec["hsv_upper_2"]) if "hsv_upper_2" in spec else None,
                min_fraction=float(spec.get("min_fraction", 0.3)),
            )
        rows_cfg = config.get("rows", {}) or {}
        self.color_sample_inset = float(rows_cfg.get("color_sample_inset_frac", 0.15))

    # ------------------------------------------------------------------ util

    def band(self, name: str) -> Optional[Band]:
        return self.bands.get(name)

    def fraction(self, bgr: np.ndarray, name: str) -> float:
        band = self.bands.get(name)
        if band is None or bgr is None or bgr.size == 0:
            return 0.0
        return band.fraction(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))

    # -------------------------------------------------------------- classify

    def classify_patch(self, bgr: np.ndarray) -> Tuple[RowColor, Dict[str, float]]:
        """Classify a background patch, returning the role and every fraction.

        The fractions are returned so callers can log *why* a row was classified
        the way it was -- essential when tuning bands against new footage.
        """
        if bgr is None or bgr.size == 0:
            return RowColor.PLAIN, {}
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        fractions = {name: band.fraction(hsv) for name, band in self.bands.items()}

        # Competitive scoring: how far past its own threshold each band got.
        best_name = None
        best_score = 1.0
        for name in self.HIGHLIGHT_BANDS:
            band = self.bands.get(name)
            if band is None or band.min_fraction <= 0:
                continue
            score = fractions.get(name, 0.0) / band.min_fraction
            if score >= 1.0 and score > best_score:
                best_score = score
                best_name = name
        if best_name is not None:
            return _ROLE_BY_BAND[best_name], fractions

        for name in self.FALLBACK_BANDS:
            band = self.bands.get(name)
            if band is None:
                continue
            if fractions.get(name, 0.0) >= band.min_fraction:
                return _ROLE_BY_BAND[name], fractions

        pink = self.bands.get("light_pink")
        if pink is not None and fractions.get("light_pink", 0.0) >= pink.min_fraction:
            return RowColor.PINK, fractions

        return RowColor.PLAIN, fractions

    def classify_row(
        self, card_bgr: np.ndarray, row_bbox: Tuple[int, int, int, int]
    ) -> Tuple[RowColor, Dict[str, float]]:
        """Classify one text row by sampling the strip of card it sits on.

        The sample is inset vertically so it cannot bleed into the row above or
        below, and widened horizontally to the full card so that a short piece
        of text still sees the whole coloured bar behind it.
        """
        if card_bgr is None or card_bgr.size == 0:
            return RowColor.PLAIN, {}
        height, width = card_bgr.shape[:2]
        x1, y1, x2, y2 = row_bbox
        row_h = max(1, y2 - y1)
        inset = int(round(row_h * self.color_sample_inset))
        sy1 = max(0, min(height - 1, y1 + inset))
        sy2 = max(sy1 + 1, min(height, y2 - inset))
        # Full card width: the coloured bar spans the card, the text does not.
        sx1 = max(0, int(width * 0.02))
        sx2 = max(sx1 + 1, int(width * 0.98))
        return self.classify_patch(card_bgr[sy1:sy2, sx1:sx2])

    def pink_fraction(self, card_bgr: np.ndarray) -> float:
        """Fraction of a WHOLE ticket card in the light-pink band.

        Reported for display only: a pink card is an OVERDUE card, not a
        finished one, so this never decides an order.
        """
        return self.fraction(card_bgr, "light_pink")

    def is_pink_on(self, card_bgr: np.ndarray) -> bool:
        band = self.bands.get("light_pink")
        if band is None:
            return False
        return self.pink_fraction(card_bgr) >= band.min_fraction

    def card_body_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Binary mask of pale card bodies (and pink cards, if enabled).

        Used by the ticket detector to find cards against the dark KDS desktop.
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        body = self.bands.get("card_body")
        mask = body.mask(hsv) if body else np.zeros(frame_bgr.shape[:2], np.uint8)
        card_cfg = self.config.get("card_detection", {}) or {}
        if card_cfg.get("accept_pink_cards", True) and "light_pink" in self.bands:
            mask = cv2.bitwise_or(mask, self.bands["light_pink"].mask(hsv))
        # A yellow-highlighted ticket (whole card yellow) is still a card.
        if "yellow" in self.bands:
            mask = cv2.bitwise_or(mask, self.bands["yellow"].mask(hsv))
        return mask


_ROLE_BY_BAND = {
    "yellow": RowColor.YELLOW,
    "grey": RowColor.GREY,
    "orange": RowColor.ORANGE,
    "blue": RowColor.BLUE,
    "magenta": RowColor.MAGENTA,
    "cyan": RowColor.CYAN,
    "paid_green": RowColor.PAID_GREEN,
    "light_pink": RowColor.PINK,
}


_CONFIG_CACHE: Dict[str, dict] = {}


def load_visual_config(path: Optional[str] = None) -> dict:
    """Load and cache ``config/kds_visual.yaml``."""
    resolved = path or resource("config/kds_visual.yaml")
    if resolved in _CONFIG_CACHE:
        return _CONFIG_CACHE[resolved]
    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        logger.error("KDS visual config not found: %s -- using defaults", resolved)
        config = {}
    _CONFIG_CACHE[resolved] = config
    return config
