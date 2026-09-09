"""KDS video understanding: OCR ticket extraction, FIFO order grouping and
validation against the production-video detection pipeline.

The production pipeline (``src/detector.py``, ``src/hotdog_tracker.py``,
``src/wrapping_state.py``, ``src/zones.py``, ``src/temporal.py``) is reused
unchanged.  This package only adds the second input -- the KDS screen -- and
the order lifecycle built on top of it.
"""

from src.kds.schemas import (
    AddOn,
    FailureCategory,
    HotdogGroup,
    KdsEvent,
    KdsEventType,
    LifecycleState,
    OrderGroup,
    TicketLine,
    TicketSnapshot,
    ValidationResult,
)

__all__ = [
    "AddOn",
    "FailureCategory",
    "HotdogGroup",
    "KdsEvent",
    "KdsEventType",
    "LifecycleState",
    "OrderGroup",
    "TicketLine",
    "TicketSnapshot",
    "ValidationResult",
]
