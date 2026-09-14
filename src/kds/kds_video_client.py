"""Bridge the KDS video reader into the existing pipeline.

:class:`KDSVideoClient` implements the same ``src.kds.client.KDSClient``
interface as ``MockKDSClient``, so ``OrderStateMachine``, the dashboard and
``src/main.py`` keep working unchanged -- they simply receive tickets that came
from OCR of a real KDS screen instead of from a JSON file.

The KDS video runs on its own :class:`~src.video.capture.VideoCaptureThread`, decoupled
from the production video, and both streams stamp events with the same
``time.monotonic`` clock.  A fixed ``sync.kds_offset_s`` in
``config/kds_visual.yaml`` corrects a known capture start difference.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, List, Optional

from src.video.capture import VideoCaptureThread
from src.kds.colors import load_visual_config
from src.kds.fifo_queue import TicketManager
from src.kds.kds_monitor import KdsMonitor
from src.kds.timeline import EventTimeline
from src.kds.client import KDSClient
from src.domain.schemas import Ticket

logger = logging.getLogger(__name__)


class KDSVideoClient(KDSClient):
    """Reads a KDS video/stream and hands confirmed PAID tickets downstream."""

    def __init__(
        self,
        source,
        config_path: Optional[str] = None,
        timeline_path: Optional[str] = None,
        realtime: bool = False,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        monitor: Optional[KdsMonitor] = None,
        start: bool = True,
    ):
        self.source = source
        self.config = load_visual_config(config_path)
        self.timeline = EventTimeline(timeline_path)
        self.manager = TicketManager()
        self.monitor = monitor or KdsMonitor(
            config=self.config, manager=self.manager, timeline=self.timeline
        )
        # A monitor supplied by the caller owns its own manager/timeline.
        self.manager = self.monitor.manager
        self.timeline = self.monitor.timeline

        self._capture = VideoCaptureThread(
            source,
            target_width=target_width,
            target_height=target_height,
            realtime=realtime,
        )
        self._thread: Optional[threading.Thread] = None
        # Media time of the production video, published by src/main.py each
        # frame.  The KDS reader will not run ahead of it, which is what keeps
        # the two videos aligned: the production loop is throttled by YOLO to
        # well under real time, while this reader is not, so left alone the KDS
        # screen races minutes ahead of the kitchen it is meant to describe.
        self._master_time: Optional[float] = None
        # How far ahead the KDS reader may get before it waits, in seconds of
        # video time.  Small but non-zero so it does not thrash on every frame.
        self._lead_tolerance_s = 0.05
        # Drift reporting.  The hold above stops the reader running ahead, but
        # nothing stops it falling behind if OCR cannot keep up with the
        # production loop, and a reader that lags is just as misaligned.
        self._lag_warn_after_s = 2.0
        self._last_lag_warn = 0.0
        self.max_lag_s = 0.0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        # Tickets confirmed PAID and not yet handed to OrderStateMachine.
        self._pending: List[Ticket] = []
        self._delivered: set = set()
        self._ended = threading.Event()
        # Extra listeners for TicketManager events (the failure recorder uses
        # this to learn when a ticket starts and how it ended).
        self._listeners: List[Callable[[str, dict], None]] = []

        self.monitor.manager._on_event = self._on_manager_event
        # Stream the annotated KDS screen to the dashboard's KDS panel.
        self.monitor.on_annotated_frame = self._publish_frame
        if start:
            self.start()

    # ------------------------------------------------------------------ thread

    def start(self) -> None:
        if self._thread is not None:
            return
        self._capture.start()
        self._thread = threading.Thread(
            target=self._run, name="kds-video-reader", daemon=True
        )
        self._thread.start()
        logger.info("KDS video reader started on %s", self.source)

    def set_master_time(self, timestamp: Optional[float]) -> None:
        """Publish the production video's current media time.

        Called once per processed frame by src/main.py.  Without it the two
        readers advance independently and drift apart by however much slower
        the detector is than the KDS reader.
        """
        self._master_time = timestamp

    def _run(self) -> None:
        held = None
        while not self._stop.is_set():
            item = held or self._capture.get_frame()
            held = None
            if item is None:
                if self._capture.consume_loop():
                    continue
                self._ended.set()
                break
            frame, media_time = item
            if frame is None:
                continue

            # Hold this frame until the production video reaches it.  The frame
            # is kept rather than dropped, so nothing on the KDS is skipped.
            master = self._master_time
            if (
                master is not None
                and media_time is not None
                and media_time > master + self._lead_tolerance_s
            ):
                held = (frame, media_time)
                time.sleep(0.005)
                continue

            # Stamp events in video time, not wall-clock time, so the KDS
            # timeline lines up with the production video and stays correct
            # however fast or slow the pipeline happens to run.
            if master is not None and media_time is not None:
                lag = master - media_time
                if lag > self.max_lag_s:
                    self.max_lag_s = lag
                if lag > self._lag_warn_after_s and (
                    time.monotonic() - self._last_lag_warn > 10.0
                ):
                    self._last_lag_warn = time.monotonic()
                    logger.warning(
                        "KDS reader is %.1fs behind the production video "
                        "(raise ocr.frame_stride in config/kds_visual.yaml if "
                        "this keeps growing)",
                        lag,
                    )

            stamp = media_time if media_time is not None else time.monotonic()
            try:
                self.monitor.process_frame(frame, stamp)
            except Exception:  # pragma: no cover - a bad frame must not kill it
                logger.exception("KDS frame processing failed")
        logger.info("KDS video reader finished")

    def stop(self) -> None:
        self._stop.set()
        try:
            self._capture.release()
        except Exception:  # pragma: no cover
            logger.debug("KDS capture release failed", exc_info=True)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def finished(self) -> bool:
        return self._ended.is_set()

    @staticmethod
    def _publish_frame(frame) -> None:
        from src import dashboard

        dashboard.update_kds_frame(frame)

    # ------------------------------------------------------------------ events

    def add_listener(self, callback: Callable[[str, dict], None]) -> None:
        """Subscribe to TicketManager events ("created", "finalized", ...)."""
        self._listeners.append(callback)

    def _on_manager_event(self, kind: str, payload: dict) -> None:
        """Queue a Ticket the moment its order group is created.

        Only "created" produces a Ticket.  A ticket's requirement is frozen at
        creation, so a later KDS reading is recorded on the timeline but never
        handed downstream.
        """
        for listener in self._listeners:
            try:
                listener(kind, payload)
            except Exception:  # pragma: no cover - a listener must not break us
                logger.exception("KDS listener failed for %s", kind)
        if kind != "created":
            return
        ticket_id = payload.get("ticket_id", "")
        group = self.manager.get(ticket_id)
        if group is None:
            return
        with self._lock:
            if ticket_id in self._delivered:
                return
            self._delivered.add(ticket_id)
            self._pending.append(group.to_ticket())
        logger.info("Ticket %s queued for the order state machine", ticket_id)

    # ------------------------------------------------------------- KDSClient API

    def get_next_ticket(self) -> Optional[Ticket]:
        with self._lock:
            if self._pending:
                return self._pending.pop(0)
        return None

    def get_all_tickets(self) -> List[Ticket]:
        return [g.to_ticket() for g in self.manager.queue]

    def mark_completed(self, ticket_id: str) -> None:
        logger.debug("OrderStateMachine reported %s completed", ticket_id)

    def mark_abandoned(self, ticket_id: str) -> None:
        logger.debug("OrderStateMachine reported %s abandoned", ticket_id)

    # ------------------------------------------------- production-side plumbing

    def record_hotdog(
        self,
        item: str = "hot-dog",
        track_id: Optional[int] = None,
        confidence: float = 0.0,
        now: Optional[float] = None,
    ):
        """Feed one temporally-confirmed production detection to the FIFO head."""
        return self.manager.record_detection(
            item=item, track_id=track_id, confidence=confidence, now=now
        )

    def dashboard_state(self) -> dict:
        return self.monitor.dashboard_state()

    @property
    def has_active_ticket(self) -> bool:
        """True while any paid ticket is still waiting to be produced.

        ``src/main.py`` uses this to idle the detector when the KDS is empty --
        with nothing ordered there is nothing to validate, so there is no
        reason to keep running YOLO at full rate.
        """
        return self.manager.active_group is not None

    @property
    def has_screen_content(self) -> bool:
        """True while the KDS screen shows anything at all.

        This is the gate ``src/main.py`` throttles detection on.  It is
        deliberately wider than :attr:`has_active_ticket`: a card appears on
        the screen before it is confirmed paid, and production routinely
        starts during that window, so detection must already be running by
        the time the ticket activates.  With a genuinely blank screen there is
        nothing to validate and nothing to miss, so YOLO stops entirely and
        both feeds run at their native frame rate.
        """
        return self.monitor.last_card_count > 0 or self.has_active_ticket

    @property
    def annotated_kds_frame(self):
        """The most recent annotated KDS screen, for the recorded view."""
        return self.monitor.last_annotated_frame

    def finalize_all(self) -> None:
        self.monitor.finalize_all(time.monotonic())
