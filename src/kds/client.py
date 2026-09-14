import json
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Set

from src.domain.schemas import Ticket, LineItem


class KDSClient(ABC):
    @abstractmethod
    def get_next_ticket(self) -> Optional[Ticket]:
        pass

    def mark_completed(self, ticket_id: str) -> None:
        """Called when an order for this ticket completes successfully."""
        pass

    def mark_abandoned(self, ticket_id: str) -> None:
        """Called when an order for this ticket is abandoned."""
        pass

    def get_all_tickets(self) -> List[Ticket]:
        return []


class NullKDSClient(KDSClient):
    """Detection-only mode (KDS_MODE=none).

    Never issues a ticket, so the state machine never opens an order and no
    order-reconciliation, OCR, KDS-video or failure-recording path is ever
    constructed.  Used to benchmark the raw capture -> detect -> track loop in
    isolation.  Every other method is inherited from KDSClient, whose defaults
    are already no-ops returning empty.
    """

    def get_next_ticket(self) -> Optional[Ticket]:
        return None


class MockKDSClient(KDSClient):
    def __init__(self, path: str, poll_interval: int = 2, loop: bool = False):
        self.path = path
        self.poll_interval = poll_interval
        self.loop = loop
        self._last_index = -1
        self._last_mtime = 0.0
        self._last_poll = 0.0
        self._tickets: List[Ticket] = []
        self._completed: Set[str] = set()
        self._abandoned: Set[str] = set()
        self._load_tickets()

    def _load_tickets(self):
        p = Path(self.path)
        if not p.exists():
            return
        data = json.loads(p.read_text())

        # Support both flat list and dict with "tickets" key
        if isinstance(data, dict):
            raw_tickets = data.get("tickets", [])
        else:
            raw_tickets = data

        self._tickets = []
        for t in raw_tickets:
            line_items = []
            for li in t.get("line_items", []):
                line_items.append(LineItem(
                    variant=li["variant"],
                    count=li["count"],
                    items=li["items"]
                ))
            
            hotdog_specs = {}
            if "hotdog_specs" in t and isinstance(t["hotdog_specs"], dict):
                hotdog_specs.update(t["hotdog_specs"])
            for k, v in t.items():
                if k.startswith("hotdog") and k not in ("total_hotdogs", "hotdog_specs"):
                    if isinstance(v, list):
                        item_counts = {}
                        for item in v:
                            item_counts[item] = item_counts.get(item, 0) + 1
                        hotdog_specs[k] = item_counts
                    elif isinstance(v, dict):
                        hotdog_specs[k] = v
            
            if hotdog_specs and not line_items:
                for hd_name, items_dict in hotdog_specs.items():
                    line_items.append(LineItem(variant=hd_name, count=1, items=items_dict))
                
            total_hotdogs = t.get("total_hotdogs", len(hotdog_specs) if hotdog_specs else 1)


            expected_items = t.get("expected_items", [])

            self._tickets.append(Ticket(
                ticket_id=t["ticket_id"],
                shortcut=t.get("shortcut", ""),
                total_hotdogs=total_hotdogs,
                line_items=line_items,
                hotdog_specs=hotdog_specs,
                expected_items=expected_items,
            ))

        self._last_index = -1

    def get_next_ticket(self) -> Optional[Ticket]:
        now = time.monotonic()
        if now - self._last_poll < self.poll_interval:
            return None

        self._last_poll = now
        p = Path(self.path)

        if not p.exists():
            return None

        mtime = p.stat().st_mtime
        if mtime != self._last_mtime:
            self._last_mtime = mtime
            self._load_tickets()

        if self.loop:
            if len(self._tickets) > 0:
                self._last_index = (self._last_index + 1) % len(self._tickets)
                return self._tickets[self._last_index]
        else:
            if self._last_index + 1 < len(self._tickets):
                self._last_index += 1
                return self._tickets[self._last_index]

        return None

    def mark_completed(self, ticket_id: str) -> None:
        self._completed.add(ticket_id)

    def mark_abandoned(self, ticket_id: str) -> None:
        self._abandoned.add(ticket_id)

    def get_all_tickets(self) -> List[Ticket]:
        return self._tickets



class DynamicKDSClient(KDSClient):
    """Generates random tickets from available zone names for sustained testing."""

    def __init__(
        self,
        zone_names: List[str],
        min_items: int = 2,
        max_items: int = 5,
        interval_range: tuple = (5, 15),
        max_tickets: int = 0,
        seed: Optional[int] = None,
        prefix: str = "ORD",
    ):
        self.zone_names = zone_names
        self.min_items = min_items
        self.max_items = max_items
        self.interval_range = interval_range
        self.max_tickets = max_tickets
        self.prefix = prefix
        self._rng = random.Random(seed)
        self._next_ticket_time = time.monotonic()
        self._ticket_counter = 0
        self._tickets_yielded = 0
        self._completed: Set[str] = set()
        self._abandoned: Set[str] = set()

    def get_next_ticket(self) -> Optional[Ticket]:
        now = time.monotonic()
        if now < self._next_ticket_time:
            return None

        if self.max_tickets > 0 and self._tickets_yielded >= self.max_tickets:
            return None

        n_items = self._rng.randint(self.min_items, self.max_items)
        items = self._rng.choices(self.zone_names, k=n_items)
        self._ticket_counter += 1
        ticket_id = f"{self.prefix}-{self._ticket_counter:04d}"

        self._tickets_yielded += 1
        delay = self._rng.uniform(*self.interval_range)
        self._next_ticket_time = now + delay

        return Ticket(ticket_id=ticket_id, expected_items=items)

    def mark_completed(self, ticket_id: str) -> None:
        self._completed.add(ticket_id)

    def mark_abandoned(self, ticket_id: str) -> None:
        self._abandoned.add(ticket_id)
