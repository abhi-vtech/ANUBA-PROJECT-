import json
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import logging
from src.domain.schemas import (
    Action,
    Order,
    OrderStatus,
    Stats,
    Ticket,
    canonical_ingredient,
)
from src.analysis.batch_validator import BatchOrderValidator, normalize_item_name
from src.analysis.cheese_gate import CHEESE_KEYS
from src.domain.ingredient_config import is_granular

logger = logging.getLogger(__name__)

# Granular ingredients: multiple pinches within this window = 1 serving count.
# Tune after reviewing inter-pinch timestamps in logs.
SERVING_GAP_S: float = 3.5

# Onion: raised to 12s because small hovers were triggering false additions.
# Worker typically scoops onions in one batch; a 12s window treats the whole
# batch as one serving and prevents multiple short-hover increments.
ONION_SERVING_GAP_S: float = 4.0

# Discrete ingredients: rapid duplicate frame events within this window are ignored.
DISCRETE_DEBOUNCE_S: float = 1.8


def _extract_ticket_counts(ticket: Ticket) -> Tuple[List[str], Counter, int]:
    expected_items = []
    counts = Counter()
    
    total_hotdogs = getattr(ticket, "total_hotdogs", 1)

    if getattr(ticket, "hotdog_specs", None):
        for hd_name, items_val in ticket.hotdog_specs.items():
            if isinstance(items_val, (list, tuple)):
                for item in items_val:
                    counts[item] += 1
                    expected_items.append(item)
            elif isinstance(items_val, dict):
                for item, qty in items_val.items():
                    counts[item] += qty
                    expected_items.extend([item] * qty)
    elif getattr(ticket, "expected_items", None):
        for item in ticket.expected_items:
            counts[item] += 1
            expected_items.append(item)

    # line_items and hotdog_specs can describe the SAME hotdogs.  When they do,
    # counting both counts every ingredient twice -- the guard below used to
    # compare a variant name against hotdog_specs keys, which are "hotdog1",
    # "hotdog2", ... so it never matched and every KDS ticket was doubled.
    specs_cover = getattr(ticket, "specs_cover_line_items", False)
    for li in getattr(ticket, "line_items", []):
        if specs_cover and getattr(ticket, "hotdog_specs", None):
            continue
        if getattr(ticket, "hotdog_specs", None) and li.variant in ticket.hotdog_specs:
            continue
        for item, qty in li.items.items():
            counts[item] += qty * li.count
            expected_items.extend([item] * (qty * li.count))
    # Normalize item names
    final_counts = Counter()
    for item, qty in counts.items():
        norm = normalize_item_name(item)
        if norm == "hot_dog":
            norm = "hot-dog"
        final_counts[norm] += qty
    
    counts = final_counts
    expected_items = list(counts.keys())

    if (total_hotdogs == 1 or total_hotdogs == 0) and getattr(ticket, "hotdog_specs", None):
        total_hotdogs = len(ticket.hotdog_specs)

    if "hot-dog" not in counts or counts["hot-dog"] != total_hotdogs:
        expected_items = [i for i in expected_items if i != "hot-dog"]
        expected_items.extend(["hot-dog"] * total_hotdogs)
        counts["hot-dog"] = total_hotdogs

    return expected_items, counts, total_hotdogs


    return expected_items, counts, total_hotdogs


class OrderStateMachine:

    def __init__(self, history_path: Optional[str] = None):
        self.current_ticket: Optional[Ticket] = None
        self.current_order = Order(ticket_id="")
        self.stats = Stats()
        self.history: List[Order] = []
        self._validation_log: List[dict] = []
        self._kds_client: Optional[object] = None
        self._history_path = history_path
        self.orders: List[Order] = []
        self.batch_validator: Optional[BatchOrderValidator] = None
        # Discrete ingredients: suppress duplicate frame events within DISCRETE_DEBOUNCE_S
        self.last_debounced_ts: dict = {}
        # Granular ingredients: track last serving timestamp for SERVING_GAP_S window
        self.last_pinch_ts: dict = {}
        # Set by src/main.py when CheesePreGate is running.  While it is, cheese
        # is counted from cheese-region exit only (see apply_cheese_takes) and
        # the well-exit place event below is ignored, or every slice would be
        # counted twice.
        self.cheese_gate_owns_cheese: bool = False

        # Load history from JSONL file if it exists
        if history_path:
            p = Path(history_path)
            if p.exists():
                try:
                    with open(p, "r") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            record = json.loads(line)
                            ticket_id = record.get("ticket_id", "")
                            status_str = record.get("status", "completed")
                            status = OrderStatus(status_str) if status_str in [s.value for s in OrderStatus] else OrderStatus.COMPLETED
                            
                            order = Order(
                                ticket_id=ticket_id,
                                expected_items=record.get("expected_items", []),
                                picked_counts=record.get("picked_counts", {}),
                                status=status,
                            )
                            order.passed = record.get("passed", True)
                            order.missing_items = record.get("missing_items", [])
                            order.extra_items = record.get("extra_items", [])
                            order.applied_sauces = record.get("applied_sauces", [])
                            order.validation_message = record.get("result", "All ingredients are present" if order.passed else "All ingredients are not present")
                            
                            self.history.append(order)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"Error loading history from {history_path}: {e}")
            self._recalculate_stats()

    def _recalculate_stats(self):
        self.stats.total_orders = len(self.history)
        self.stats.passed_orders = sum(1 for o in self.history if o.passed)
        self.stats.failed_orders = self.stats.total_orders - self.stats.passed_orders

    def set_kds_client(self, client: object) -> None:
        """Set the KDS client for lifecycle feedback (mark_completed/mark_abandoned)."""
        self._kds_client = client
        if hasattr(client, "get_all_tickets"):
            tickets = client.get_all_tickets()
            
            # Start with unique tickets from history
            seen_history_ids = set()
            for hist_order in reversed(self.history):
                if hist_order.ticket_id in seen_history_ids:
                    continue
                seen_history_ids.add(hist_order.ticket_id)
                self.orders.insert(0, hist_order) # Insert history at beginning

            for ticket in tickets:
                expected_items, counts, total_hotdogs = _extract_ticket_counts(ticket)
                
                # Check if this ticket was completed in history
                hist_order = next((o for o in reversed(self.history) if o.ticket_id == ticket.ticket_id), None)
                status = OrderStatus.PENDING
                picked_counts = {}
                passed = True
                missing = []
                extra = []
                
                if hist_order:
                    status = hist_order.status
                    picked_counts = dict(hist_order.picked_counts)
                    passed = hist_order.passed
                    missing = list(hist_order.missing_items) if hist_order.missing_items else []
                    extra = list(hist_order.extra_items) if hist_order.extra_items else []

                existing = next((o for o in self.orders if o.ticket_id == ticket.ticket_id), None)
                if existing:
                    # Update active ticket properties but preserve historical detection results
                    existing.expected_items = expected_items
                    existing.remaining_counts = dict(counts)
                    existing.required_counts = dict(counts)
                else:
                    order = Order(
                        ticket_id=ticket.ticket_id,
                        expected_items=expected_items,
                        remaining_counts=dict(counts),
                        required_counts=dict(counts),
                        picked_counts=picked_counts,
                        status=status,
                    )
                    order.passed = passed
                    order.missing_items = missing
                    order.extra_items = extra
                    self.orders.append(order)

    def requires_cheese(self) -> bool:
        """True when the order in progress asks for cheese from any well."""
        required = (
            self.batch_validator.required_counts
            if self.batch_validator
            else self.current_order.required_counts
        )
        return any(key in CHEESE_KEYS for key in required)

    def apply_cheese_takes(self, takes, pre_confirmation: bool = False) -> int:
        """Record cheese slices confirmed out of the cheese region.

        Returns how many were applied.

        ``pre_confirmation`` marks takes that were buffered while no order was
        in progress and are being replayed onto a ticket that has just been
        confirmed.  Those are applied only to a ticket that actually asks for
        cheese: with nothing on the board at the time there is nothing tying
        the slice to this ticket rather than the one before it, and charging it
        to an order that never wanted cheese would fail a correct order on
        evidence that is not about it.

        A live take is applied whatever the ticket says.  That is the
        wrong-cheese signal -- ``BatchOrderValidator`` sees a well that is not
        in ``required_counts``, reports it as extra, and ``calculate_validation``
        turns that into "Wrong Ingredients" -- so it must not be dropped.
        """
        if not takes:
            return 0
        if self.current_ticket is None or self.current_order.status != OrderStatus.IN_PROGRESS:
            return 0
        if pre_confirmation and not self.requires_cheese():
            logger.info(
                "Discarding %d pre-confirmation cheese take(s) -- ticket %s does "
                "not ask for cheese, so the slice belongs to another order",
                len(takes), self.current_order.ticket_id,
            )
            return 0

        required = (
            self.batch_validator.required_counts
            if self.batch_validator
            else self.current_order.required_counts
        )
        wanted_cheese = sorted(k for k in required if k in CHEESE_KEYS)

        applied = 0
        for take in takes:
            # A take from BEFORE this ticket existed is credited to the cheese
            # the ticket asks for, not to the well it was tagged with.
            #
            # The trip is the evidence, and the trip is all the evidence there
            # is: the slice was carried out of the cheese region seconds before
            # the ticket reached the head of the queue, so which well it came
            # from says nothing about whether this order got its cheese -- and
            # a well read as the neighbouring one then failed an order whose
            # cheese was on the dog.  One trip settles the requirement however
            # many slices it was, because a worker picks several at once.
            #
            # A LIVE take keeps its own well, so a swiss slice put on a
            # yellow-cheese ticket while that ticket is being made is still a
            # wrong ingredient.
            credit_to = wanted_cheese if (pre_confirmation and wanted_cheese) else [take.item]

            # Same debounce as the discrete branch of on_action: a region exit
            # is already a completed carry, so the short window only guards
            # against one slice being reported twice.
            last_t = self.last_debounced_ts.get(take.item, 0.0)
            if (take.timestamp - last_t) < DISCRETE_DEBOUNCE_S:
                continue
            self.last_debounced_ts[take.item] = take.timestamp

            for item in credit_to:
                self.current_order.picked_counts[item] = (
                    self.current_order.picked_counts.get(item, 0) + 1
                )
                if self.batch_validator:
                    self.batch_validator.on_place_event(item)

            # The journey exists to explain a WRONG verdict, so the take that
            # caused one has to be in it.  Counting it here but recording
            # nothing leaves the journey showing the opposite of the verdict:
            # while the gate owns cheese, `on_action` drops the bin events --
            # but those are still written as journey steps, so a ticket judged
            # wrong for a sliced-yellow take reads as two grated-yellow places
            # and no sliced at all, which is evidence against its own verdict.
            client = self._kds_client
            if client is not None and hasattr(client, "record_place"):
                for item in credit_to:
                    client.record_place(
                        self.current_order.ticket_id, item,
                        take.timestamp, zone=take.well,
                    )
            applied += 1

            if pre_confirmation and wanted_cheese:
                logger.info(
                    "Cheese replayed onto ticket %s: a trip out of the cheese "
                    "region at t=%.1f (well read as %r) credits %s",
                    self.current_order.ticket_id, take.timestamp, take.well,
                    ", ".join(credit_to),
                )
            elif take.item in wanted_cheese:
                logger.info(
                    "Cheese added: %s from %r -> ticket %s",
                    take.item, take.well, self.current_order.ticket_id,
                )
            else:
                logger.warning(
                    "WRONG CHEESE on ticket %s: %s taken from %r, ticket asks for %s",
                    self.current_order.ticket_id, take.item, take.well,
                    ", ".join(wanted_cheese) if wanted_cheese else "no cheese",
                )

        if applied:
            self.calculate_validation(self.current_order, is_final=False)
        return applied

    def on_kds_ticket(self, ticket: Ticket):
        # Only accept a new ticket when no order is in progress
        if (
            self.current_ticket is not None
            and self.current_order.status == OrderStatus.IN_PROGRESS
        ):
            return

        order = next((o for o in self.orders if o.ticket_id == ticket.ticket_id), None)
        if order is None:
            expected_items, counts, total_hotdogs = _extract_ticket_counts(ticket)
            
            order = Order(
                ticket_id=ticket.ticket_id,
                shortcut=ticket.shortcut,
                expected_items=expected_items,
                remaining_counts=dict(counts),
                required_counts=dict(counts),
                picked_counts={},
                hotdog_count=total_hotdogs,
                status=OrderStatus.IN_PROGRESS,
            )
            self.orders.append(order)

        else:
            expected_items, counts, total_hotdogs = _extract_ticket_counts(ticket)
            order.shortcut = ticket.shortcut
            order.expected_items = expected_items
            order.hotdog_count = total_hotdogs
            order.remaining_counts = dict(counts)
            order.required_counts = dict(counts)
            order.status = OrderStatus.IN_PROGRESS
            order.picked_counts = {}
            order.applied_sauces = []  # Reset sauce state for new cycle
            order.passed = False
            order.missing_items = []
            order.extra_items = []
            order.wrong_items = []
            order.validation_message = ""

        self.current_ticket = ticket
        self.current_order = order
        self._validation_log.clear()
        
        # Initialize batch validator for the new ticket
        self.batch_validator = BatchOrderValidator(ticket, strict_no_extras=False)
        self.calculate_validation(self.current_order, is_final=False)


    def update_ticket_requirements(self, ticket: Ticket) -> bool:
        """Re-point the order in progress at a CHANGED ticket.

        The crew edits orders on the KDS while the food is being made, so the
        requirement is not fixed at creation: kds-ocr re-emits the whole ticket
        whenever its items change, and this applies that new requirement to the
        order already open.

        Progress is deliberately KEPT.  `picked_counts` records what was
        physically observed going on, which an edit to the ticket does not
        undo -- so it carries over, and `remaining_counts` is recomputed as
        "what the new ticket asks for, less what we have already seen".
        Rebuilding the order instead (as `on_kds_ticket` does for a fresh
        ticket) would discard that evidence and judge the order on only the
        part built after the edit.

        Returns True when the update was applied.
        """
        if ticket is None or self.current_ticket is None:
            return False
        if ticket.ticket_id != self.current_ticket.ticket_id:
            # An update for some other ticket: the FIFO source is responsible
            # for only sending updates for the open one.
            return False
        if self.current_order.status != OrderStatus.IN_PROGRESS:
            return False

        expected_items, counts, total_hotdogs = _extract_ticket_counts(ticket)
        order = self.current_order
        order.expected_items = expected_items
        order.required_counts = dict(counts)
        order.hotdog_count = total_hotdogs
        order.remaining_counts = {
            item: max(0, qty - order.picked_counts.get(item, 0))
            for item, qty in counts.items()
        }
        self.current_ticket = ticket

        # Carry the observations across the rebuild, or the new validator
        # starts blind and every already-applied ingredient reads as missing.
        observed = dict(self.batch_validator.observed_counts) if self.batch_validator else {}
        self.batch_validator = BatchOrderValidator(ticket, strict_no_extras=False)
        self.batch_validator.observed_counts.update(observed)
        self.calculate_validation(order, is_final=False)
        return True

    def _normalize(self, name: str) -> str:
        """Normalize an ingredient name to its canonical form.

        Delegates to :func:`src.domain.schemas.canonical_ingredient` so the KDS
        side resolves names identically; see the note there.
        """
        return canonical_ingredient(name)

    def on_action(self, action: Action) -> Optional[str]:
        if self.current_ticket is None:
            return None

        if self.current_order.status != OrderStatus.IN_PROGRESS:
            return None

        matched_key = None

        if action.action_type == "pickup":
            ingredient = action.zone_name
            norm_ing = self._normalize(ingredient)
            for k in self.current_order.remaining_counts.keys():
                if self._normalize(k) == norm_ing:
                    matched_key = k
                    break
            target_key = matched_key or ingredient
            self.current_order.remaining_counts[target_key] = (
                self.current_order.remaining_counts.get(target_key, 0) - 1
            )
            self.current_order.picked_counts[target_key] = (
                self.current_order.picked_counts.get(target_key, 0) + 1
            )

            self._validation_log.append(
                {
                    "ingredient": target_key,
                    "status": "MATCH" if matched_key else "PICKUP",
                    "track_id": action.track_id,
                    "timestamp": action.timestamp,
                }
            )


        elif action.action_type == "pick":
            ingredient = action.zone_name
            norm_ing = self._normalize(ingredient)
            for k in self.current_order.remaining_counts.keys():
                if self._normalize(k) == norm_ing:
                    matched_key = k
                    break
            self._validation_log.append(
                {
                    "ingredient": matched_key or ingredient,
                    "status": "CARRIED",
                    "track_id": action.track_id,
                    "timestamp": action.timestamp,
                }
            )

        elif action.action_type == "hover":
            ingredient = action.zone_name
            norm_ing = self._normalize(ingredient)
            for k in self.current_order.remaining_counts.keys():
                if self._normalize(k) == norm_ing:
                    matched_key = k
                    break
            self._validation_log.append(
                {
                    "ingredient": matched_key or ingredient,
                    "status": "HOVER",
                    "track_id": action.track_id,
                    "timestamp": action.timestamp,
                }
            )

        elif action.action_type in ("place", "sauce"):
            ingredient = action.zone_name
            norm_ing = self._normalize(ingredient)
            now = action.timestamp

            # Cheese belongs to CheesePreGate while it is running: it confirms a
            # slice on cheese-region exit, not well exit, so counting the
            # well-exit event here too would count every slice twice -- and
            # would fire on a hand moving between two adjacent cheese wells,
            # which carries nothing anywhere.
            if self.cheese_gate_owns_cheese and norm_ing in CHEESE_KEYS:
                return matched_key

            # Only count ingredients that are required by the active KDS ticket.
            # Cheese is exempt: the three cheese wells are interchangeable to
            # the eye but not to the ticket, so a slice from the wrong well has
            # to reach the validator to be judged a wrong ingredient.  Dropping
            # it here would make a swiss slice on a C/C look like no cheese at
            # all -- reported as missing rather than as the wrong cheese.
            if (
                self.batch_validator
                and norm_ing not in self.batch_validator.required_counts
                and norm_ing not in CHEESE_KEYS
            ):
                return matched_key

            incremented = False

            if is_granular(norm_ing):
                # ── Granular branch (onions, relish, grated_yellow_cheese) ──────
                # Multiple pinches within the serving-gap are treated as one serving.
                # Onions use a longer gap (12s) to prevent small hovers from
                # each being counted as a separate addition.
                if "onion" in norm_ing:
                    gap = ONION_SERVING_GAP_S
                else:
                    gap = SERVING_GAP_S
                last_pinch = self.last_pinch_ts.get(norm_ing)
                self.last_pinch_ts[norm_ing] = now  # always record the pinch time
                if last_pinch is None or (now - last_pinch) >= gap:
                    # New serving — count it
                    self.current_order.picked_counts[norm_ing] = (
                        self.current_order.picked_counts.get(norm_ing, 0) + 1
                    )
                    incremented = True
                # else: same burst/serving — skip, don't increment
            else:
                # ── Discrete branch (all other ingredients) ───────────────────
                # Suppress rapid duplicate frame events within DISCRETE_DEBOUNCE_S.
                last_t = self.last_debounced_ts.get(norm_ing, 0.0)
                
                # Chilli uses a shorter debounce to allow for fast second scoops
                debounce_time = 0.8 if "chilli" in norm_ing else DISCRETE_DEBOUNCE_S
                
                if (now - last_t) < debounce_time:
                    return matched_key
                self.last_debounced_ts[norm_ing] = now
                self.current_order.picked_counts[norm_ing] = (
                    self.current_order.picked_counts.get(norm_ing, 0) + 1
                )
                incremented = True

            if not incremented:
                return matched_key

            if action.action_type == "sauce":
                if norm_ing not in self.current_order.applied_sauces:
                    self.current_order.applied_sauces.append(norm_ing)

            # Delegate to BatchOrderValidator only when count was incremented
            if self.batch_validator:
                self.batch_validator.on_place_event(norm_ing)

            self._validation_log.append(
                {
                    "ingredient": norm_ing,
                    "status": "PLACED" if action.action_type == "place" else "SAUCE",
                    "track_id": action.track_id,
                    "timestamp": now,
                }
            )

        self.calculate_validation(self.current_order, is_final=False)
        return matched_key



    def abandon_current_order(self) -> Optional[Order]:
        """Drop the order in progress WITHOUT judging it.

        For a ticket that stopped being verifiable after we opened it -- the
        crew voided it, or edited every hot dog off it.  It is not judged,
        because there is no longer a requirement to judge it against; it is
        simply released so the next ticket can start.
        """
        if self.current_ticket is None:
            return None
        if self.current_order.status != OrderStatus.IN_PROGRESS:
            return None
        self._abandon_current_order()
        return self.history[-1]

    def _abandon_current_order(self):
        """Finalize the current order as ABANDONED when a new ticket arrives."""
        self._finalize_order(OrderStatus.ABANDONED)

    def finalize_current_order(self) -> Optional[Order]:
        """Manually finalize the current order as COMPLETED.

        Records missing and extra items based on what was placed so far.
        Returns the finalized Order, or None if no order is in progress.
        """
        if self.current_ticket is None:
            return None
        if self.current_order.status != OrderStatus.IN_PROGRESS:
            return None
        self._finalize_order(OrderStatus.COMPLETED)
        return self.history[-1]

    def _finalize_order(self, status: OrderStatus = OrderStatus.COMPLETED):
        self.current_order.status = status

        # Run final validation
        self.calculate_validation(self.current_order, is_final=(status == OrderStatus.COMPLETED))

        # An ABANDONED order was never judged -- the ticket stopped being
        # verifiable after we opened it (voided, or every hot dog edited off
        # it).  Counting it as failed would inflate the error rate with orders
        # nobody got wrong, and hide the real failures among them.  Accuracy is
        # reported over the orders we actually checked.
        if status != OrderStatus.ABANDONED:
            self.stats.total_orders += 1
            if self.current_order.passed:
                self.stats.passed_orders += 1
            else:
                self.stats.failed_orders += 1

        self.history.append(self.current_order)

        # Notify KDS client of order lifecycle
        if self._kds_client is not None:
            if status == OrderStatus.COMPLETED and hasattr(
                self._kds_client, "mark_completed"
            ):
                self._kds_client.mark_completed(self.current_order.ticket_id)
            elif status == OrderStatus.ABANDONED and hasattr(
                self._kds_client, "mark_abandoned"
            ):
                self._kds_client.mark_abandoned(self.current_order.ticket_id)

        # Save to file if history_path is set
        if self._history_path:
            self._append_order_to_file(self.current_order)

        # Clear current ticket/order so the next KDS poll picks up a new ticket
        self.current_ticket = None
        self.current_order = Order(ticket_id="")
        self._validation_log.clear()

    def calculate_validation(self, order: Order, is_final: bool = False):
        if not order.ticket_id:
            return

        if self.batch_validator:
            if "hot-dog" in order.picked_counts:
                self.batch_validator.observed_counts["hot-dog"] = order.picked_counts["hot-dog"]
            result_dict = self.batch_validator.validate(is_final=is_final)
            
            # Sync back adjusted observed_counts to picked_counts so UI shows the forgiven amounts
            for ing, cnt in self.batch_validator.observed_counts.items():
                if cnt > 0:
                    order.picked_counts[ing] = cnt
                    
            order.passed = result_dict["passed"]
            order.missing_items = list(result_dict["missing"].keys())
            order.extra_items = list(result_dict["extra"].keys())
            # Extras are carried for the dashboard and the journey, but they no
            # longer decide anything: the order is verified once the required
            # items are all present, and something added on top of that does
            # not take the verification away.  Only `missing` fails an order,
            # so the message comes straight from the validator.
            order.wrong_items = list(result_dict["extra"].keys())
            order.validation_message = result_dict["message"]

            order.dashboard_slots = self.batch_validator.distribute_for_dashboard(result_dict["missing"])
            details = []
            for ingredient, count in self.batch_validator.observed_counts.items():
                if count > 0:
                    details.append({
                        "name": ingredient,
                        "quantity": count,
                        "timestamp": None,
                        "status": "correct"
                    })
            order.added_items_details = details
            return

        # Fallback for pickup actions operating on remaining_counts
        missing = [item for item, count in order.remaining_counts.items() if count > 0]
        extra = [item for item, count in order.remaining_counts.items() if count < 0]
        wrong = [item for item in extra if item not in order.expected_items]

        order.missing_items = missing
        order.extra_items = extra
        order.wrong_items = wrong

        passed = (len(missing) == 0 and len(extra) == 0 and len(wrong) == 0)
        order.passed = passed

        if wrong and extra:
            order.validation_message = "Wrong + Extra Ingredients"
        elif wrong:
            order.validation_message = "Wrong Ingredients"
        elif passed:
            order.validation_message = f"Order {order.ticket_id}: correct — all items confirmed."
        else:
            msg_parts = []
            if missing:
                msg_parts.append(f"missing: {', '.join(missing)}")
            if extra:
                msg_parts.append(f"extra: {', '.join(extra)}")
            order.validation_message = f"Order {order.ticket_id}: issue detected — recheck order, " + "; ".join(msg_parts) + "."

        details = []
        detail_map = {}
        for log_entry in self._validation_log:
            ing = log_entry.get("ingredient")
            ts = log_entry.get("timestamp")
            if ing:
                detail_map[ing] = ts
        for ing, ts in detail_map.items():
            st = "wrong" if ing in wrong else ("extra" if ing in extra else "correct")
            details.append({
                "name": ing,
                "quantity": order.picked_counts.get(ing, 1),
                "timestamp": ts,
                "status": st
            })



        if not details:
            for ingredient, count in order.picked_counts.items():
                if count > 0:
                    st = "extra" if ingredient in extra else ("wrong" if ingredient in wrong else "correct")
                    details.append({
                        "name": ingredient,
                        "quantity": count,
                        "timestamp": None,
                        "status": st
                    })
        order.added_items_details = details






    def _append_order_to_file(self, order: Order):
        """Append a completed order to the JSONL history file."""
        record = self._order_record(order)
        p = Path(self._history_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Explicit UTF-8 for the same reason as the journey file: the record
        # carries the validation message, and the default encoding on Windows
        # cannot hold the em dash in it.
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _order_record(self, order: Order) -> dict:
        """Create a persisted history record for an order."""
        if order.passed:
            result = "All ingredients are present"
        elif order.extra_items:
            result = "extra item added"
        else:
            result = "All ingredients are not present"
        import time
        now = time.time()
        time_str = time.strftime("%I:%M %p", time.localtime(now))
        record: dict = {
            "ticket_id": order.ticket_id,
            "status": order.status.value,
            "passed": order.passed,
            "result": result,
            "expected_items": order.expected_items,
            "picked_counts": order.picked_counts,
            "time": time_str,
        }
        if order.missing_items:
            record["missing_items"] = order.missing_items
        if order.extra_items:
            record["extra_items"] = order.extra_items
        if order.applied_sauces:
            record["applied_sauces"] = order.applied_sauces
        return record

    def save_history(self) -> None:
        """Write full order history to a JSON snapshot file (.json)."""
        if not self._history_path:
            return
        p = Path(self._history_path)
        # Snapshot goes to .json, leaving .jsonl as the live append-only stream
        snapshot_path = p.with_suffix(".json")
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        records = [self._order_record(o) for o in self.history]
        snapshot_path.write_text(json.dumps(records, indent=2))

    def get_current_order(self) -> Order:
        return self.current_order

    def get_stats(self) -> Stats:
        return self.stats

    def get_validation_log(self) -> List[dict]:
        return self._validation_log
