import json
from collections import Counter
from pathlib import Path
from typing import List, Optional

from src.schemas import Action, Order, OrderStatus, Stats, Ticket


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
                counts = Counter(ticket.expected_items)
                
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
                    existing.expected_items = list(ticket.expected_items)
                    existing.remaining_counts = dict(counts)
                else:
                    order = Order(
                        ticket_id=ticket.ticket_id,
                        expected_items=list(ticket.expected_items),
                        remaining_counts=dict(counts),
                        picked_counts=picked_counts,
                        status=status,
                    )
                    order.passed = passed
                    order.missing_items = missing
                    order.extra_items = extra
                    self.orders.append(order)

    def on_kds_ticket(self, ticket: Ticket):
        # Only accept a new ticket when no order is in progress
        if (
            self.current_ticket is not None
            and self.current_order.status == OrderStatus.IN_PROGRESS
        ):
            return

        order = next((o for o in self.orders if o.ticket_id == ticket.ticket_id), None)
        if order is None:
            counts = Counter(ticket.expected_items)
            order = Order(
                ticket_id=ticket.ticket_id,
                expected_items=list(ticket.expected_items),
                remaining_counts=dict(counts),
                picked_counts={},
                status=OrderStatus.IN_PROGRESS,
            )
            self.orders.append(order)
        else:
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
        self.calculate_validation(self.current_order, is_final=False)


    def _normalize(self, name: str) -> str:
        """Helper to normalize names using canonical mappings (handles casing, underscores, spaces, aliases)."""
        if not name:
            return ""

        # Initial cleaning: lower case, strip, replace multiple spaces/underscores
        cleaned = name.lower().replace("_", " ").strip()

        # Direct canonical names lookup
        canonical_map = {
            "yellow mustard sauce": "yellow_mustard",
            "yellow_mustard_sauce": "yellow_mustard",
            "yellow mustard": "yellow_mustard",

            "pickles (rounds)": "pickle_rounds",
            "pickle rounds": "pickle_rounds",
            "pickles rounds": "pickle_rounds",
            "pickle round": "pickle_rounds",

            "pickles (spears)": "pickle_spears",
            "pickle spears": "pickle_spears",
            "pickles spears": "pickle_spears",
            "pickle spear": "pickle_spears",
            "pickel swears": "pickle_spears",
            "pickle swears": "pickle_spears",

            "diced onions": "diced_onions",
            "diced onion": "diced_onions",
            "onions": "onions",
            "onion": "onions",

            "yellow cheese": "yellow_cheese",
            "yellow cheese (sliced)": "yellow_cheese",
            "yellow cheese sliced": "yellow_cheese",

            "grated yellow cheese": "grated_yellow_cheese",
            "chilli grated yellow cheese": "grated_yellow_cheese",

            "tomato": "tomato",
            "tomatoes": "tomato",

            "swiss cheese": "swiss_cheese",
            "relish": "relish",
            "ketchup sauce": "ketchup",
            "ketchup": "ketchup",

            "sport (wax) peppers": "sport_peppers",
            "sport peppers": "sport_peppers",
            "sport wax peppers": "sport_peppers",
            "wax peppers": "sport_peppers",
        }

        # If it matches a key in the map, return the mapped value
        if cleaned in canonical_map:
            return canonical_map[cleaned]

        # Fallback to standard word cleaning
        if "(" in cleaned:
            cleaned = cleaned.split("(")[0].strip()
        words = cleaned.split()
        cleaned_words = []
        for w in words:
            if w.endswith("s") and w != "swiss":
                w = w[:-1]
            cleaned_words.append(w)
        return "_".join(cleaned_words)

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
            self._validation_log.append(
                {
                    "ingredient": matched_key or ingredient,
                    "status": "PICKUP",
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

        elif action.action_type == "place":
            ingredient = action.zone_name
            remaining = self.current_order.remaining_counts

            # Find matching expected ingredient key using normalization
            norm_ing = self._normalize(ingredient)
            for k in remaining.keys():
                if self._normalize(k) == norm_ing:
                    matched_key = k
                    break

            if matched_key is not None:
                # Use the matched KDS key so counts align correctly
                remaining[matched_key] -= 1
                self.current_order.picked_counts[matched_key] = (
                    self.current_order.picked_counts.get(matched_key, 0) + 1
                )
                self._validation_log.append(
                    {
                        "ingredient": matched_key,
                        "status": "MATCH" if remaining[matched_key] >= 0 else "EXTRA",
                        "track_id": action.track_id,
                        "timestamp": action.timestamp,
                    }
                )
            else:
                if ingredient not in remaining:
                    remaining[ingredient] = -1
                else:
                    remaining[ingredient] -= 1
                self.current_order.picked_counts[ingredient] = (
                    self.current_order.picked_counts.get(ingredient, 0) + 1
                )
                self._validation_log.append(
                    {
                        "ingredient": ingredient,
                        "status": "EXTRA",
                        "track_id": action.track_id,
                        "timestamp": action.timestamp,
                    }
                )

        elif action.action_type == "sauce":
            sauce = action.zone_name

            # Check if normalized sauce is already applied
            norm_sauce = self._normalize(sauce)
            already_applied = False
            for s in self.current_order.applied_sauces:
                if self._normalize(s) == norm_sauce:
                    already_applied = True
                    break

            remaining = self.current_order.remaining_counts

            # Find matching expected ingredient key
            for k in remaining.keys():
                if self._normalize(k) == norm_sauce:
                    matched_key = k
                    break

            if matched_key is not None:
                # Match found: map to the expected KDS key name
                if matched_key not in self.current_order.applied_sauces:
                    self.current_order.applied_sauces.append(matched_key)
                self.current_order.picked_counts[matched_key] = (
                    self.current_order.picked_counts.get(matched_key, 0) + 1
                )
                remaining[matched_key] -= 1
                self._validation_log.append(
                    {
                        "ingredient": matched_key,
                        "status": "MATCH" if remaining[matched_key] >= 0 else "EXTRA",
                        "track_id": action.track_id,
                        "timestamp": action.timestamp,
                    }
                )
            else:
                # No match found: register as extra sauce
                if sauce not in self.current_order.applied_sauces:
                    self.current_order.applied_sauces.append(sauce)
                self.current_order.picked_counts[sauce] = (
                    self.current_order.picked_counts.get(sauce, 0) + 1
                )
                if sauce not in remaining:
                    remaining[sauce] = -1
                else:
                    remaining[sauce] -= 1

                self._validation_log.append(
                    {
                        "ingredient": sauce,
                        "status": "EXTRA",
                        "track_id": action.track_id,
                        "timestamp": action.timestamp,
                    }
                )

        self.calculate_validation(self.current_order, is_final=False)
        return matched_key



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

        expected_counts = Counter(order.expected_items)
        picked_counts = order.picked_counts

        details = []
        missing_items = []
        extra_items = []
        wrong_items = []

        # Process expected ingredients
        for ingredient, expected_qty in expected_counts.items():
            picked_qty = picked_counts.get(ingredient, 0)

            # Find the timestamp of when this ingredient was last added
            timestamps = [x["timestamp"] for x in self._validation_log if x.get("ingredient") == ingredient and "timestamp" in x]
            timestamp = timestamps[-1] if timestamps else None

            if picked_qty > 0:
                if picked_qty <= expected_qty:
                    details.append({
                        "name": ingredient,
                        "quantity": picked_qty,
                        "timestamp": timestamp,
                        "status": "correct"
                    })
                else:
                    details.append({
                        "name": ingredient,
                        "quantity": picked_qty,
                        "timestamp": timestamp,
                        "status": "extra"
                    })
                    extra_items.append(ingredient)
            else:
                if is_final:
                    details.append({
                        "name": ingredient,
                        "quantity": 0,
                        "timestamp": None,
                        "status": "missing"
                    })
                    missing_items.append(ingredient)

        # Process wrong ingredients (picked but not expected at all)
        for ingredient, picked_qty in picked_counts.items():
            if ingredient not in expected_counts:
                # Find the timestamp
                timestamps = [x["timestamp"] for x in self._validation_log if x.get("ingredient") == ingredient and "timestamp" in x]
                timestamp = timestamps[-1] if timestamps else None

                details.append({
                    "name": ingredient,
                    "quantity": picked_qty,
                    "timestamp": timestamp,
                    "status": "wrong"
                })
                wrong_items.append(ingredient)

        order.added_items_details = details
        order.missing_items = missing_items
        order.extra_items = extra_items
        order.wrong_items = wrong_items

        if order.status == OrderStatus.ABANDONED:
            order.passed = False
            order.validation_message = "Abandoned"
        else:
            order.passed = (len(missing_items) == 0 and len(extra_items) == 0 and len(wrong_items) == 0)

            parts = []
            if len(missing_items) > 0:
                parts.append("Missing")
            if len(wrong_items) > 0:
                parts.append("Wrong")
            if len(extra_items) > 0:
                parts.append("Extra")

            if len(parts) == 0:
                order.validation_message = "Success"
            else:
                order.validation_message = " + ".join(parts) + (" Ingredients" if len(parts) > 1 else " Ingredient")


    def _append_order_to_file(self, order: Order):
        """Append a completed order to the JSONL history file."""
        record = self._order_record(order)
        p = Path(self._history_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
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
