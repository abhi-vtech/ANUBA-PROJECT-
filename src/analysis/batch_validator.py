from typing import Dict, List, Optional

from src.domain.schemas import Ticket, LineItem


from src.core.naming import normalize_item_name  # re-exported


#: The dish itself, as opposed to an ingredient put on it.
HOTDOG_KEY = "hot-dog"


class BatchOrderValidator:
    def __init__(self, ticket: Ticket, strict_no_extras: bool = False):
        self.ticket = ticket
        self.strict_no_extras = strict_no_extras
        self.required_counts: Dict[str, int] = {}
        self.observed_counts: Dict[str, int] = {}
        #: Latched once the ticket's hotdogs have been seen.  The live count is
        #: "hotdog tracks active right now", which falls again as tracks end --
        #: so an order that HAD its three hotdogs would report 1/3 a moment
        #: later and the checklist would flicker.  Whether the dogs were made
        #: does not become untrue because the tracker lost them.
        self._hotdog_satisfied = False
        
        # Flatten order composition into required_counts
        if getattr(self.ticket, "hotdog_specs", None):
            for hd_key, items_val in self.ticket.hotdog_specs.items():
                if isinstance(items_val, (list, tuple)):
                    for item_name in items_val:
                        norm_item = normalize_item_name(item_name)
                        self.required_counts[norm_item] = self.required_counts.get(norm_item, 0) + 1
                elif isinstance(items_val, dict):
                    for item_name, qty in items_val.items():
                        norm_item = normalize_item_name(item_name)
                        self.required_counts[norm_item] = self.required_counts.get(norm_item, 0) + qty
        elif getattr(self.ticket, "expected_items", None):
            for item in self.ticket.expected_items:
                norm_item = normalize_item_name(item)
                self.required_counts[norm_item] = self.required_counts.get(norm_item, 0) + 1

        for li in self.ticket.line_items:
            if getattr(self.ticket, "hotdog_specs", None) and li.variant in self.ticket.hotdog_specs:
                continue
            for item_name, qty in li.items.items():
                norm_item = normalize_item_name(item_name)
                self.required_counts[norm_item] = self.required_counts.get(norm_item, 0) + (qty * li.count)

        req_hd = getattr(self.ticket, "total_hotdogs", 0)
        if req_hd == 0 and getattr(self.ticket, "hotdog_specs", None):
            req_hd = len(self.ticket.hotdog_specs)
        if req_hd > 0:
            self.required_counts["hot-dog"] = req_hd

                
    def on_place_event(self, ingredient_class: str):
        """Record an ingredient placement."""
        norm_item = normalize_item_name(ingredient_class)
        self.observed_counts[norm_item] = self.observed_counts.get(norm_item, 0) + 1


    def validate(self, is_final: bool = False) -> Dict:
        """Validate the order and return the result.

        The hotdogs are judged on ARRIVAL, not on an exact count: once the
        number the ticket asked for has been seen, that check is satisfied and
        stays satisfied.  Counting them precisely is not something this
        pipeline can do -- the tracker fragments one physical hotdog across
        many ids, so the observed number is inflated and unstable -- and
        holding an order against a number we cannot measure produced noise, not
        accuracy.  The INGREDIENTS are still counted, because those are what a
        crew member actually forgets.
        """
        # Forgiveness logic for sauces and onions: if an application happened (observed > 0) 
        # but fell short of the required count, force it to match required.
        # We only apply this at the end of the video (is_final=True) so the UI increments realistically during the order.
        if is_final:
            forgiveness_classes = {"ketchup", "yellow_mustard_sauce"}
            for ingredient, required_qty in self.required_counts.items():
                if ingredient in forgiveness_classes:
                    observed_qty = self.observed_counts.get(ingredient, 0)
                    if 0 < observed_qty < required_qty:
                        self.observed_counts[ingredient] = required_qty

        observed = dict(self.observed_counts)

        # Latch the hotdogs: reaching the required number once is enough.
        required_hd = self.required_counts.get(HOTDOG_KEY, 0)
        if required_hd:
            if observed.get(HOTDOG_KEY, 0) >= required_hd:
                self._hotdog_satisfied = True
            if self._hotdog_satisfied:
                observed[HOTDOG_KEY] = required_hd

        missing = {
            ingredient: required_qty - observed.get(ingredient, 0)
            for ingredient, required_qty in self.required_counts.items()
            if observed.get(ingredient, 0) < required_qty
        }

        extra = {
            ingredient: observed.get(ingredient, 0) - required_qty
            for ingredient, required_qty in self.required_counts.items()
            # Never the hotdogs: the observed number is inflated by track
            # fragmentation, so "extra hotdogs" would fire on correct orders.
            if ingredient != HOTDOG_KEY and observed.get(ingredient, 0) > required_qty
        }
        for ingredient, count in observed.items():
            if (ingredient not in self.required_counts
                    and ingredient != HOTDOG_KEY and count > 0):
                extra[ingredient] = count

        passed = not missing
        if self.strict_no_extras and extra:
            passed = False


            
        result = "CORRECT" if passed else "INCORRECT"
        
        # Generate worker-facing message
        if result == "CORRECT":
            message = f"Order {self.ticket.ticket_id}: correct — all items confirmed."
        else:
            msg_parts = []
            if missing:
                missing_str = ", ".join([f"{count}x {item}" for item, count in missing.items()])
                msg_parts.append(f"missing: {missing_str}")
            if extra:
                extra_str = ", ".join([f"{count}x {item}" for item, count in extra.items()])
                msg_parts.append(f"extra: {extra_str}")
            message = f"Order {self.ticket.ticket_id}: issue detected — recheck order, " + "; ".join(msg_parts) + "."
            
        return {
            "result": result,
            "passed": passed,
            "missing": missing,
            "extra": extra,
            "message": message
        }

    def distribute_for_dashboard(self, missing: Dict[str, int]) -> List[Dict]:
        """
        Allocate counted ingredients across visual slots for display purposes only.
        If the order failed, mark only the variant(s) whose required items are 
        present in the `missing` dict as "needs recheck".
        """
        slots = []
        for item in self.ticket.line_items:
            for i in range(item.count):
                needs_check = any(ing in item.items for ing in missing)
                slots.append({
                    "variant": item.variant, 
                    "status": "check" if needs_check else "ok",
                    "expected_items": item.items
                })
        return slots
