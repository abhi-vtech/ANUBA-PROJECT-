"""Canonical ingredient naming, shared by every layer.

Lives in `core` because both the runtime-neutral validator and the legacy
`BatchOrderValidator` need it, and `core` may not import analysis code.  Pure
stdlib, no state: one spelling of an ingredient in, one canonical key out.

OCR gives "Pickles (Rounds)", the mock JSON gives "pickle_rounds", a zone is
named "pickel swears" -- all three have to land on the same key or the order
never validates.
"""
from __future__ import annotations


def normalize_item_name(name: str) -> str:
    if not name:
        return ""
    cleaned = name.lower().replace("_", " ").strip()
    canonical_map = {
        "yellow mustard sauce": "yellow_mustard_sauce",
        "yellow_mustard_sauce": "yellow_mustard_sauce",
        "yellow mustard": "yellow_mustard_sauce",
        "yellow_mustard": "yellow_mustard_sauce",
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
    if cleaned in canonical_map:
        return canonical_map[cleaned]
    if "(" in cleaned:
        cleaned = cleaned.split("(")[0].strip()
    words = cleaned.split()
    cleaned_words = []
    for w in words:
        if w.endswith("s") and w != "swiss":
            w = w[:-1]
        cleaned_words.append(w)
    return "_".join(cleaned_words)
