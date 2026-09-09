"""
Ingredient configuration: granular vs. discrete classification.

Granular ingredients are physically dispensed as several small pinches/sprinkles
per serving. Counting them motion-by-motion overcounts, so they use a longer
serving-gap debounce (SERVING_GAP_S) rather than the short frame-level debounce
used for discrete items.

Discrete ingredients (a squeeze, a spear, a slice, a scoop) are naturally
one-motion-per-serving and keep the existing 1.8s debounce.
"""

GRANULAR_INGREDIENTS: frozenset = frozenset({
    "onions",
    "diced_onions",
    "relish",
    "grated_yellow_cheese",
})

# Everything else (pickle_spears, sport_peppers, hot_dog, yellow_mustard_sauce,
# ketchup_sauce, swiss_cheese, chilli, tomato, ...) is DISCRETE by default.


def is_granular(ingredient: str) -> bool:
    """Return True if the ingredient should use serving-gap debouncing."""
    return ingredient in GRANULAR_INGREDIENTS
