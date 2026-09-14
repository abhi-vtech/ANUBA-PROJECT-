"""Moved to :mod:`src.domain.ingredient_config`.

Kept so existing imports (and anything outside this repo) keep working.
Aliases the module object itself, so ``src.ingredient_config`` and ``src.domain.ingredient_config`` are the
same module -- no duplicated state, private names included.
"""
import sys

from src.domain.ingredient_config import *  # noqa: F401,F403
from src import domain as _pkg

sys.modules[__name__] = getattr(_pkg, "ingredient_config")
