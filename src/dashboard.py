"""Moved to :mod:`src.ui.dashboard`.

Kept so existing imports (and anything outside this repo) keep working.
Aliases the module object itself, so ``src.dashboard`` and ``src.ui.dashboard`` are the
same module -- no duplicated state, private names included.
"""
import sys

from src.ui.dashboard import *  # noqa: F401,F403
from src import ui as _pkg

sys.modules[__name__] = getattr(_pkg, "dashboard")
