"""Moved to :mod:`src.analysis.wrapping_state`.

Kept so existing imports (and anything outside this repo) keep working.
Aliases the module object itself, so ``src.wrapping_state`` and ``src.analysis.wrapping_state`` are the
same module -- no duplicated state, private names included.
"""
import sys

from src.analysis.wrapping_state import *  # noqa: F401,F403
from src import analysis as _pkg

sys.modules[__name__] = getattr(_pkg, "wrapping_state")
