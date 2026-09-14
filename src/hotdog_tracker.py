"""Moved to :mod:`src.analysis.hotdog_tracker`.

Kept so existing imports (and anything outside this repo) keep working.
Aliases the module object itself, so ``src.hotdog_tracker`` and ``src.analysis.hotdog_tracker`` are the
same module -- no duplicated state, private names included.
"""
import sys

from src.analysis.hotdog_tracker import *  # noqa: F401,F403
from src import analysis as _pkg

sys.modules[__name__] = getattr(_pkg, "hotdog_tracker")
