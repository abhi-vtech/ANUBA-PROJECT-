"""Moved to :mod:`src.analysis.tracker`.

Kept so existing imports (and anything outside this repo) keep working.
Aliases the module object itself, so ``src.tracker`` and ``src.analysis.tracker`` are the
same module -- no duplicated state, private names included.
"""
import sys

from src.analysis.tracker import *  # noqa: F401,F403
from src import analysis as _pkg

sys.modules[__name__] = getattr(_pkg, "tracker")
