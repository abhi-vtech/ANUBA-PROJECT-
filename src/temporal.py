"""Moved to :mod:`src.analysis.temporal`.

Kept so existing imports (and anything outside this repo) keep working.
Aliases the module object itself, so ``src.temporal`` and ``src.analysis.temporal`` are the
same module -- no duplicated state, private names included.
"""
import sys

from src.analysis.temporal import *  # noqa: F401,F403
from src import analysis as _pkg

sys.modules[__name__] = getattr(_pkg, "temporal")
