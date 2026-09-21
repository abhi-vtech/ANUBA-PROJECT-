"""Moved to :mod:`src.analysis.exit_detector`.

Kept so existing imports (and anything outside this repo) keep working.
Aliases the module object itself, so ``src.exit_detector`` and ``src.analysis.exit_detector`` are the
same module -- no duplicated state, private names included.
"""
import sys

from src.analysis.exit_detector import *  # noqa: F401,F403
from src import analysis as _pkg

sys.modules[__name__] = getattr(_pkg, "exit_detector")
