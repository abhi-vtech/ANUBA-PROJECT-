"""Moved to :mod:`src.inference.detector`.

Kept so existing imports (and anything outside this repo) keep working.
Aliases the module object itself, so ``src.detector`` and ``src.inference.detector`` are the
same module -- no duplicated state, private names included.
"""
import sys

from src.inference.detector import *  # noqa: F401,F403
from src import inference as _pkg

sys.modules[__name__] = getattr(_pkg, "detector")
