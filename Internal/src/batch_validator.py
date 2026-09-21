"""Moved to :mod:`src.analysis.batch_validator`.

Kept so existing imports (and anything outside this repo) keep working.
Aliases the module object itself, so ``src.batch_validator`` and ``src.analysis.batch_validator`` are the
same module -- no duplicated state, private names included.
"""
import sys

from src.analysis.batch_validator import *  # noqa: F401,F403
from src import analysis as _pkg

sys.modules[__name__] = getattr(_pkg, "batch_validator")
