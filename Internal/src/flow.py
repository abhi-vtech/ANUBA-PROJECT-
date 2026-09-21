"""Moved to :mod:`src.analysis.flow`.

Kept so existing imports (and anything outside this repo) keep working.
Aliases the module object itself, so ``src.flow`` and ``src.analysis.flow`` are the
same module -- no duplicated state, private names included.
"""
import sys

from src.analysis.flow import *  # noqa: F401,F403
from src import analysis as _pkg

sys.modules[__name__] = getattr(_pkg, "flow")
