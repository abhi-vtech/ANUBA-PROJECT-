"""Moved to :mod:`src.analysis.state_machine`.

Kept so existing imports (and anything outside this repo) keep working.
Aliases the module object itself, so ``src.state_machine`` and ``src.analysis.state_machine`` are the
same module -- no duplicated state, private names included.
"""
import sys

from src.analysis.state_machine import *  # noqa: F401,F403
from src import analysis as _pkg

sys.modules[__name__] = getattr(_pkg, "state_machine")
