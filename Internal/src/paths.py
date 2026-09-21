"""Moved to :mod:`src.domain.paths`.

Kept so existing imports (and anything outside this repo) keep working.
Aliases the module object itself, so ``src.paths`` and ``src.domain.paths`` are the
same module -- no duplicated state, private names included.
"""
import sys

from src.domain.paths import *  # noqa: F401,F403
from src import domain as _pkg

sys.modules[__name__] = getattr(_pkg, "paths")
