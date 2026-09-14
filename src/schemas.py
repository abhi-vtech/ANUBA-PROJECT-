"""Moved to :mod:`src.domain.schemas`.

Kept so existing imports (and anything outside this repo) keep working.
Aliases the module object itself, so ``src.schemas`` and ``src.domain.schemas`` are the
same module -- no duplicated state, private names included.
"""
import sys

from src.domain.schemas import *  # noqa: F401,F403
from src import domain as _pkg

sys.modules[__name__] = getattr(_pkg, "schemas")
