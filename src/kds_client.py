"""Moved to :mod:`src.kds.client`.

Kept so existing imports (and anything outside this repo) keep working.
Aliases the module object itself, so ``src.kds_client`` and ``src.kds.client`` are the
same module -- no duplicated state, private names included.
"""
import sys

from src.kds.client import *  # noqa: F401,F403
from src import kds as _pkg

sys.modules[__name__] = getattr(_pkg, "client")
