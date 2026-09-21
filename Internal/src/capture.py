"""Moved to :mod:`src.video.capture`.

Kept so existing imports (and anything outside this repo) keep working.
Aliases the module object itself, so ``src.capture`` and ``src.video.capture`` are the
same module -- no duplicated state, private names included.
"""
import sys

from src.video.capture import *  # noqa: F401,F403
from src import video as _pkg

sys.modules[__name__] = getattr(_pkg, "capture")
