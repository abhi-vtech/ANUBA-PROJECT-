"""Moved to :mod:`src.video.video_recorder`.

Kept so existing imports (and anything outside this repo) keep working.
Aliases the module object itself, so ``src.video_recorder`` and ``src.video.video_recorder`` are the
same module -- no duplicated state, private names included.
"""
import sys

from src.video.video_recorder import *  # noqa: F401,F403
from src import video as _pkg

sys.modules[__name__] = getattr(_pkg, "video_recorder")
