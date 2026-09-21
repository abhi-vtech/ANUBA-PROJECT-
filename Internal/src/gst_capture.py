"""Moved to :mod:`src.video.gst_capture`.

Kept so existing imports (and anything outside this repo) keep working.
Aliases the module object itself, so ``src.gst_capture`` and ``src.video.gst_capture`` are the
same module -- no duplicated state, private names included.
"""
import sys

from src.video.gst_capture import *  # noqa: F401,F403
from src import video as _pkg

sys.modules[__name__] = getattr(_pkg, "gst_capture")
