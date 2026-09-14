"""Moved to :mod:`src.analysis.feed_analysis`.

Kept so existing imports (and anything outside this repo) keep working.
Aliases the module object itself, so ``src.feed_analysis`` and ``src.analysis.feed_analysis`` are the
same module -- no duplicated state, private names included.
"""
import sys

from src.analysis.feed_analysis import *  # noqa: F401,F403
from src import analysis as _pkg

sys.modules[__name__] = getattr(_pkg, "feed_analysis")
