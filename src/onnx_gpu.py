"""Moved to :mod:`src.inference.onnx_gpu`.

Kept so existing imports (and anything outside this repo) keep working.
Aliases the module object itself, so ``src.onnx_gpu`` and ``src.inference.onnx_gpu`` are the
same module -- no duplicated state, private names included.
"""
import sys

from src.inference.onnx_gpu import *  # noqa: F401,F403
from src import inference as _pkg

sys.modules[__name__] = getattr(_pkg, "onnx_gpu")
