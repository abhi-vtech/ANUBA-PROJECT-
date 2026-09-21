"""Run exported ONNX graphs on the Jetson GPU through ONNX Runtime.

Ultralytics' ONNX backend (ultralytics/nn/backends/onnx.py) only ever asks ONNX
Runtime for the CUDA execution provider.  On this Orin NX that runs the
segmentation model at 57 ms per inference; the TensorRT execution provider
with an FP16 engine runs it at 16.9 ms.  `tensorrt_first` puts TensorRT at the
front of the provider list for the session Ultralytics creates.  Everything
Ultralytics builds after that, including its IO binding, comes from that
session, so nothing else changes.

    ONNX_PROVIDER=tensorrt   (default) TensorRT FP16; falls back to CUDA, then CPU
    ONNX_PROVIDER=cuda       CUDA
    ONNX_PROVIDER=cpu        CPU

The first TensorRT load builds the FP16 engine (~8 minutes on this Jetson) and
caches it; later loads read the cache in seconds.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path


def _configured_provider() -> str:
    """`onnx_provider` from config/model.yaml, if it is set there."""
    try:
        import yaml

        from src.domain.paths import resource

        with open(resource("config/model.yaml")) as fh:
            return str((yaml.safe_load(fh) or {}).get("onnx_provider") or "")
    except Exception:
        return ""


def onnx_provider() -> str:
    """The provider that will actually be used: "tensorrt", "cuda" or "cpu"."""
    wanted = (os.environ.get("ONNX_PROVIDER") or _configured_provider() or "tensorrt").strip().lower()
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:
        return "cpu"
    has_cuda = "CUDAExecutionProvider" in available
    if wanted == "tensorrt" and has_cuda and "TensorrtExecutionProvider" in available:
        return "tensorrt"
    if wanted in ("tensorrt", "cuda") and has_cuda:
        return "cuda"
    return "cpu"


@contextlib.contextmanager
def tensorrt_first(cache_dir):
    """While active, sessions that request CUDA get TensorRT FP16 in front of it."""
    import onnxruntime as ort

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    tensorrt = (
        "TensorrtExecutionProvider",
        {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache),
        },
    )
    original = ort.InferenceSession

    def session(path_or_bytes, *args, providers=None, **kwargs):
        names = [p if isinstance(p, str) else p[0] for p in (providers or [])]
        if "CUDAExecutionProvider" in names and "TensorrtExecutionProvider" not in names:
            providers = [tensorrt, *providers]
        return original(path_or_bytes, *args, providers=providers, **kwargs)

    ort.InferenceSession = session
    try:
        yield
    finally:
        ort.InferenceSession = original


def session_providers(yolo) -> list:
    """Providers of the ONNX Runtime session inside an Ultralytics YOLO model."""
    stack, seen = [getattr(yolo, "predictor", None)], set()
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        sess = getattr(obj, "session", None)
        if sess is not None and hasattr(sess, "get_providers"):
            return list(sess.get_providers())
        for name in ("model", "backend"):
            child = getattr(obj, name, None)
            if child is not None:
                stack.append(child)
    return []
