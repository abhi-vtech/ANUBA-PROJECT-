#!/usr/bin/env python3
"""Build a TensorRT engine from the configured YOLO checkpoint.

    uv run python scripts/export_tensorrt.py            # FP16, imgsz from config
    uv run python scripts/export_tensorrt.py --int8     # INT8 (needs calibration data)

Ultralytics routes this through ONNX internally (exporter.export_engine calls
export_onnx first), so a .onnx is produced as a by-product and then compiled.

THE ENGINE IS NOT PORTABLE.  It is built for this exact GPU, TensorRT version,
precision and input shape.  Rebuild it after changing the model, the imgsz, the
JetPack/TensorRT version, or when moving to another Jetson.  Keep the .pt as the
source of truth and the .onnx as the portable interchange copy.

Requires the system TensorRT python bindings to be importable from the venv:
    ln -sfn /usr/lib/python3.12/dist-packages/tensorrt .venv/lib/python3.12/site-packages/tensorrt
"""
import argparse
import sys
import time
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", help="checkpoint (default: model_path from config/model.yaml)")
    ap.add_argument("--imgsz", type=int, help="build resolution (default: imgsz from config, else 640)")
    ap.add_argument("--workspace", type=int, default=4, help="builder workspace, GiB (default 4)")
    ap.add_argument("--int8", action="store_true", help="INT8 instead of FP16; needs calibration data")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config/model.yaml").read_text())
    weights = args.weights or cfg.get("model_path")
    imgsz = args.imgsz or int(cfg.get("imgsz") or 640)
    src = (root / weights) if not Path(weights).is_absolute() else Path(weights)
    if not src.exists():
        print(f"error: weights not found: {src}", file=sys.stderr)
        return 1

    try:
        import tensorrt as trt
    except ImportError:
        print("error: tensorrt not importable from this venv. Symlink the system bindings:\n"
              "  ln -sfn /usr/lib/python3.12/dist-packages/tensorrt "
              ".venv/lib/python3.12/site-packages/tensorrt", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    print(f"tensorrt  : {trt.__version__}")
    print(f"weights   : {src}")
    print(f"imgsz     : {imgsz}")
    print(f"precision : {'INT8' if args.int8 else 'FP16'}")
    print("building — this takes several minutes; TensorRT profiles kernels on the real device\n")
    t0 = time.time()
    out = YOLO(str(src)).export(
        format="engine", imgsz=imgsz, half=not args.int8, int8=args.int8,
        dynamic=False, simplify=True, workspace=args.workspace,
    )
    print(f"\nbuilt in {time.time()-t0:.0f}s: {out}")
    print(f"run it : MODEL_PATH='{Path(out).relative_to(root)}' ./run.sh videos/<clip>.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
