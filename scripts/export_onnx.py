#!/usr/bin/env python3
"""Export the configured YOLO checkpoint to ONNX.

    uv run python scripts/export_onnx.py                    # uses config/model.yaml
    uv run python scripts/export_onnx.py --weights x.pt --imgsz 640
    uv run python scripts/export_onnx.py --dynamic          # variable input size

The graph is written next to the source checkpoint with a .onnx suffix.  Point
the pipeline at it with MODEL_PATH, or by editing model_path in
config/model.yaml -- src/detector.py picks the backend up from the extension
and needs no other change.

NOTE ON SPEED: on JetPack 7 / cp312 the installable onnxruntime wheel exposes
CPUExecutionProvider only (no CUDA or TensorRT provider), so an ONNX graph runs
on the CPU and is roughly 13x slower than the .pt on CUDA.  Export ONNX for
portability or as the intermediate step toward a TensorRT .engine -- not to make
this Jetson faster.
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", help="checkpoint to export (default: model_path from config/model.yaml)")
    ap.add_argument("--imgsz", type=int, help="export resolution (default: imgsz from config, else 640)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamic", action="store_true",
                    help="variable input size; slower but accepts any imgsz at runtime")
    ap.add_argument("--half", action="store_true",
                    help="fp16 graph; only useful for a GPU provider, not the CPU one")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config/model.yaml").read_text())
    weights = args.weights or cfg.get("model_path")
    imgsz = args.imgsz or int(cfg.get("imgsz") or 640)

    src = (root / weights) if not Path(weights).is_absolute() else Path(weights)
    if not src.exists():
        print(f"error: weights not found: {src}", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    print(f"weights : {src}")
    print(f"imgsz   : {imgsz}{' (dynamic)' if args.dynamic else ' (fixed)'}")
    out = YOLO(str(src)).export(
        format="onnx", imgsz=imgsz, opset=args.opset,
        simplify=True, dynamic=args.dynamic, half=args.half,
    )
    print(f"\nexported: {out}")
    print(f"run it  : MODEL_PATH='{Path(out).relative_to(root)}' ./run.sh videos/<clip>.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
