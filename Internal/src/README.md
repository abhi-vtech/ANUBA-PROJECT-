# `src/` layout

Layered, with dependencies pointing **downward only**:

```
  main.py            app entry point
  ui/                dashboard (FastAPI + templates)
  analysis/          what the frames mean - state machines, feed analysis, validation
  core/              runtime-neutral pipeline core (see core/README.md)
  video/             frame sources and sinks - capture, gst_capture, video_recorder
  inference/         model execution - detector (.pt/.onnx/.engine), onnx_gpu
  kds/               KDS: OCR, ticket parsing, client
  domain/            types + static config - schemas, zones, ingredient_config, paths
  system_monitor.py  standalone tegrastats telemetry
```

`domain/` imports nothing from the rest. `core/` imports only stdlib and
`core.naming` — that constraint is what lets the same analysis run behind
Ultralytics, DeepStream and a JSONL replay, so keep it.

## Legacy import paths still work

Every module that moved left a shim at its old path:

```python
from src.detector import Detector        # still works
from src.inference.detector import Detector   # canonical
```

The shim aliases the module object (`sys.modules[__name__] = ...`), so
`src.detector is src.inference.detector` is `True` — one module, no duplicated
state, private names included. Prefer the canonical path in new code; the
shims exist so nothing outside this repo breaks.

## Where things went

| was | now |
|---|---|
| `src/schemas.py` `zones.py` `ingredient_config.py` `paths.py` | `src/domain/` |
| `src/detector.py` `onnx_gpu.py` | `src/inference/` |
| `src/capture.py` `gst_capture.py` `video_recorder.py` | `src/video/` |
| `src/state_machine.py` `wrapping_state.py` `temporal.py` `flow.py` `feed_analysis.py` `hotdog_tracker.py` `cart_state_machine.py` `exit_detector.py` `tracker.py` `batch_validator.py` | `src/analysis/` |
| `src/dashboard.py` | `src/ui/` |
| `src/kds_client.py` | `src/kds/client.py` |

`normalize_item_name` moved from `batch_validator` to `src/core/naming.py`, so
`core` no longer imports analysis code. `batch_validator` re-exports it, so
`from src.batch_validator import normalize_item_name` is unchanged.

## Not touched

`src/main.py` (1,944 lines) and `src/analysis/hotdog_tracker.py` (2,046) were
moved but not split. They are the live production path and have no test
coverage; splitting them is a separate job that needs tests first.
