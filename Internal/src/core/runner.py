"""Build a source from a pipeline YAML and run the engine over it.

    python -m src.core.runner pipelines/ultralytics_pt.yaml
    python -m src.core.runner pipelines/deepstream.yaml
    python -m src.core.runner pipelines/replay.yaml --record out/frames.jsonl

The only thing that differs between those three is the YAML.  Nothing in
src/core/engine.py, order_rules.py or ticket_spec.py knows which one ran.

A pipeline file looks like:

    source:
      kind: ultralytics          # ultralytics | deepstream | replay
      video: videos/kds_3.mp4
      model: "rf_trained/weights (1).engine"
      conf: 0.5
    zones: config/zones.json
    ticket:
      json: config/kds_mock.json
      index: 0
    engine:
      dwell_s: 0.5
    extras:
      advisory: true
    output: output/run.json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional

from src.core.contract import Frame
from src.core.engine import AnalysisEngine, EngineConfig, load_zones
from src.core.naming import normalize_item_name
from src.core.order_rules import ExtrasPolicy
from src.core.ticket_spec import TicketSpec


def load_pipeline(path: str) -> Dict[str, Any]:
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def build_source(spec: Dict[str, Any]):
    """Instantiate the source named by `kind`. The one place runtimes appear."""
    kind = str(spec.get("kind", "replay")).lower()

    if kind == "replay":
        from src.core.sources.replay_source import ReplaySource

        return ReplaySource(spec["path"], limit=spec.get("limit"))

    if kind == "ultralytics":
        from src.core.sources.ultralytics_source import UltralyticsSource

        return UltralyticsSource(
            video=spec["video"],
            model_path=spec["model"],
            conf=float(spec.get("conf", 0.5)),
            fps=spec.get("fps"),
            detector_kwargs=spec.get("detector") or {},
            max_frames=spec.get("max_frames"),
        )

    if kind == "deepstream":
        from src.core.sources.deepstream_source import DeepStreamSource

        return DeepStreamSource(
            labels=_read_labels(spec["labels"]),
            build_pipeline=_deepstream_builder(),
            config=spec,
        )

    raise ValueError("unknown source kind: {0!r}".format(kind))


def _read_labels(path: str):
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def _deepstream_builder():
    """Return a callable that builds the pyservicemaker pipeline from config.

    Kept here rather than in the source adapter so the element graph is data
    driven: every element property below comes from the pipeline YAML.
    """

    def build(config: Dict[str, Any], operator):
        from pyservicemaker import Pipeline, Probe

        width = int(config.get("width", 1280))
        height = int(config.get("height", 720))
        pipeline = (
            Pipeline(config.get("name", "oa-core"))
            .add("filesrc", "src", {"location": config["video"]})
            .add("h264parse", "parser")
            .add("nvv4l2decoder", "decoder")
            .add("nvstreammux", "mux", {"batch-size": 1, "width": width,
                                        "height": height, "batched-push-timeout": 40000})
            .add("nvinfer", "infer", {"config-file-path": config["infer_config"]})
            .add("nvtracker", "tracker", {
                "tracker-width": int(config.get("tracker_width", 640)),
                "tracker-height": int(config.get("tracker_height", 384)),
                "ll-lib-file": config["tracker_lib"],
                "ll-config-file": config["tracker_config"],
            })
            .add("fakesink", "sink", {"sync": 0})
        )
        pipeline.link("src", "parser", "decoder")
        pipeline.link(("decoder", "mux"), ("", "sink_%u"))
        pipeline.link("mux", "infer", "tracker", "sink")
        # After the tracker, never after infer -- object_id does not exist yet.
        pipeline.attach("tracker", Probe("core", operator))
        return pipeline

    return build


def build_ticket(spec: Optional[Dict[str, Any]], known_items=None) -> Optional[TicketSpec]:
    if not spec:
        return None
    if "json" in spec:
        with open(spec["json"]) as fh:
            data = json.load(fh)
        tickets = data.get("tickets", data) if isinstance(data, dict) else data
        if isinstance(tickets, dict):
            tickets = [tickets]
        return TicketSpec.from_json(tickets[int(spec.get("index", 0))], known_items=known_items)
    if "inline" in spec:
        return TicketSpec.from_json(spec["inline"], known_items=known_items)
    return None


def run(pipeline_path: str, record: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_pipeline(pipeline_path)
    zones = load_zones(cfg["zones"])
    engine_cfg = EngineConfig(**(cfg.get("engine") or {}))
    policy = ExtrasPolicy(**{
        k: (frozenset(v) if isinstance(v, list) else v)
        for k, v in (cfg.get("extras") or {}).items()
    })

    well_items = {
        normalize_item_name(z.name)
        for z in zones if z.zone_type in engine_cfg.well_types
    }
    ticket = build_ticket(cfg.get("ticket"), known_items=well_items)

    engine = AnalysisEngine(zones, spec=ticket, config=engine_cfg, policy=policy)
    source = build_source(cfg["source"])

    recorder = open(record, "w") if record else None
    try:
        for frame in source:
            if recorder is not None:
                recorder.write(json.dumps(_frame_to_json(frame)) + "\n")
            engine.process(frame)
    finally:
        if recorder is not None:
            recorder.close()
        source.close()

    result = engine.summary()
    result["pipeline"] = pipeline_path
    result["source_kind"] = cfg["source"].get("kind")
    if cfg.get("output"):
        with open(cfg["output"], "w") as fh:
            json.dump(result, fh, indent=2)
    return result


def _frame_to_json(frame: Frame) -> Dict[str, Any]:
    return {
        "index": frame.index,
        "t": round(frame.t, 4),
        "width": frame.width,
        "height": frame.height,
        "objects": [
            {"track_id": o.track_id, "label": o.label,
             "bbox": [round(v, 2) for v in o.bbox],
             "confidence": round(o.confidence, 4)}
            for o in frame.objects
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the analysis core over any pipeline.")
    ap.add_argument("pipeline")
    ap.add_argument("--record", help="also write every frame to this JSONL for replay")
    args = ap.parse_args(argv)

    result = run(args.pipeline, record=args.record)
    verdict = result.get("verdict")
    print("frames: {0}  events: {1}".format(result["frames"], len(result["events"])))
    if verdict:
        print(verdict["message"])
        print(json.dumps(verdict, indent=2))
    return 0 if (verdict is None or verdict["correct"]) else 1


if __name__ == "__main__":
    sys.exit(main())
