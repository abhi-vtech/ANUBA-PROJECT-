import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from src.paths import resource
from src.cart_state_machine import CartStateMachine, CartEvent, EventType, CartState

app = FastAPI()
templates = Jinja2Templates(directory=resource("templates"))

# Global State Machines
state_machine = None
_cart_path = Path("config/cart_state.json")
if _cart_path.exists():
    try:
        _cart_path.unlink()
    except Exception:
        pass

cart_machine = CartStateMachine(container_id="Assembly Tray #1", persistence_path="config/cart_state.json", load_from_disk=False)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_latest_frame: Optional[bytes] = None
_latest_ticket: Optional[Any] = None
_latest_order: Optional[Any] = None
_latest_stats: Optional[Any] = None
_latest_detections: dict = {}
_latest_events: list = []
_latest_validation_log: list = []
_latest_track_states: dict = {}
_cached_zones: list = []
_hotdog_log: dict = {}
_wrapping_done_ids: list = []   # track_ids permanently wrapped (DONE state)
_fps: float = 0.0
_last_frame_time: float = 0.0
_frame_lock = asyncio.Lock()


# Default color palette for zones (used when color not specified in config)
DEFAULT_ZONE_COLORS = [
    "#3b82f6",
    "#22c55e",
    "#f59e0b",
    "#ef4444",
    "#8b5cf6",
    "#ec4899",
    "#14b8a6",
    "#f97316",
    "#84cc16",
    "#06b6d4",
]


def load_zones():
    """Load zones from config. Uses color from config if defined, otherwise auto-assigns.
    Applies ingredient aliases from KDS mock config to zone display names."""
    zones_path = Path(resource("config/zones.json"))
    if zones_path.exists():
        zones = json.loads(zones_path.read_text())
        # Use color from config if defined, otherwise auto-assign
        for i, zone in enumerate(zones):
            if "color" not in zone:
                zone["color"] = DEFAULT_ZONE_COLORS[i % len(DEFAULT_ZONE_COLORS)]
        return zones
    return []


def update_frame_data(
    frame: np.ndarray,
    ticket: Any,
    order: Any,
    stats: Any,
    detections: dict = None,
    events: list = None,
    validation_log: list = None,
    track_states: dict = None,
    hotdog_log: dict = None,
    wrapping_done_ids=None,
):
    global \
        _latest_frame, \
        _latest_ticket, \
        _latest_order, \
        _latest_stats, \
        _latest_detections, \
        _latest_events, \
        _latest_validation_log, \
        _fps, \
        _last_frame_time, \
        _hotdog_log, \
        _wrapping_done_ids
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    _latest_frame = buf.tobytes()
    _latest_ticket = ticket
    _latest_order = order
    _latest_stats = stats
    if detections:
        _latest_detections = detections
    if events:
        _latest_events = events
    if validation_log is not None:
        _latest_validation_log = validation_log
    if track_states is not None:
        _latest_track_states = track_states
    if hotdog_log is not None:
        _hotdog_log = hotdog_log
    if wrapping_done_ids is not None:
        _wrapping_done_ids = list(wrapping_done_ids)

    # Calculate FPS
    now = time.time()
    if _last_frame_time > 0:
        _fps = 0.9 * _fps + 0.1 * (1.0 / (now - _last_frame_time))
    _last_frame_time = now


def add_event(
    event_type: str,
    zone: str = None,
    item: str = None,
    duration: float = None,
):
    """Add a live action event."""
    global _latest_events
    now = time.time()
    time_str = time.strftime("%H:%M:%S", time.localtime(now))

    if event_type == "pickup":
        if item and item.lower() == "onions":
            desc = f"she picked <span class='zone'>{item}</span>"
        else:
            desc = f"Picked up <span class='zone'>{item}</span>"
        if duration:
            desc += f"<br>({duration:.1f}s dwell)"
        desc += "<br><span class='highlight'>✓ Added</span>"
    elif event_type == "pick":
        desc = f"Carried <span class='zone'>{item}</span> to Assembly"
        if duration:
            desc += f"<br>({duration:.1f}s)"
    elif event_type == "place":
        desc = f"Placed <span class='zone'>{item}</span> in Assembly Zone"
        if duration:
            desc += f"<br>({duration:.1f}s)"
    elif event_type == "sauce":
        desc = f"Sauce applied: <span class='zone'>{item}</span>"
    else:
        desc = f"{event_type}"
        if item:
            desc += f": <span class='zone'>{item}</span>"
        if duration:
            desc += f"<br>({duration:.1f}s)"

    if event_type in ("pickup", "pick", "place", "sauce") and item:
        cart_machine.process_event(
            CartEvent(
                event_type=EventType.INGREDIENT_ADDED,
                ingredient_name=item,
                roi_id=zone or "ROI_Assembly",
                confidence=0.95,
            )
        )
    elif event_type in ("container_removed", "reset", "hotdog_exited"):
        cart_machine.process_event(
            CartEvent(event_type=EventType.CONTAINER_REMOVED, roi_id=zone or "ROI_Assembly")
        )

    _latest_events.insert(0, {"time": time_str, "description": desc})
    _latest_events = _latest_events[:20]  # Keep last 20 events


def get_cart_data() -> dict:
    """Returns current serializable Cart State Machine status dictionary, synchronized with hotdog tracker."""
    active_hotdog_id = None
    items_list = cart_machine.get_ingredient_names()
    counts = cart_machine.get_ingredient_counts()

    if _hotdog_log:
        active_rec = None
        # Find active/wrapping hotdog in log
        for rec in reversed(list(_hotdog_log.values())):
            if rec.get("active") or rec.get("status") in ("in_progress", "wrapping"):
                active_rec = rec
                break
        if active_rec is None and len(_hotdog_log) > 0:
            active_rec = list(_hotdog_log.values())[-1]

        if active_rec:
            hid = active_rec.get("hotdog_id") or active_rec.get("track_id")
            if hid is not None:
                active_hotdog_id = f"#{hid}"
            rec_items = active_rec.get("item_names", [])
            rec_counts = active_rec.get("item_counts", {})
            if rec_items:
                items_list = rec_items
                counts = rec_counts

    return {
        "state": cart_machine.state.value,
        "container_id": cart_machine.container_id or "Assembly Tray #1",
        "active_hotdog_id": active_hotdog_id,
        "ingredients": items_list,
        "counts": counts,
        "ingredient_details": [
            {
                "name": ing.name,
                "confidence": ing.confidence,
                "roi_id": ing.roi_id,
                "added_at": ing.added_at,
            }
            for ing in cart_machine.ingredients
        ],
        "transition_history": cart_machine.transition_history[-6:],
        "updated_at": cart_machine.updated_at,
    }


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/anuba_logo_icon.jpg")
async def get_anuba_logo():
    return FileResponse(resource("templates/anuba_logo_icon.jpg"))


@app.get("/logo_italic.jpg")
async def get_logo_italic():
    return FileResponse(resource("templates/logo_italic.jpg"))


@app.get("/logo_italic.png")
async def get_logo_italic_png():
    return FileResponse(resource("templates/logo_italic.png"))


@app.get("/zones")
async def get_zones():
    return load_zones()


@app.get("/api/cart")
async def api_cart():
    """Live Cart State Machine status JSON."""
    return get_cart_data()


@app.post("/api/cart/reset")
async def api_cart_reset():
    """Trigger cart reset mechanism (tied to tray removal)."""
    cart_machine.process_event(CartEvent(event_type=EventType.CONTAINER_REMOVED, roi_id="ROI_Assembly"))
    return {"status": "success", "cart": get_cart_data()}


@app.post("/api/cart/event")
async def api_cart_event(req: Request):
    """Post an event payload into the Cart State Machine."""
    data = await req.json()
    evt_type_str = data.get("event_type", EventType.INGREDIENT_ADDED.value)
    evt_type = EventType(evt_type_str) if evt_type_str in [e.value for e in EventType] else EventType.INGREDIENT_ADDED
    
    event = CartEvent(
        event_type=evt_type,
        ingredient_name=data.get("ingredient_name"),
        roi_id=data.get("roi_id", "ROI_Assembly"),
        container_id=data.get("container_id"),
        confidence=float(data.get("confidence", 1.0)),
    )
    cart_machine.process_event(event)
    return {"status": "success", "cart": get_cart_data()}


@app.get("/api/stats")
async def api_stats():
    """Current order statistics as JSON (also used as the container health probe)."""
    return {
        "stats": asdict(_latest_stats) if _latest_stats else None,
        "ticket": asdict(_latest_ticket) if _latest_ticket else None,
        "order": asdict(_latest_order) if _latest_order else None,
        "orders": [asdict(o) for o in state_machine.orders] if state_machine else [],
        "history": [state_machine._order_record(o) for o in state_machine.history] if state_machine else [],
        "fps": _fps,
        "cart": get_cart_data(),
    }


@app.get("/api/hotdog_log")
async def api_hotdog_log():
    """Per-hotdog item log — additive endpoint, not tied to any core system."""
    return {
        "total_hotdogs": len(_hotdog_log),
        "hotdogs": _hotdog_log,
    }


async def mjpeg_generator():
    """Generate MJPEG stream."""
    while True:
        if _latest_frame is not None:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + _latest_frame + b"\r\n"
            )
        await asyncio.sleep(0.033)  # ~30 FPS


@app.get("/video")
async def video_feed():
    """MJPEG video streaming endpoint."""
    return StreamingResponse(
        mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global _cached_zones
    _cached_zones = load_zones()
    try:
        while True:
            await websocket.send_json(
                {
                    "ticket": asdict(_latest_ticket) if _latest_ticket else None,
                    "order": asdict(_latest_order) if _latest_order else None,
                    "stats": asdict(_latest_stats) if _latest_stats else None,
                    "orders": [asdict(o) for o in state_machine.orders] if state_machine else [],
                    "history": [state_machine._order_record(o) for o in state_machine.history] if state_machine else [],
                    "detections": _latest_detections,
                    "events": _latest_events,
                    "validation_log": _latest_validation_log,
                    "track_states": _latest_track_states,
                    "fps": _fps,
                    "zones": _cached_zones,
                    "hotdog_log": _hotdog_log,
                    "wrapping_done_ids": _wrapping_done_ids,
                    "cart": get_cart_data(),
                }
            )
            await asyncio.sleep(0.1)
    except Exception:
        pass

