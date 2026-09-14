# KDS Ticket Reading, FIFO Grouping & Order Validation

Second video input for the Order Accuracy pipeline: reads the **KDS screen**,
extracts paid tickets, and validates them against what the existing
**production video** pipeline detects.

The production pipeline is reused unchanged — `src/inference/detector.py`,
`src/hotdog_tracker.py`, `src/wrapping_state.py`, `src/zones.py`,
`src/temporal.py`, `src/state_machine.py` are untouched.

```
      KDS VIDEO                              PRODUCTION VIDEO
          |                                         |
  TicketCardDetector                        Detector + ByteTrack
          |                                         |
    TicketParser  --- colours ---> RowColor    HotdogTracker
          |                                         |
 TicketStabilityTracker                     WrappingStateMachine
          |                                         |
          +------------> TicketManager <------------+
                       (FIFO + validation)
                              |
                    dashboard / kds_timeline.jsonl
```

## Running

```bash
python run_kds_dual.py --kds <kds video|rtsp> --production <kitchen video|rtsp>
```

Dashboard at <http://localhost:8000>; the timeline is also appended to
`output/kds_timeline.jsonl`. Equivalent env vars: `KDS_MODE=video`,
`KDS_SOURCE=...`, `VIDEO_SOURCE=...`.

## Configuration

| File | Purpose |
| --- | --- |
| `config/kds_shortcuts.yaml` | shortcut → physical item, add-on vocabulary, ignored products. **The only place menu items are defined.** |
| `config/kds_visual.yaml` | HSV colour bands, card detection, OCR, temporal stability (incl. the disappearance grace period) |

Adding a menu item is a config edit — no code change.

## Verified KDS visual semantics

Read off the live capture (`Wienerschnitzel …__kds__…mkv`) and the reference
screenshots in `videos/kds_images/`:

| Element | Appearance |
| --- | --- |
| Ticket ID | `CHK 259` top-left, elapsed timer top-right |
| Hotdog item | **yellow** bar (`2 Org C/C`, `1 AB Kraut`) |
| Add-on | **grey** *or* **orange** bar under its parent (`Onion`, `2 Ketchup`) |
| Other products | plain (`SML FRY`), **cyan** (`SM COKE`), **blue** (`TENDER 2PK`), **magenta** (`Dlx Ch Brg`) |
| NOT PAID | green-on-black `Subtotal 22.58` |
| PAID | green-on-black `*** Paid *** 22.58` |

### The verdict comes from the ticket disappearing — never from pink

The card body escalates **white → yellow → light pink** as the order ages: the
same three tickets read white/yellow/pink at 0:39/2:04/4:59 and were all pink
by 7:18, purely from waiting.

Pink therefore means **overdue**, not finished. Judging an order there would
announce CORRECT/WRONG for an order still being made, so pink decides nothing.
There is no blink detection in the pipeline at all.

An order is judged **once**, when its ticket **disappears** from the KDS — the
moment it is bumped, which is when the order is genuinely over. Anything still
open at shutdown is finalised too, so no order is left unjudged. Because the
card is gone by then, the CORRECT/WRONG result is drawn as a banner on the KDS
feed for a few seconds (`src/kds/overlay.py`).

A fully yellow or pink card also defeats the yellow-bar rule, so the parser
detects that state and falls back to resolving items by content instead.

## What "detected" can honestly mean

`rf_trained/weights (1).pt` has 16 classes and **one** hotdog class
(`hot-dog`) — there is no per-variant class, and no chili/cheese/onion classes.
Consequently:

* **Quantity** is measured directly and is reliable.
* **Type** is only *inferred*, from the ingredients the zone/`TemporalTracker`
  path attributed to a track (`_infer_hotdog_item` in `src/main.py`, Jaccard
  ≥ 0.5 or it stays the bare `hot-dog`).
* Every verdict carries `type_match_confident=False`. A type mismatch alone
  does not fail an order unless `strict_type_matching=True`.
* **Add-ons are never marked detected** — they are extracted and stored from
  the KDS so the architecture is ready, but no production evidence exists yet.

## Modules

| Module | Role |
| --- | --- |
| `schemas.py` | `TicketSnapshot`, `HotdogGroup`/`AddOn` tree, `OrderGroup`, `LifecycleState`, `ALLOWED_TRANSITIONS` |
| `shortcut_map.py` | config-driven shortcut/add-on resolution; UNKNOWN never force-fitted |
| `ocr_engine.py` | `OcrEngine` adapter — `RapidOcrEngine` (default), `StubOcrEngine` (tests) |
| `colors.py` | HSV bands, competitive row classification, pink fraction |
| `ticket_detector.py` | finds cards inside the dark desktop, splits the grid on gutters, keeps stable slots |
| `ticket_parser.py` | card → `TicketSnapshot`; payment, hotdog bars, add-on parenting |
| `stability.py` | temporal voting: one ticket per card, paid confirmed over N reads; declares disappearance after a grace period |
| `overlay.py` | annotates the KDS feed with ticket state and the verdict banner |
| `fifo_queue.py` | `TicketManager` — FIFO, lifecycle, final validation, failure categories |
| `kds_monitor.py` | per-frame orchestration, routes events into the manager |
| `kds_video_client.py` | `KDSClient` implementation so `OrderStateMachine` is unchanged |
| `timeline.py` | event timeline for the dashboard and disk |

## Business rules → where they live

| Rule | Implementation |
| --- | --- |
| 1 Only PAID tickets enter | `stability.py` `paid_confirm_observations`; `TicketManager.create_from_paid_ticket` rejects unpaid |
| 2 Only yellow bars are hotdogs | `ticket_parser._extract_items` |
| 3 Grey below yellow is that hotdog's add-on | `_extract_items` parent tracking (incl. non-hotdog parents) |
| 4 Only configured shortcuts | `shortcut_map.resolve_shortcut` → `UNKNOWN_SHORTCUT` |
| 5 FIFO by paid time | `TicketManager._enqueue` sorts on `paid_at` |
| 6 No early failure | only `_finalize` can produce a verdict |
| 7 Order end | ticket disappearance (`TicketStabilityTracker.sweep`); pink is ignored |
| 8 Final validation | `_finalize` freezes evidence, then `validate` |
| 9 Remove after disappearance | `on_ticket_disappeared` — the only verdict trigger |
| 10 Report what was wrong | `FailureCategory` + `_explain` |

## Tuning for new hardware

```bash
python scripts/calibrate_kds_colors.py <kds video|screenshot>
```

Reports cards found per frame, measured HSV per role, rows that matched no
band, and the pink-fraction distribution.

## Performance

OCR costs ~0.6 s per card, so a 3-card screen is ~1.8 s. `ocr.frame_stride`
(default 10) controls how often OCR runs; card **presence** is still checked on
every frame, because presence is what decides when an order ends. Confirmation counts are in OCR passes, so
`frame_stride × paid_confirm_observations` is the real latency before a paid
ticket is created (~3 s at 10 fps).

## Tests

```bash
pytest tests/test_kds_vision.py -q     # 69 tests, all ten required scenarios
```

They drive the real parser/stability/FIFO code with a `StubOcrEngine` and
synthetic cards painted in the true KDS colours, so no model download or video
file is needed.
