# `src/core` — one analysis core, any runtime

The runtime is chosen by a YAML file. No Python changes when it changes.

```bash
python -m src.core.runner pipelines/ultralytics_pt.yaml    # .pt      ~12 fps
python -m src.core.runner pipelines/ultralytics_trt.yaml   # .engine  ~16 fps
python -m src.core.runner pipelines/deepstream.yaml        # nvinfer  ~45 fps
python -m src.core.runner pipelines/replay.yaml            # no GPU, instant
```

`ultralytics_pt.yaml` and `ultralytics_trt.yaml` differ by one line: `model:`.

## Layering

```
 pipelines/*.yaml      which runtime, which video, which ticket, which policy
        |
 src/core/runner.py    wiring only
        |
 src/core/sources/     ultralytics | deepstream | replay   <- runtime lives HERE, and only here
        |
 src/core/contract.py  Frame + TrackedObject               <- the seam
        |
 src/core/engine.py    wells, dwell, debounce, hotdog identity
        |
 src/core/order_rules.py  the three checks
```

**The rule:** nothing from `contract.py` downward may import `torch`, `cv2`,
`ultralytics` or `pyservicemaker`. If it needs to, the layering is broken.
`engine.py` and `order_rules.py` currently import only stdlib — that is what
makes the same object runnable behind all three runtimes, and testable with
neither a GPU nor a video.

Sources map class **ids** to class **names** at the boundary, so no rule ever
depends on "hot-dog is index 6". That coupling — the thing that makes
`[class-attrs-6]` in `config_infer.txt` fragile — stops at the contract.

All thresholds are in **media-time seconds**, never frame counts. A 12 fps run
and a 45 fps run over the same footage therefore produce identical verdicts;
`test_frame_rate_does_not_change_the_verdict` pins this.

## Adding a runtime

Write one class with `__iter__` yielding `Frame` and a `close()`, register it in
`runner.build_source()`. Nothing else changes. The adapters are 60–117 lines.

## The three order checks

`OrderValidator.validate()` runs them in order; each can fail the order alone.

| # | Check | Fails with |
|---|-------|-----------|
| 1 | Required hotdogs present | `MISSING_HOTDOG` |
| 2 | Every required item applied, per group | `MISSING_ITEM` |
| 3 | Nothing applied that wasn't asked for | `UNEXPECTED_ITEM` / `FORBIDDEN_ITEM` / `WRONG_QUANTITY` |

### Check 3 is new

The existing pipeline **cannot** perform it. `src/state_machine.py` drops any
place event whose well is not on the ticket before the validator sees it:

```python
# Only count ingredients that are required by the active KDS ticket
if self.batch_validator and norm_ing not in self.batch_validator.required_counts:
    return matched_key
```

So chilli added to a plain dog is indistinguishable from a correct plain dog.
Here the engine observes **every** well and the validator decides what the
observation meant. The decision moved from the event-producing edge to the
judging centre — that is the whole fix.

Extras are graded, because they don't all deserve the same response:

- `FORBIDDEN` — ticket said NO ONIONS, onions went on. **Always fails**, even in
  advisory mode: an explicit instruction was violated.
- `UNEXPECTED` — a well not on the ticket at all. Fails only when
  `advisory: false`.
- `OVER` — a required item applied more than asked, past `over_tolerance`
  (default 1, because sauce passes are bursty).

### Roll it out in advisory mode first

```yaml
extras:
  advisory: true       # report extras, never fail an order on them
  over_tolerance: 1
  ignore: [sauce_vessel, cheese_region]
```

Run advisory on real traffic, read the `extras` field, then flip to `false`.
A brand-new signal should not be allowed to fail orders on day one.

## Groups

`TicketSpec` keeps the per-hotdog structure that `required_counts` flattens
away, so "hotdog 1 got both mustards, hotdog 2 got none" is representable.
It reads all four ticket shapes already in the repo (`hotdog1: [...]`,
`hotdog_specs`, `line_items`, flat `expected_items`) plus OCR
`TicketSnapshot`, and normalises every name through
`core.naming.normalize_item_name` — so OCR's "Pickles (Rounds)" and JSON's
`pickle_rounds` land on one key.

`NO ONIONS` / `W/O RELISH` become `ItemReq(negated=True)`.

### Items with no well are reported, not ignored

Pass `known_items` and anything the ticket names that has no well lands in
`spec.unverifiable`. On the real `config/kds_mock.json` this immediately
surfaces that the ticket asks for `yellow_mustard_sauce` while `zones.json`
has only a generic `sauce vessel` well — previously that requirement silently
passed. It is now reported as *not checkable*, which is the honest answer.

## Tests

```bash
python -m unittest tests.test_order_rules -v     # 23 tests, no GPU, ~0.1s
```

Record any real run and replay it to re-judge it after a rules change:

```bash
python -m src.core.runner pipelines/deepstream.yaml --record output/frames.jsonl
python -m src.core.runner pipelines/replay.yaml      # same frames, new rules
```

`test_replay_source_reproduces_the_live_verdict` asserts the replay path and the
live path give byte-identical verdicts.
