# KDS reading via kds-ocr

The KDS screen is read by **[kds-ocr](https://github.com/anuba-technologies/kds-ocr)**
running as a child process. This package consumes what it emits and presents it
to the production pipeline as a ticket source.

```
   KDS SCREEN (RTSP or recording)          PRODUCTION VIDEO
            |                                     |
   kds-ocr  (child process)              Detector + ByteTrack
   segment -> OCR -> parse -> track               |
            |                              zones / TemporalTracker
     recipes.jsonl  (one JSON per emission)       |
            |                                     |
     EmissionTailer                                |
            |                                      |
     IngredientMapper   (their names -> our zones/classes, as counts)
            |                                      |
     KdsOcrClient  ------> OrderStateMachine <-----+
            |                (requirement, progress, verdict)
       JourneyLog ------> dashboard + output/ticket_journeys.jsonl
```

kds-ocr is a **separate process**, not an import: its `reference/` lookups
resolve relative to its own repo root, its EasyOCR/torch stack stays off our
import path, and a crash in the reader cannot take the detection loop down.

## Running

```yaml
# config/model.yaml
kds_mode: kdsocr
kds_source: videos/…__kds__2026_09_13_12_to_13_p0007_PDT.mkv   # or rtsp://…
kdsocr_repo: kds-ocr/
```

```bash
KDS_MODE=kdsocr KDS_SOURCE=videos/…_kds_…mkv python main.py
KDS_MODE=kdsocr KDS_SOURCE="$KDS_RTSP" python main.py          # live
```

The child's own (verbose) log goes to `output/kdsocr/reader.log`; the stream we
consume is `output/kdsocr/recipes.jsonl`. Both are rotated per run, so a stale
file never replays as this run's tickets.

**The KDS screen is not shown here.** kds-ocr emits the ticket as JSON and that
is all this dashboard renders; the screen itself has its own dashboard in the
kds-ocr repo (`scripts/live_status.py`).

## Recording the dashboard

`RECORD_DASHBOARD=<path>.mkv` records the dashboard **as a browser window**,
through the existing `scripts/record_dashboard.py`: a hidden Xorg display,
Firefox in kiosk mode on the page, captured with GStreamer `ximagesrc`
straight into the Jetson's hardware H.264 encoder. The file is the browser
window, not a second rendering of it that could drift from the page.

It runs as a child process and is stopped with SIGINT, never SIGKILL: it has
to close the GStreamer pipeline and tear down Firefox and the display, and a
killed recorder leaves an unplayable file and an orphaned Xorg.

Nothing composes video frames. An earlier attempt here drew a dashboard-alike
with PIL -- it cost ~19 ms on the pinned main loop and was a second thing to
keep in step with the page -- and `scripts/record_dashboard.py` already solved
this properly, so the drawn version was removed.

## The four things this package guarantees

### 1. Quantities are counts, never weights

`mapping.as_count` collapses `oz`, `pinch`, `g`, `ml` to **1** — you cannot
watch 1.5 oz of chili go on, you can only see chili applied — while countable
units (`each`, `slice`, `leaf`) keep their number. kds-ocr already does this in
`recipes/convert.py`; it is re-asserted here because this layer is the boundary,
and a float or an `oz` string from a future upstream change must not silently
become a requirement the crew cannot satisfy.

### 2. The group structure survives

One kds-ocr hot-dog line becomes one `LineItem`: "2 ORG C/C" is one line of
count 2, not two anonymous dogs. kds-ocr has **already** multiplied each
ingredient by the line quantity, while `LineItem.items` is per-hotdog and gets
multiplied again downstream — so `_per_dog` divides it back out, and only claims
the grouping when it divides exactly. An indivisible line keeps its total rather
than being silently rounded away.

### 3. A changed ticket updates the order in place

The crew edits orders while the food is being made. kds-ocr re-emits the whole
ticket on every item change; `take_updates()` reports those, and
`OrderStateMachine.update_ticket_requirements` applies the new requirement
while **keeping** `picked_counts` — what was physically observed going on is not
undone by an edit to the ticket. `remaining_counts` becomes "what the new ticket
asks for, less what we have already seen".

Timer ticks, payment state, amount changes and age colour are **not** updates:
`RecipeEmission.signature` compares only the item set, so they never churn the
order.

### 4. The bump is the verdict trigger

An order is judged when kds-ocr emits `bumped`. Nothing here re-derives "the
card disappeared" — the reader owns that question and answers it explicitly, so
there is no second, disagreeing copy of the rule.

## What is never called wrong

A verdict is only ever recorded for an order we actually had evidence for.
Three cases close a ticket **without** a verdict (`correct = None`), and none of
them counts against accuracy:

| case | why |
| --- | --- |
| `status: voided` | the crew struck the ticket — cancel, do not verify a build |
| `status: blocked` | kds-ocr refused to guess the items; there is no requirement to check |
| bumped before we opened it | the order was made while the pipeline was still on an earlier ticket |
| run ended while still open | no bump ever arrived |

A ticket with **no hot dogs** on it (fries and a drink) is different again: it
is read, understood, and recorded, but it is not a problem and raises no
warning — there is simply nothing for the CV side to verify.

## What reaches the required checklist

`config/kdsocr_ingredients.yaml` is the only place kds-ocr's ingredient names
are mapped to ours. Every ingredient lands in exactly one of three buckets.

### `map` — the toppings we check

19 names, resolving to a bin zone in `config/zones.json` or a YOLO class:
chili, american cheese, shredded cheddar, onion, diced onion, grilled onions,
relish, tomato, tomato half, pickle spear, pickle chips, sport peppers,
sauerkraut, swiss cheese, pastrami, jalapeno poppers, plus mustard and ketchup
as detector classes.

These are the only things that become a *required item*.

### `base` — covered by the hotdog count, dropped silently

The bun and the sausage itself are in **every** hot dog. They are not toppings
a crew member can forget independently — if they were missing there would be no
hot dog at all — so they are covered by the **hotdog count**
(`required_hotdogs`), not by an ingredient check.

They are dropped **silently**: not required, and not reported as "not
checkable" either. Naming `bun` against every single ticket is noise that
buries the toppings that do matter.

The dog *variant* (`all beef dog`, `polish dog`, `veggie dog`, `corn dog`) is
here for the same reason, plus a second: the detector has one undifferentiated
`hot-dog` class, so the variant is unknowable anyway. We count dogs; we do not
tell them apart.

So a real 3-dog ticket becomes exactly this:

```
AB C/C     x1   ->  grated_yellow_cheese 1, chilli 1
PLSH KRAUT x2   ->  yellow_mustard_sauce 1, sauerkraut 1   (per dog)

required checklist:  grated_yellow_cheese 1, chilli 1,
                     yellow_mustard_sauce 2, sauerkraut 2,
                     hot-dog 3
not checkable:       (nothing)
```

A plain dog, which has *only* base components, reduces to `hot-dog: 2` — the
required hotdogs and nothing else.

> `polish dog` sits in `base` even though a `polish hot dog` bin zone exists and
> could in principle be observed. Move that one line back into `map` if you
> later want the polish bin checked as a topping.

### `unverifiable` — real toppings we cannot see

`celery salt`, `bbq sauce`, `bacon`, `mayo`, `ranch`, `lettuce`. A crew member
really can forget these and we would not know, so unlike base components they
**are** carried into the verdict as "not checkable" — a missing signal must
never read as compliance.

An ingredient in **none** of the three lists is `unknown`: logged once, never
guessed at, never dropped from the record.

## Pacing a recorded run

A live run needs no pacing — both feeds are the present moment.

A recorded one does: kds-ocr replays its hour at 1x while our loop runs at
whatever rate the detector manages, so without pacing an hour of tickets can
arrive long before the production video reaches the food they describe.
kds-ocr stamps each emission with `screen_at` (the ticket's time in the
*recording's* clock); `clock.py` turns that into seconds-into-the-recording, and
`set_master_time()` holds each emission until the production feed gets there.
Held emissions are applied in **recording** order, so an `updated` can never
land before its `appeared`.

This needs the filename convention `…_12_to_13_p0007_PDT` = 12:00:07. Without
it, pacing is disabled with a warning and emissions apply as they are read.

## The ticket journey

A verdict alone is not actionable: "missing chilli" does not say whether the
chilli was never applied, was applied before the ticket was read, or whether
the ticket itself changed halfway through. So `journey.py` records every input
that moved the order, in order — each requirement, each observed application,
each finished hotdog, each kds-ocr alert, then the verdict.

It drives the dashboard's **Ticket journey** panel (wrong first, then
unverified, then correct) and is appended to `output/ticket_journeys.jsonl`,
one object per closed ticket.

## Tests

```bash
python -m pytest tests/test_kdsocr.py -q      # 61 tests, no video or OCR model
```

They drive the real mapping, emission, journey, client and pacing code with
hand-written kds-ocr records, so nothing is downloaded and no child process is
started.
