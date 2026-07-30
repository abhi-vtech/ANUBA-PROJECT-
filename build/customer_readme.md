# Object Action Detection — Pipeline

Real-time kitchen order verification: detects hand picks and places from an overhead camera, validates them against KDS (Kitchen Display System) tickets, and reports pass/fail on a live dashboard.

This folder (`External/`) is a **self-contained distribution**. The binary (`bin/oad-pipeline`) is platform-specific — use the build matching your operating system (macOS or Linux). No Python, no source code, and no build tools are needed on your machine.

> ⚠️ **Platform exclusivity**
> A bundle built on macOS will **not** run on Linux or Windows (and vice versa). Nuitka compiles to native machine code and bundles OS-specific libraries (`.dylib` on macOS, `.so` on Linux). There is no cross-compilation — to ship for a different OS you must run the build script on that OS.

---

## What you need

| Requirement | Details |
|---|---|
| **Computer** | macOS (Apple Silicon or Intel), Linux (x86_64 or ARM64), or Windows (x64) |
| **Camera** | Overhead webcam, IP camera (RTSP), or pre-recorded video file |
| **Disk space** | ~2 GB free (the bundle is ~1.5 GB) |
| **RAM** | 4 GB minimum, 8 GB recommended |

---

## Step-by-step — first run

### 1. Unzip the bundle

```bash
unzip oad-pipeline-v0.1.0-macos-arm64.zip
cd External/
```

### 2. (Optional) Replace the sample video

The bundle includes a small sample clip at `input/sample.mp4` for testing. To use your own video:

```bash
# Copy your video into the input/ folder
cp /path/to/your/video.mp4 input/my_video.mp4

# Tell the pipeline to use it
# (see "Configuration" below)
```

### 3. Run the pipeline

**macOS / Linux:**
```bash
./setup/run.sh
```

**Windows (PowerShell or Command Prompt):**
```powershell
.\setup\run.ps1
```

> **Note:** If `run.ps1` is not present, open PowerShell in the `External/` folder and run:
> ```powershell
> $env:PATH += ";$PWD\libraries"
> .\bin\oad-pipeline.exe
> ```

You will see output like:
```
Starting oad-pipeline...
  EXTERNAL_DIR      = /path/to/External
  VIDEO_SOURCE      = input/sample.mp4
  MODEL_PATH        = /path/to/External/rf_trained/yolo26n.pt
  LOG_LEVEL         = WARNING
  Dashboard         = http://localhost:8000
```

Wait **10–15 seconds** for the model to load (first time only). Then open **http://localhost:8000** in a browser for the live dashboard.

### 4. Stop the pipeline

Press `Ctrl+C` in the terminal, or run:

**macOS / Linux:**
```bash
pkill -f oad-pipeline
```

**Windows (PowerShell):**
```powershell
Stop-Process -Name oad-pipeline -ErrorAction SilentlyContinue
```

---

## Video sources

Set `VIDEO_SOURCE` to one of:

| Value | Meaning |
|---|---|
| `0` | Built-in webcam (macOS / Linux) |
| `1`, `2`, ... | Other webcam indices |
| `path/to/clip.mp4` | Local video file (auto-replays on EOF) — relative to `External/` |
| `rtsp://user:pass@host/stream` | RTSP network camera |

**How to change:** Edit `config/model.yaml` and set the `source:` key, or set the `VIDEO_SOURCE` environment variable before running:

**macOS / Linux:**
```bash
VIDEO_SOURCE=input/my_video.mp4 ./setup/run.sh
```

**Windows (PowerShell):**
```powershell
$env:VIDEO_SOURCE = "input\my_video.mp4"
.\setup\run.ps1
```

**RTSP example:**
```bash
VIDEO_SOURCE=rtsp://admin:password@192.168.1.50:554/stream ./setup/run.sh
```

**Webcam example:**
```bash
VIDEO_SOURCE=0 ./setup/run.sh
```

---

## Configuration

All config lives in `config/`. Edit YAML/JSON in place with any text editor; the binary reads these files on startup.

### `config/model.yaml` — Main settings

| Key | Default | Description |
|---|---|---|
| `source` | `input/sample.mp4` | Video input: file path, `0` (webcam), or `rtsp://...` |
| `model_path` | `rf_trained/yolo26n.pt` | Model weights (`.pt` file or `.mlpackage` directory) |
| `model_type` | `yolo` | `yolo`, `yoloe`, or `yolo-world` |
| `tracker_type` | `botsort` | `botsort`, `bytetrack`, or `deepsort` |
| `confidence_threshold` | `0.5` | Detection confidence cutoff (0.0–1.0). Lower = more detections, more false positives |
| `pick_dwell_ms` | `200` | Min hand dwell in a bin zone to register a **pick** (milliseconds) |
| `place_dwell_ms` | `500` | Min hand dwell in assembly zone to register a **place** (milliseconds) |
| `frame_width` | `1280` | Video frame width. Used for zone coordinate scaling |
| `frame_height` | `720` | Video frame height. Used for zone coordinate scaling |
| `fps` | `30` | Playback pacing for file sources. Does not affect real-time camera sources |
| `kds_mode` | `mock` | `mock` (read from `kds_mock.json`) or `dynamic` (random ticket generation) |
| `kds_poll` | `2` | Seconds between KDS ticket checks |
| `kds_history` | `logs/orders.jsonl` | Where completed orders are written |

**Example — change to a new video and lower detection threshold:**
```yaml
source: input/my_video.mp4
confidence_threshold: 0.4
pick_dwell_ms: 150
```

### `config/zones.json` — Polygonal regions

Polygonal ROIs for ingredient bins and the assembly zone. Coordinates are **normalized** `[x, y]` in the range `[0, 1]`.

**Why normalized?** The same zone config works for any camera resolution. `x = pixel_x / frame_width`, `y = pixel_y / frame_height`.

**Example zone:**
```json
{
  "id": "zone_01",
  "name": "tomato",
  "zone_type": "bin",
  "color": "#ec4899",
  "polygon": [
    [0.61198, 0.32870],
    [0.61042, 0.41759],
    [0.55677, 0.41759],
    [0.55937, 0.32870]
  ]
}
```

| Field | Meaning |
|---|---|
| `id` | Unique zone identifier |
| `name` | Human-readable name. **Must match** an item in a KDS ticket's `expected_items` list |
| `zone_type` | `bin` (ingredient source — hand dwelling here = pick) or `assembly` (drop-off zone — hand leaving here after carrying = place) |
| `color` | Hex color for dashboard display |
| `polygon` | Array of `[x, y]` points defining the zone boundary (normalized 0–1) |

**To add a new zone:**
1. Open a frame from your camera in any image editor (Preview, GIMP, Photoshop)
2. Note the pixel coordinates of the zone corners
3. Divide each coordinate by the frame size:
   - `x = pixel_x / 1280` (if frame_width is 1280)
   - `y = pixel_y / 720` (if frame_height is 720)
4. Add the new zone object to `config/zones.json`
5. Restart `./setup/run.sh`

### `config/kds_mock.json` — Simulated KDS tickets

```json
{
  "tickets": [
    {
      "ticket_id": "V5-004",
      "expected_items": [
        "yellow cheese (sliced)",
        "chilli",
        "yellow_mustard_sauce",
        "sport (wax) peppers"
      ]
    }
  ],
  "aliases": {
    "cheese": "yellow cheese (sliced)"
  }
}
```

| Field | Meaning |
|---|---|
| `ticket_id` | Unique ticket identifier |
| `expected_items` | List of item names. Each name must match a zone `name` from `zones.json` |
| `aliases` | (Optional) Maps shorthand names to full zone names. A ticket saying `"cheese"` will be treated as `"yellow cheese (sliced)"` |

**How it works:** When a hand enters a bin zone, the system starts a dwell timer. If the hand stays longer than `pick_dwell_ms`, a **pick** is registered. When the tracked hand then enters the assembly zone and stays longer than `place_dwell_ms`, a **place** is registered. The system checks whether the picked item's zone `name` appears in the active ticket's `expected_items`. If yes → `MATCH`. If no → `EXTRA`. If the ticket has items not yet picked → `MISSING`.

### `config/tracker.yaml` — Tracker tuning

Usually leave at defaults. Change only if you see tracking issues:

| Key | Default | When to change |
|---|---|---|
| `tracker_type` | `botsort` | Switch to `bytetrack` or `deepsort` if BoT-SORT performs poorly |
| `track_buffer` | `150` | Increase if tracks are lost during brief occlusions |
| `track_high_thresh` | `0.5` | Decrease if detections are being missed |
| `optical_flow.enabled` | `true` | Set `false` to disable optical flow co-motion analysis |

---

## Environment variables

Every config key can be overridden via environment variable (env wins over file). This is useful for quick testing without editing files.

**macOS / Linux:**
```bash
VIDEO_SOURCE=input/my_video.mp4 \
MODEL_PATH=rf_trained/my_model.pt \
CONFIDENCE_THRESHOLD=0.4 \
LOG_LEVEL=INFO \
./setup/run.sh
```

**Windows (PowerShell):**
```powershell
$env:VIDEO_SOURCE = "input\my_video.mp4"
$env:MODEL_PATH = "rf_trained\my_model.pt"
$env:CONFIDENCE_THRESHOLD = "0.4"
$env:LOG_LEVEL = "INFO"
.\setup\run.ps1
```

| Variable | Default | Description |
|---|---|---|
| `VIDEO_SOURCE` | from config | `0`, file path (relative to External/), or `rtsp://...` |
| `MODEL_PATH` | from config | Path to model file or `.mlpackage` |
| `MODEL_TYPE` | `yolo` | `yolo`, `yoloe`, or `yolo-world` |
| `TRACKER_TYPE` | `botsort` | `botsort`, `bytetrack`, or `deepsort` |
| `CONFIDENCE_THRESHOLD` | `0.5` | Float |
| `PICK_DWELL_MS` | `200` | Integer ms |
| `PLACE_DWELL_MS` | `500` | Integer ms |
| `FRAME_WIDTH` | `1280` | Integer pixels |
| `FRAME_HEIGHT` | `720` | Integer pixels |
| `FPS` | `30` | Integer |
| `KDS_MODE` | `mock` | `mock` or `dynamic` |
| `OPTICAL_FLOW_ENABLED` | `true` | `true` / `false` |
| `LOG_LEVEL` | `WARNING` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_METRICS_INTERVAL` | `5` | Seconds between JSON metric logs |

---

## Updating the model

The model lives in `rf_trained/`. To swap in a different model:

1. **Stop** the running binary (`Ctrl+C`)
2. **Copy** your new model into `rf_trained/`:
   ```bash
   cp /path/to/your/new_model.pt rf_trained/
   ```
3. **Edit** `config/model.yaml` to point at it:
   ```yaml
   model_path: rf_trained/new_model.pt
   ```
4. **Restart** `./setup/run.sh`

Supported model formats:
- `.pt` — PyTorch weights (fastest to swap, largest file)
- `.mlpackage` — CoreML directory (macOS optimized, smaller)

---

## Outputs

After running, these files/folders are created:

| Path | Content |
|---|---|
| `output/orders.jsonl` | One JSON object per completed order, append-only log |
| `output/orders.json` | Full snapshot of all orders, rewritten on each completion |
| `logs/` | Runtime logs. To capture: `./setup/run.sh > logs/pipeline.log 2>&1` |
| `http://localhost:8000` | Live dashboard: MJPEG video feed, WebSocket event stream, current ticket/order/stats |

---

## Troubleshooting

### "Model not found"

**Cause:** `MODEL_PATH` is wrong or the model file isn't at `rf_trained/`.

**Fix:**
```bash
ls rf_trained/           # Check what's there
cat config/model.yaml     # Check model_path value
```

### "Library not loaded" / "image not found" / "libcoremltools.dylib not found"

**Cause:** The dynamic linker can't find the bundled libraries in `libraries/`.

**Fix:** Always run via the launcher:
```bash
./setup/run.sh
```

If running the binary directly, you must set the path manually:
```bash
# macOS
DYLD_LIBRARY_PATH=libraries ./bin/oad-pipeline

# Linux
LD_LIBRARY_PATH=libraries ./bin/oad-pipeline

# Windows (PowerShell)
$env:PATH += ";$PWD\libraries"
.\bin\oad-pipeline.exe
```

### "No zones defined"

**Cause:** `config/zones.json` is missing or empty.

**Fix:** Check the file exists and contains at least one zone object. Restore from your original bundle if corrupted.

### Dashboard not loading

**Cause 1:** The binary is still loading the model (takes 10–15 seconds on first run).

**Fix:** Wait, then refresh the browser.

**Cause 2:** Port 8000 is already in use.

**Fix:**
```bash
# macOS / Linux
lsof -i :8000       # Find what's using it
pkill -f oad-pipeline # Stop the old instance
./setup/run.sh      # Restart

# Windows (PowerShell)
Get-NetTCPConnection -LocalPort 8000   # Find what's using it
Stop-Process -Name oad-pipeline -ErrorAction SilentlyContinue
.\setup\run.ps1
```

### Picks not registering

**Cause:** The dwell time is too high, or the confidence threshold is too high.

**Fix:** Edit `config/model.yaml`:
```yaml
pick_dwell_ms: 150          # Lower = more sensitive
confidence_threshold: 0.4   # Lower = more detections
```

Optical flow (enabled by default) reduces dwell when hand + zone move together. If you want pure dwell-based detection, disable it in `config/tracker.yaml`:
```yaml
optical_flow:
  enabled: false
```

### Performance is slow

**Cause:** The model is the bottleneck.

**Fix:**
1. Use a smaller model (e.g., `yolo11n.pt` instead of `yolo26s.pt`)
2. Lower resolution: set `frame_width: 640`, `frame_height: 480` in `config/model.yaml`
3. Use a `.mlpackage` model on macOS (CoreML is hardware-accelerated on Apple Silicon)

---

## File layout

```
External/                       ← ship this folder
├── bin/
│   └── oad-pipeline            ← the compiled binary (~1.2 GB)
│                                 (oad-pipeline.exe on Windows)
├── libraries/                  ← bundled dependencies (~360 MB)
│                                 (.dylib on macOS, .so on Linux, .dll on Windows)
├── config/                     ← editable YAML/JSON configs
│   ├── model.yaml              ← video source, resolution, model, thresholds
│   ├── tracker.yaml            ← tracker tuning
│   ├── zones.json              ← polygonal ROIs
│   └── kds_mock.json           ← simulated KDS tickets
├── templates/
│   └── index.html              ← dashboard frontend
├── rf_trained/                 ← model weights (.pt or .mlpackage)
├── input/                      ← sample clip + your own videos
├── output/                     ← order reports (created on first run)
├── logs/                       ← runtime logs (created on first run)
├── setup/
│   └── run.sh                  ← launcher (sets library path, execs binary)
│                                 (run.ps1 on Windows)
└── readme.md                   ← this file
```
