# Object Action Detection

Restaurant food preparation monitoring system using computer vision. Tracks hands and tools from an overhead camera, validates ingredient picks/places against spatial zones and dwell-time thresholds, and reconciles physical assembly with Kitchen Display System (KDS) orders.

## Repository layout

This repo is split into two top-level folders:

- **`Internal/`** — everything for the development team: source code, tests, training scripts, build infrastructure, dev configs, and this README. Shipped via git.
- **`External/`** — the **customer-facing deliverable**. Built from `Internal/` by `build/build_nuitka.sh`. Customers receive this folder only — no source code, no model training data, no build scripts. See `External/readme.md` for the customer guide.

---

## Prerequisites

Before you can run the pipeline or build the customer bundle, you need:

| Requirement | Version | Why |
|---|---|---|
| **Python** | 3.10+ | Runtime language |
| **UV** | latest | Python package manager (replaces pip + venv) |
| **C compiler** | clang (macOS) or gcc (Linux) | Required by Nuitka and some Python wheels |
| **ccache** | latest | Speeds up Nuitka rebuilds (strongly recommended) |
| **ffmpeg** | 5.0+ | Video clip extraction and preprocessing |

### Install prerequisites

**macOS (Apple Silicon or Intel):**
```bash
# 1. Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install tools
brew install uv ccache ffmpeg

# 3. Xcode Command Line Tools (provides clang)
xcode-select --install
```

**Linux (Ubuntu/Debian):**
```bash
# 1. Install tools
sudo apt-get update
sudo apt-get install -y python3-dev python3-venv build-essential ccache ffmpeg

# 2. Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (native or WSL2):**

> **Recommendation:** Use **WSL2** (Windows Subsystem for Linux) and follow the Linux instructions above. The build script (`build_nuitka.sh`) is a Bash script and the easiest path is WSL2.
>
> If you must build natively on Windows (no WSL), you need:

```powershell
# 1. Install Python 3.10+ from python.org or Microsoft Store
# 2. Install UV
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 3. Install Visual Studio Build Tools 2022 (C++ compiler)
#    https://visualstudio.microsoft.com/visual-cpp-build-tools/
#    Select: "Desktop development with C++" workload

# 4. Install ccache and ffmpeg via Chocolatey
choco install ccache ffmpeg
```

---

## Environment setup (from scratch)

```bash
# 1. Clone the repo
git clone <repo-url>
cd object-action-detection/Internal

# 2. Create the virtual environment and install dependencies
uv sync

# 3. Verify the environment
uv run python -c "import ultralytics, cv2, fastapi; print('OK')"
```

The `uv sync` command reads `pyproject.toml` and `uv.lock` and installs every dependency into `Internal/.venv/`.

---

## Quick Start — run the dev pipeline

```bash
# From the Internal/ directory:
uv run python -m src.main
```

Then open **http://localhost:8000** for the live dashboard.

To run with a specific video:
```bash
uv run python -m src.main --source videos/clips/v5/seg_seg001.mp4
```

---

## Building the customer bundle (External/)

The customer receives `External/` — a self-contained folder with a compiled binary, bundled libraries, config, and model. No Python, no source code, no build tools needed on their machine.

> ⚠️ **Build on the target OS**
> Nuitka produces a native binary for the machine it runs on. A build done on macOS creates a macOS-only bundle; a build done on Linux creates a Linux-only bundle. There is no cross-compilation. If you need to ship to both macOS and Linux, run the build script on each platform separately.

### Step 1 — install Nuitka into the dev environment

```bash
cd Internal/
uv pip install nuitka
```

Nuitka is **not** listed in `pyproject.toml` because it's a build-only dependency, not a runtime dependency.

### Step 2 — place your model

The build script looks for a model at `rf_trained/`:
```bash
ls rf_trained/
# Expected: yolo26n.pt, yolo11n.pt, or a .mlpackage directory
```

If you don't have a trained model yet, you can:
- Use a baseline YOLO model (e.g. `yolo11n.pt` from Ultralytics)
- Train your own: see [Training Pipeline](#training-pipeline) below
- Export to CoreML: see [Export Formats](#export-formats) below

### Step 3 — place a sample video (optional but recommended)

The build script bundles a sample clip so the customer can test immediately:
```bash
# The build script auto-finds videos in these locations:
#   - ../videos/clips/v5/seg_seg004.mp4  (repo-level videos/)
#   - input/sample.mp4                  (Internal/input/)

mkdir -p input
cp your_sample_clip.mp4 input/sample.mp4
```

### Step 4 — run the build

**macOS / Linux:**
```bash
# From the Internal/ directory:
./build/build_nuitka.sh
```

**Windows (native, no WSL):**
Nuitka on Windows requires a different invocation. You can either:
1. Open **Developer PowerShell for VS 2022** and run the equivalent commands manually (see `build/build_nuitka.sh` for flags), or
2. Use WSL2 and run the Bash script above (recommended).

What the script does:
1. Checks that Python, Nuitka, and all runtime deps are available
2. Compiles `src/main.py` into a standalone binary via Nuitka (takes 3–10 minutes on first run, ~1 minute with ccache)
3. Assembles `External/` at the repo root with:
   - `bin/oad-pipeline` — the compiled binary (~1.2 GB)
   - `libraries/` — bundled `.dylib`/`.so` dependencies (~360 MB)
   - `config/` — customer-editable YAML/JSON configs
   - `templates/` — dashboard HTML
   - `rf_trained/` — the model
   - `input/sample.mp4` — bundled sample video
   - `setup/run.sh` — launcher script
   - `readme.md` — customer-facing documentation

### Platform-specific build notes

| Platform | Binary name | Libraries | CoreML support | Packaging |
|----------|-------------|-----------|----------------|-----------|
| **macOS** | `bin/oad-pipeline` | `.dylib` | ✅ `.mlpackage` hardware-accelerated | `zip` |
| **Linux** | `bin/oad-pipeline` | `.so` | ❌ Not available | `tar.gz` |
| **Windows** | `bin/oad-pipeline.exe` | `.dll` | ❌ Not available | `zip` |

> **Linux tip:** Use a `.pt` or `.engine` model, not `.mlpackage` (CoreML is Apple-only). Set `DYLD_LIBRARY_PATH` → `LD_LIBRARY_PATH` in `run.sh` (already handled automatically).
>
> **Windows tip:** If building natively, replace `--include-package-data=jinja2` and other Unix-style paths in the build script with Windows equivalents. The launcher script (`run.sh`) already detects Windows (`OSTYPE==msys`) and picks `.exe`.

### Step 5 — verify the bundle

```bash
cd ../External
./setup/run.sh
```

Wait ~10 seconds for the model to load, then open **http://localhost:8000**.

### Step 6 — ship to customer

**macOS:**
```bash
cd ..
zip -r oad-pipeline-v0.1.0-macos-arm64.zip External/
```

**Linux:**
```bash
cd ..
tar czf oad-pipeline-v0.1.0-linux-x86_64.tar.gz External/
```

**Windows:**
```powershell
# In PowerShell
Compress-Archive -Path External\ -DestinationPath oad-pipeline-v0.1.0-windows-x64.zip
```

The customer unzips and runs `./setup/run.sh` (macOS/Linux) or `setup\run.cmd` (Windows). No installation needed.

### Build requirements (summary)

| What | Why | Check with |
|---|---|---|
| Python 3.10+ | Nuitka compiles from Python source | `python --version` |
| C compiler | Nuitka translates Python to C then compiles | `clang --version` or `gcc --version` |
| ccache | Caches object files; rebuilds go from 10 min → 1 min | `ccache --version` |
| Nuitka 4.1.2+ | The compiler itself | `python -c "import nuitka"` |
| All runtime deps | Must be importable so Nuitka can analyze them | `uv run python -c "import ultralytics, cv2, fastapi"` |
| ~4 GB free disk | The build produces a 1.5 GB bundle + intermediate files | `df -h .` |
| ~8 GB RAM | Loading torch + compiling large C files | N/A |

---

## Architecture

Four-layer pipeline:

1. **Dynamic Tracking** — YOLO object detection with BoT-SORT tracking for hands, tongs, and squeeze bottles. Supports PyTorch, CoreML (`.mlpackage`), TensorRT (`.engine`), and ONNX exports.
2. **Spatial Zoning** — Pre-mapped polygonal ROIs for ingredient bins and the assembly zone, loaded from `config/zones.json`.
3. **Temporal Action Validation** — Dwell-time thresholds (`pick_dwell_ms`, `place_dwell_ms`) to confirm purposeful picks vs. transient motion.
4. **State Machine** — Reconciles KDS orders against physical assembly, computes pass/fail and error rates.

---

## End-to-end code flow: how detection and counting work

The runtime starts in [src/main.py](src/main.py). That file loads the YAML config, creates the detector, zone manager, temporal tracker, optical-flow analyzer, and the additive hotdog tracker, then runs the camera/video frame loop.

### 1) Frame capture and detection

- [src/capture.py](src/capture.py) reads the source video or webcam frames.
- [src/detector.py](src/detector.py) runs YOLO inference with `model.track(...)` using the configured tracker.
- Every detected object becomes a `Detection` dataclass containing:
  - `track_id`
  - `bbox = (x1, y1, x2, y2)`
  - `class_name`
  - `confidence`
- The detector chooses the GPU when CUDA is available and otherwise falls back to CPU.

### 2) Spatial zoning and hand-state validation

- [src/zones.py](src/zones.py) loads the polygonal zone map from [config/zones.json](config/zones.json).
- [src/temporal.py](src/temporal.py) receives the detections, the zone manager, and the frame size.
- For each tracked hand, it keeps a `TrackState` with:
  - current zone
  - entry time
  - pending pick queue
  - carried items
  - centroid trajectory
- `TemporalTracker.update()` converts raw movement into a state machine:
  - `idle`
  - `idle_in_zone`
  - `pending_pick`
  - `transit_pending`
  - `carrying`
  - `carrying_in_assembly`
- Dwell timing (`pick_dwell_ms`, `place_dwell_ms`, timeout windows) is used to distinguish a real pick from a brief hover.

### 3) Hotdog-level ingredient counting (additive path)

- [src/hotdog_tracker.py](src/hotdog_tracker.py) is an additive layer. It does not replace the main temporal/state logic. It watches the live detections and associates items with a specific hotdog.
- A hotdog record keeps:
  - `hotdog_id`
  - `track_id`
  - `items_added`
  - `item_counts`
- The tracker normalizes item aliases such as `sport (wax) peppers` → `sport_peppers` and `chili` → `chilli`.
- For each non-hotdog detection, it checks whether the item overlaps the hotdog bounding box.
- It then applies a dwell gate before recording the item. That prevents a quick hover from being counted as a real application.
- The per-item dwell override allows classes like `sport_peppers` to require a longer sustained overlap before they are committed.
- Once the dwell threshold is satisfied and the hand gate is satisfied, the item is appended to `items_added` and the count is incremented in `item_counts`.

### 4) Order reconciliation and final report

- [src/state_machine.py](src/state_machine.py) compares the physical ingredient timeline to the expected KDS order.
- It decides whether the order is passed/failed and builds the remaining item count, extra item, and missing item details.
- [src/main.py](src/main.py) emits metrics logs and the final `hotdog_summary` event at the end of the run.
- [run_10min_video.py](run_10min_video.py) is the batch runner used for video replay. It launches the pipeline on a fixed video and writes the summary JSON to [output/hotdog_summary.json](output/hotdog_summary.json).

### 5) What the final JSON looks like

The final summary output contains:

- `total_hotdogs`
- `orders`
  - `hotdog_id`
  - `track_id`
  - `item_names`
  - `item_counts`
  - `items_added`
  - `completed`

That is the part that makes the chilli count visible in the final report, rather than only in the temporary live tracker state.

---

## Project Structure

```
Internal/
├── src/                          # Core application modules
│   ├── main.py                   # Entry point: capture → detect → track → validate → dashboard
│   ├── capture.py                # Threaded video capture (webcam / file / RTSP)
│   ├── detector.py               # YOLO inference + BoT-SORT tracking wrapper
│   ├── zones.py                  # Polygonal zone manager (bins + assembly)
│   ├── temporal.py               # Dwell-time action validator (pick / place)
│   ├── state_machine.py          # Order reconciliation & validation logic
│   ├── kds_client.py             # KDS client (abstract + mock JSON implementation)
│   ├── dashboard.py              # FastAPI dashboard (WebSocket + REST + MJPEG stream)
│   ├── schemas.py                # Dataclasses: Detection, Zone, Action, Ticket, Order, Stats
│   ├── tracker.py                # Additional tracking utilities
│   └── paths.py                  # Path resolver (dev vs. frozen binary)
├── config/
│   ├── model.yaml                # Model, source, dwell thresholds, frame size
│   ├── tracker.yaml              # BoT-SORT tracker configuration
│   ├── zones.json                # Polygon definitions for bins & assembly zone
│   └── kds_mock.json             # Simulated KDS ticket queue
├── training/                     # Standalone training pipeline
│   ├── collect_frames.py         # Extract frames from video
│   ├── prepare_data.py           # Organize images/labels into YOLO dataset
│   └── train.py                  # Fine-tune YOLO on custom data
├── utils/
│   ├── export.py                 # CoreML / ONNX export
│   ├── export_tensorrt.py        # TensorRT export (Jetson)
│   ├── extract_frames.py         # Batch frame extraction
│   ├── video_cropper.py          # CLI video cropping
│   └── video_cropper_gui.py      # GUI video cropping
├── tests/
│   └── test_zones.py             # Zone geometry tests
├── docs/
│   ├── architecture.md           # Layered architecture reference
│   └── method.md                 # Implementation methodology
├── templates/
│   └── index.html                # Dashboard frontend
├── weights/                      # Trained model weights
├── rf_trained/                   # Roboflow-trained models + CoreML packages
├── dataset_raw_frames/           # Raw frame dumps for training
├── build/
│   ├── build_nuitka.sh           # Build script: compiles binary + assembles External/
│   ├── run.sh                    # Launcher script (copied to External/setup/)
│   └── customer_readme.md        # Customer-facing docs (copied to External/readme.md)
├── pyproject.toml                # UV project config + dependencies
├── Dockerfile                    # CUDA 12.4 GPU image for T4 deployment
├── docker-compose.yml            # Docker Compose with GPU passthrough & env config
├── uv.lock                       # Pinned dependency lockfile
└── README.md                     # This file
```

---

## Docker Deployment (NVIDIA T4 GPU)

### Prerequisites

- EC2 instance with NVIDIA T4 GPU (e.g. `g4dn.xlarge`)
- NVIDIA driver + container toolkit installed:
  ```bash
  # Ubuntu 22.04 example
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-container-toolkit/gpgkey | sudo apt-key add -
  curl -s -L https://nvidia.github.io/nvidia-container-toolkit/$distribution/nvidia-container-toolkit.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```

### Run with Video File

```bash
# Copy your video and model weights into the project
scp sample_video.mp4 ec2-user@<instance>:/app/videos/
scp -r rf_trained/ ec2-user@<instance>:/app/

# Build and run
docker compose up -d --build
```

### Run with RTSP Camera

```bash
VIDEO_SOURCE=rtsp://user:pass@192.168.1.100:554/stream docker compose up -d
```

### Environment Variables

All config values can be overridden via environment variables (take precedence over `config/model.yaml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_SOURCE` | from config | Video input: file path, `0` (webcam), or `rtsp://...` |
| `MODEL_PATH` | `rf_trained/yolo26n.mlpackage` | Model weights path |
| `MODEL_TYPE` | `yolo` | Model family (`yolo`, `yoloe`, `yolo-world`) |
| `TRACKER_TYPE` | `botsort` | Tracker backend (`botsort`, `bytetrack`, `deepsort`) |
| `CONFIDENCE_THRESHOLD` | `0.5` | Detection confidence cutoff |
| `PICK_DWELL_MS` | `200` | Minimum dwell time for pick registration |
| `PLACE_DWELL_MS` | `500` | Minimum dwell time for place registration |
| `FRAME_WIDTH` | `1280` | Target frame width |
| `FRAME_HEIGHT` | `720` | Target frame height |
| `FPS` | `30` | Playback pacing for file sources |
| `KDS_MODE` | `mock` | KDS client mode (`mock` or `dynamic`) |
| `OPTICAL_FLOW_ENABLED` | `true` | Enable optical flow co-motion analysis |
| `LOG_METRICS_INTERVAL` | `5` | Seconds between performance metric logs |
| `LOG_LEVEL` | `WARNING` | Python log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### Performance Metrics

Metrics are logged as structured JSON every `LOG_METRICS_INTERVAL` seconds:

```
2026-05-28 10:15:30 src.metrics INFO: {"event":"metrics","fps":29.8,"frame_count":1490,"elapsed_s":50.0,"loop_ms":33.5,"detect_ms":12.1,"flow_ms":1.8,"temporal_ms":0.4,"detections":{"hand":2},"total_orders":5,"passed_orders":3,"failed_orders":1,"accuracy_pct":60.0,"error_rate_pct":20.0}
```

Stream metrics logs:

```bash
docker compose logs -f oad-pipeline 2>&1 | grep "src.metrics"
```

### Useful Commands

```bash
# Rebuild after dependency or Dockerfile changes
docker compose up -d --build

# View live logs
docker compose logs -f oad-pipeline

# Stop
docker compose down

# Check GPU utilization inside container
docker compose exec oad-pipeline nvidia-smi
```

---

## Configuration

### `config/model.yaml`

| Key | Description | Example |
|-----|-------------|---------|
| `source` | Video input: `0` (webcam), file path, or `rtsp://...` | `sample_video.mp4` |
| `model_type` | Model family: `yolo`, `yoloe`, `yolo-world` | `yolo` |
| `model_path` | Model file: `.pt`, `.mlpackage`, `.engine` | `rf_trained/yolo11n.mlpackage` |
| `prompt_classes` | Text prompts for open-vocabulary models (optional) | `["hand", "tongs"]` |
| `pick_dwell_ms` | Minimum dwell time to register a **pick** | `800` |
| `place_dwell_ms` | Minimum dwell time to register a **place** | `500` |
| `frame_width` / `frame_height` | Target resolution | `1280` / `720` |
| `fps` | Playback pacing for file sources | `30` |
| `confidence_threshold` | Detection confidence cutoff | `0.5` |

### `config/tracker.yaml`

BoT-SORT tracker parameters (loaded by `detector.py`):

| Key | Default | Description |
|-----|---------|-------------|
| `tracker_type` | `botsort` | Tracker algorithm |
| `track_high_thresh` | `0.5` | High-confidence threshold for new tracks |
| `track_low_thresh` | `0.1` | Low-confidence threshold for matching |
| `new_track_thresh` | `0.6` | Threshold to spawn a new track |
| `track_buffer` | `120` | Frames to keep lost tracks alive |
| `match_thresh` | `0.8` | IoU matching threshold |
| `fuse_score` | `True` | Fuse appearance + IoU scores |
| `gmc_method` | `sparseOptFlow` | Global motion compensation |
| `with_reid` | `False` | Enable ReID model |

### `config/zones.json`

Array of zone objects:

```json
{
  "id": "zone_01",
  "name": "lettuce_bin",
  "zone_type": "bin",
  "color": "#8b5cf6",
  "polygon": [[0.28, 0.31], [0.33, 0.31], [0.33, 0.4], [0.28, 0.4]]
}
```

- `zone_type`: `bin` (ingredient source) or `assembly` (drop-off zone)
- `polygon`: Normalized coordinates `[x, y]` in `[0, 1]` range

### `config/kds_mock.json`

Simulated ticket queue:

```json
[
  {"ticket_id": "T001", "expected_items": ["lettuce", "tomato", "patty"]}
]
```

---

## Model Support

| Model | Type | Prompts | Speed | CoreML | Jetson Nano |
|-------|------|---------|-------|--------|-------------|
| YOLOv8 / YOLO11 | Closed-set | None | Fastest ✅ | ✅ `.mlpackage` | ✅ `.engine` |
| YOLOE | Open-vocabulary | Text / Visual | Fast ✅ | ⚠️ Limited | ⚠️ Limited |
| YOLO-World | Open-vocabulary | Text only | Fast ✅ | ⚠️ Limited | ⚠️ Limited |

### Export Formats

```bash
# CoreML (macOS / iOS)
python utils/export.py --weights yolo11n.pt --format coreml

# TensorRT (Jetson / NVIDIA GPU)
python utils/export_tensorrt.py --weights yolo11n.pt --half

# ONNX
python utils/export.py --weights yolo11n.pt --format onnx
```

> **Note:** Open-vocabulary models (YOLOE / YOLO-World) require prompts baked in at export time. Runtime `set_classes()` is **not** supported in CoreML/TensorRT/ONNX.

### YOLOE & YOLO-World on Jetson Nano

- **Memory:** 4 GB shared RAM. Text encoders add ~100–300 MB overhead. Use `half=True` (FP16).
- **Performance:** ~5–15 FPS at 640×480. For 30 FPS real-time, prefer standard closed-set YOLO.
- **Advanced:** Run the text encoder offline to generate class embeddings, then use only the vision backbone at runtime.

---

## Training Pipeline

Training scripts are isolated in `training/` and run independently:

```bash
# 1. Extract frames from video
python training/collect_frames.py video.mp4 -o dataset_raw_frames/

# 2. Organize into YOLO dataset structure
python training/prepare_data.py --images dataset_raw_frames/ --labels data/labels -o data/dataset

# 3. Train
python training/train.py --data data/dataset/data.yaml --epochs 100 --imgsz 640 --weights yolov8n.pt

# 4. Export best weights
python utils/export_tensorrt.py --weights runs/detect/train/weights/best.pt --half
```

---

## Testing

```bash
# Run zone geometry tests
pytest tests/test_zones.py

# Run all tests
pytest
```

---

## Dashboard API

The dashboard (`dashboard.py`) exposes:

- **WebSocket** `/ws` — Real-time detection & event stream
- **REST** `/api/stats` — Current order statistics
- **MJPEG** `/video_feed` — Annotated video stream
- **HTML** `/` — Live dashboard UI (`templates/index.html`)

---

## Development

```bash
# Linting (Ruff)
ruff check src/
ruff format src/

# Type checking (optional)
mypy src/
```

Ruff is configured in `pyproject.toml` with `ANN` (type annotation) rules ignored.

---

## Dependencies

Core dependencies (see `pyproject.toml`):

- `ultralytics>=8.3.0` — YOLO inference & tracking
- `opencv-python>=4.10.0` — Video I/O & annotation
- `fastapi>=0.115.0` + `uvicorn[standard]>=0.32.0` — Dashboard server
- `numpy`, `pydantic`, `pyyaml`, `lap`, `coremltools`
- `clip` (Ultralytics fork) — Text encoder for open-vocabulary models
