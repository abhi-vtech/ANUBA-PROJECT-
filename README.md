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

## Jetson (JetPack 7.x / Orin) setup

Verified on: **Jetson Orin NX 16 GB**, JetPack 7.2.1, L4T R39.2.0, Ubuntu 24.04 aarch64,
CUDA 13.2, cuDNN 9.20, TensorRT 10.16, driver 595.78, Python 3.12.

### Why `pyproject.toml` differs from a desktop checkout

The dependency set is aarch64-aware. Three things matter on a Jetson:

1. **torch comes from the `cu130` index, not `cu121`.** The old `pytorch-cu121`
   index publishes **x86_64 wheels only** — there is no aarch64 artifact at all,
   so `uv sync` could never resolve on a Jetson. JetPack 7 ships CUDA 13.x, and
   `https://download.pytorch.org/whl/cu130` is the only official channel with
   `manylinux_2_28_aarch64` CUDA wheels.
2. **The index is marker-scoped.** Only `sys_platform == 'linux' and
   platform_machine == 'aarch64'` resolves from it, so macOS and linux-x86_64
   checkouts (and the CoreML `--extra export` path) are unchanged.
3. **torch floor is 2.10.** The cu130 aarch64 channel has no torchvision 0.24,
   which is the only pair for torch 2.9.x — so 2.9.x is unresolvable there.

`torchaudio` and `inference-sdk` were dropped: neither is imported anywhere in
the tree, and together they pulled in ~1 GB (incl. `supervision`) for nothing.
`scipy` was **added** — `src/hotdog_tracker.py` imports
`scipy.optimize.linear_sum_assignment` directly but it was only ever present
transitively via ultralytics.

### Setup

```bash
# uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

cd Internal/
uv sync            # ~5.8 GB venv, several minutes on first run
uv run python -c "import torch;print(torch.cuda.is_available())"   # -> True
```

System packages still needed (JetPack does not ship them):

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg ccache
```

`gcc`, `g++`, `make`, `build-essential` and `python3.12-dev` are already present
on a stock JetPack 7 image.

### Running it

`run.sh` is the Jetson entry point. It creates `.venv` on first use, creates the
gitignored `output/` and `videos/` directories, prints an environment summary,
validates the video source, then execs `python -m src.main`.

```bash
./run.sh                        # source from config/model.yaml
./run.sh videos/kds_3.mp4       # override the source
./run.sh --check                # environment doctor, runs nothing
./run.sh --cpu videos/clip.mp4  # force CPU (debugging)
./run.sh --exit-on-end clip.mp4 # quit when the video ends
```

Every pipeline env var still applies, e.g.
`LOG_LEVEL=INFO CONFIDENCE_THRESHOLD=0.4 ./run.sh videos/clip.mp4`.
The dashboard serves on <http://localhost:8000>.

### Recording the detection video

Set `RECORD_VIDEO` to write the annotated feed — exactly the frame the dashboard
shows, with boxes, masks, zones and trails — to a video file. It is off by
default and changes nothing the pipeline detects.

```bash
./run.sh --record --exit-on-end videos/clip.mp4   # -> output/recordings/clip_<time>.mkv
./bench/transcode.sh output/recordings/clip_<time>.mkv   # -> same folder, .mp4 (H.264)
```

| Variable | Default | Meaning |
|---|---|---|
| `RECORD_VIDEO` | unset (off) | `1` = auto-name in `output/recordings/`; or an explicit path (use `.mkv`) |
| `RECORD_FPS` | pipeline fps, else 30 | playback rate of the file |
| `RECORD_ENCODER` | `nvenc` | `nvenc` = the Jetson's hardware H.264 encoder; `opencv` = CPU mp4v |
| `RECORD_BITRATE` | `4000000` | hardware encoder bitrate, bits per second |
| `RECORD_FOURCC` | `mp4v` | codec tag for the OpenCV fallback |
| `RECORD_HUD` | `1` | burn in source timestamp + frame number |

With GStreamer and the NVIDIA plugins available (see *Hardware video, ONNX on
the GPU, and the Analysis window* below), frames are encoded by
`nvv4l2h264enc`: 300 frames in 1.4 s, all 300 decodable, already H.264, so no
transcode is needed. Without them the recorder falls back to OpenCV mp4v, which
`bench/transcode.sh` converts afterwards as described below.

How it behaves, all measured on camA with the TensorRT engine:

- **Every frame, in order.** Frames go to a writer thread through a bounded
  queue that blocks instead of dropping. A 240-frame clip recorded exactly 240
  frames.
- **About 4-5% overhead.** The encode runs on one of the idle cores rather than
  the pinned main loop: 15.4-15.8 FPS recording vs 16.0-16.6 without.
- **Use `.mkv`, not `.mp4`.** An `.mp4` is unreadable if the process dies before
  it finalises. Hard-killed mid-write, `.mkv` stayed readable, while `.mp4` did
  not.
- **Transcode afterwards, not during.** This OpenCV build cannot write H.264, so
  the recording is `mp4v`. `bench/transcode.sh` converts it with a static ffmpeg
  in `bench/.tools/` (outside the project venv, so `uv sync` won't remove it)
  at ~147 fps, and the file comes out ~4x smaller. It uses every core, so never
  run it while a benchmark is being measured.
- **A stopped run still leaves a playable file.** Stopping by hand (Ctrl+C or
  kill) may skip the clean close, but the `.mkv` stays playable up to the last
  few buffered frames.

### Hardware video, ONNX on the GPU, and the Analysis window

**ONNX models run on the GPU.** On the Jetson, `pyproject.toml` installs
`onnxruntime-gpu` from NVIDIA's Jetson AI Lab index in place of the CPU-only
PyPI build. `src/onnx_gpu.py` then puts ONNX Runtime's TensorRT execution
provider (FP16) in front of the CUDA provider Ultralytics requests.

For the segmentation model:

- **Speed per inference** (standalone): CPU 652 ms, CUDA 57 ms, TensorRT FP16 16.9 ms.
- **Accuracy:** per-frame detection counts match the `.pt` model on 44 of 50
  real frames (248 vs 252 in total).

The first load builds an engine (~8-10 minutes) into `rf_trained/ort_trt_cache/`.
ONNX Runtime keys that cache on the model path, so a different path string
(relative vs absolute) triggers a rebuild. Choose the provider with
`ONNX_PROVIDER` or `onnx_provider:` = `tensorrt` | `cuda` | `cpu`.

**Video is decoded on the hardware decoder.** `src/gst_capture.py` reads
MKV/MP4/MOV files with H.264/H.265, and RTSP streams, through `nvv4l2decoder`
and `nvvidconv`, the elements DeepStream is built on. Anything else falls back
to OpenCV.

- **Every frame is delivered,** matching OpenCV's decode to within 3/255 on
  average.
- **It frees a CPU core rather than adding FPS.** Handing a 1280x720 frame to
  Python costs 7.6 ms, against 6.8 ms for OpenCV's CPU decode.

Choose with `INGEST` or `ingest:` = `gstreamer` | `opencv`. The GStreamer
Python bindings come from the system:

    ln -sfn /usr/lib/python3/dist-packages/gi .venv/lib/python3.12/site-packages/gi

**The Analysis window** replaces the KDS panels in the dashboard's left sidebar
while no KDS feed is live. It shows:

- system stats: FPS, detect time, GPU, CPU, RAM, temperature, power, progress
- hotdogs by state
- sauces and items added
- detections per class
- an event feed

It is built by `src/feed_analysis.py`, which only reads what the pipeline
already computes:

| State | When |
|---|---|
| on counter | a hotdog is detected |
| wrapping | a wrapper or clamshell covers at least half of it for 0.4 s |
| wrapped | a `wrapped` detection over it, a `closed_reg_clamshell` there, or its clamshell stays within 40 px for 2 s after the hotdog disappears |
| outgoing | a hand touches the exit line (`config/exit_line.json`); the oldest wrapped hotdog goes out, one per touch |

Picks from the ingredient bins (`zone_type: bin` in `config/zones.json`) are
deliberately not counted. The thresholds live under `lifecycle:` in
`config/model.yaml`. A run writes `output/feed_analysis.json` every 30 s and at
exit, and `/api/analysis` serves the live snapshot.

**Recording the dashboard as a browser shows it.** `scripts/record_dashboard.py`:

1. starts a hidden 1920x1080 X display (Xorg with xrdp's `xrdpdev` driver; no
   sudo needed),
2. opens Firefox in kiosk mode on the dashboard,
3. captures the display with `ximagesrc` into the hardware encoder.

`bench/run_onnx_dashboard.sh` runs a whole recorded session detached and leaves
these files in `output/recordings/`:

| File | What it is |
|---|---|
| `dashboard_<label>.mp4` | the dashboard at wall-clock speed |
| `dashboard_<label>_camera_speed.mp4` | the same video retimed to camera speed, no re-encode |
| `detections_<label>.mp4` | the annotated feed |

### Measured performance

Segmentation model (`rf_trained/`, 16 classes), 1280x720, FP16, 40W power mode:

| Path | Latency | Throughput |
|---|---|---|
| YOLO inference only, CUDA + FP16 | ~36 ms/frame | **~28 FPS** |
| Full `Detector.detect()` as configured | ~59 ms/frame | ~17 FPS |
| CPU fallback | ~715 ms/frame | ~1.4 FPS |

**FP16 is now actually applied.** `half: true` existed in `config/tracker.yaml`
but was read only inside the DeepSORT branch of `detector.py`, so the
bytetrack/botsort path — the one in use — always ran FP32. `Detector` now takes
`half` and `imgsz`, wired from `config/model.yaml` (`HALF` / `IMGSZ` env
overrides). FP16 is auto-disabled when CUDA is absent, since torch has no fp16
CPU kernels for most ops. Worth ~1.17x on its own.

#### The tracker is the bottleneck, not the model

Inference is ~36 ms; the tracker adds ~24 ms on top, all of it CPU-bound:

| Tracker configuration | Throughput |
|---|---|
| `config/tracker.yaml` as shipped (botsort + `sparseOptFlow` GMC) | ~17 FPS |
| same, with `gmc_method: none` | ~28 FPS |
| `tracker_type: bytetrack` | ~28 FPS |

Note the mismatch: `config/model.yaml` sets `tracker_type: bytetrack`, but
`config/tracker.yaml` line 2 sets `tracker_type: botsort`, and **Ultralytics
honours the YAML** — so BoT-SORT with sparse-optical-flow global motion
compensation is what actually runs. That GMC pass is what emits the recurring
`not enough matching points` warning, and it costs ~64% of the frame budget.
The `wrap_station:` block in the same file documents itself as overriding
"the global ByteTrack params", which suggests botsort is a leftover.

This is left **unchanged** because switching trackers changes tracking
behaviour, not just speed. To take the ~28 FPS, set line 2 of
`config/tracker.yaml` to `tracker_type: bytetrack`, and re-validate accuracy on
real footage before shipping it.

`imgsz` is exposed but does little here (480 vs 640 was within noise) precisely
because the tracker, not inference, dominates.

#### About the 40W power mode

The board ships in mode 4 (40W). Mode 0 (`MAXN_SUPER`) removes the caps
(`MAX_FREQ -1` on CPU, GPU and EMC), but measured against this hardware the win
is narrower than it sounds:

- **GPU clock: no gain.** The 40W cap is `GPU MAX_FREQ 1173000000`, and
  `available_frequencies` tops out at exactly 1173 MHz — the GPU is already at
  its hardware ceiling and is running there now. MAXN_SUPER cannot clock it higher.
- **CPU: little or none.** Mode 4 nominally caps the A78 cores at 1497.6 MHz,
  but `scaling_max_freq` currently reads 1984000 (1.98 GHz = `cpuinfo_max_freq`),
  so the cores are already permitted to the hardware ceiling.
- **What MAXN_SUPER does buy** is the removal of the *total power budget* and
  the EMC (memory) cap. That matters for sustained load — it is throttling
  headroom, not a higher peak clock. Idle draw here is ~8.4 W against a 40 W
  budget, so the cap only binds under continuous inference.

```bash
sudo nvpmodel -m 0     # MAXN_SUPER: lifts the power budget + EMC cap
sudo jetson_clocks     # pin clocks to max (stops downclocking between frames)
```

Of the two, `jetson_clocks` is the more useful one here, since it prevents the
governor from dropping clocks between frames. Neither is required to run.

### Model formats (.pt / .onnx / .engine)

`src/inference/detector.py` selects the backend from the weights file extension. The
detection, tracking, zoning and state-machine logic is identical across all
three — only device and precision defaults change, so swapping formats needs no
code edit anywhere else:

| Extension | Runtime | Device here | Isolated | Full pipeline |
|---|---|---|---|---|
| `.pt` | PyTorch, FP16 | CUDA | 50.1 ms / 19.96 FPS | **13.1 FPS** |
| `.engine` | TensorRT, FP16 | CUDA | **41.7 ms / 24.01 FPS** | **16.2 FPS** |
| `.onnx` | ONNX Runtime, FP32 | **CPU only** | 694.0 ms / 1.44 FPS | 1.4 FPS |

Measured on the same 50 real frames from `camA`, through the same `Detector`,
changing nothing but the weights file. "Isolated" is `Detector.detect()` alone;
"full pipeline" is `main.py` including overlays, dashboard encode and flow.

Build them with:

```bash
uv run python scripts/export_onnx.py        # -> .onnx  (portable, opset 17)
uv run python scripts/export_tensorrt.py    # -> .engine (FP16, this device only)
```

Both scripts read `model_path`/`imgsz` from `config/model.yaml`. The TensorRT
build took **500 s** here and needs the system bindings visible to the venv:

```bash
ln -sfn /usr/lib/python3.12/dist-packages/tensorrt \
        .venv/lib/python3.12/site-packages/tensorrt
```

Symlink only the `tensorrt` package — putting all of `dist-packages` on the path
would let the system numpy/cv2 shadow the venv's.

Export ONNX with:

```bash
uv run python scripts/export_onnx.py              # reads config/model.yaml
uv run python scripts/export_onnx.py --dynamic    # variable input size
```

Then point the pipeline at it, with no other change:

```bash
MODEL_PATH='rf_trained/weights (1).onnx' ./run.sh videos/<clip>.mp4
```

#### TensorRT gives 1.20x, not the 2-3x usually quoted

The engine is faster, but far less dramatically than TensorRT's reputation
suggests, and it is worth understanding why before planning around it:

- The `.pt` baseline was **already FP16 on CUDA with cuDNN**. TensorRT's big
  wins are usually measured against an FP32 baseline; most of that gap was
  already closed.
- TensorRT only accelerates the **inference kernel**. The ~4 ms ByteTrack
  association, the mask/polygon postprocessing and the ~29 ms of overlay and
  dashboard work per frame are untouched — and those now dominate.

End to end that is 13.1 -> 16.2 FPS. Real, worth having, and still short of
20 FPS: at a 38 ms detect time with ~30 ms of fixed CPU work, the pipeline
cannot reach 20 FPS without also cutting that overhead.

**FP16 shifts detections slightly.** Across the 50-frame sample the engine and
the checkpoint agreed on **47/50 frames**, with 249 vs 252 total detections
(**-1.2%**). That is the expected cost of reduced precision, not a bug — but it
means the engine needs accuracy validation on real footage before it replaces
the `.pt` in production. `config/model.yaml` still points at the `.pt` for that
reason; switch with `MODEL_PATH` once you have validated it.

> **ONNX is slower on this box, by a lot.** Measured on 50 real frames through
> the same `Detector`: `.pt` on CUDA+FP16 is **51.1 ms/frame (19.6 FPS)**, the
> `.onnx` graph is **694.7 ms/frame (1.44 FPS)** — **13.6x slower** — with
> equivalent output (5.0 vs 5.1 detections/frame, so the export is numerically
> sound). The cause is not the format: the `onnxruntime` wheel installable for
> linux-aarch64 / cp312 exposes `CPUExecutionProvider` only. There is no CUDA or
> TensorRT execution provider, and `onnxruntime-gpu` publishes aarch64 wheels
> only for cp313/cp314. So the graph runs on the CPU.
>
> Export ONNX for **portability**, or as the intermediate step toward a
> TensorRT `.engine` — never to make this Jetson faster.

Two details the backend guard handles, so you do not have to:

- Passing `device=0` with an ONNX model makes Ultralytics try to `pip install
  onnxruntime-gpu` on *every load* (which fails — no cp312 aarch64 wheel), print
  a misleading "CUDA requested" warning, then fall back to CPU anyway. The
  detector asks for CPU directly instead.
- The graph is exported with `dynamic=False`, so its input is fixed at export
  size. A mismatched `imgsz` would fail inside ONNX Runtime; the detector pins it
  and warns instead.

### Known caveats on Orin

- **`sm_87` warning at import is expected and benign.** PyTorch's aarch64 CUDA
  wheels are built for `sm_80/90/100/110/120` and print
  *"No published PyTorch CUDA builds ... support this GPU"* because Orin is
  `sm_87`. The sm_80 cubins are binary-compatible with sm_87, and this was
  verified end to end: native elementwise/reduction kernels, cuBLAS matmul,
  cuDNN conv2d, **torchvision CUDA NMS**, and full YOLO `predict` + `bytetrack`
  all execute with correct numerics. NVIDIA has not yet published a JetPack 7
  (`jp7`) index on `pypi.jetson-ai-lab.io` — only `jp6` and `sbsa` exist — so
  there is currently no sm_87-native wheel to prefer over this one.
- **OCR runs on CPU.** The PyPI `onnxruntime` wheel exposes only
  `CPUExecutionProvider`; there is no CUDA EP. This is intentional and matches
  the RapidOCR note above (~0.65 s per KDS screen crop). The
  `GPU device discovery failed: .../card1/device/vendor` line it logs on Tegra
  is cosmetic.
- **TensorRT** (10.16) is installed system-wide. Exporting the model to
  `.engine` is the next step for real gains over the ~28 FPS PyTorch path.

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
- [src/inference/detector.py](src/inference/detector.py) runs YOLO inference with `model.track(...)` using the configured tracker.
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
