#!/usr/bin/env bash
# Launcher for the oad-pipeline binary. The binary runs fine on its own; this
# script just sets sensible defaults, points at bundled libraries, and ensures
# output/logs directories exist.
#
# Location: External/setup/run.sh
# Layout assumption:
#   External/
#   ├── bin/oad-pipeline      ← the binary
#   ├── libraries/            ← bundled .dylib/.so
#   ├── rf_trained/           ← model
#   ├── config/, templates/   ← editable config
#   ├── input/, output/, logs/← runtime data
#
# Override defaults by setting env vars before calling run.sh, e.g.:
#   VIDEO_SOURCE=rtsp://camera.local/stream LOG_LEVEL=INFO ./setup/run.sh

set -euo pipefail

# Resolve External/ from this script's location (External/setup/run.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Pick a binary name that matches the platform.
BIN_NAME="oad-pipeline"
if [[ "${OSTYPE:-}" == "msys" || "${OSTYPE:-}" == "cygwin" || "${OSTYPE:-}" == "win32" ]]; then
    BIN_NAME="oad-pipeline.exe"
fi

BIN_PATH="${EXTERNAL_DIR}/bin/${BIN_NAME}"
if [[ ! -x "${BIN_PATH}" && ! -f "${BIN_PATH}" ]]; then
    echo "Binary ${BIN_PATH} not found."
    echo "Re-run build/build_nuitka.sh (from Internal/), or copy oad-pipeline into External/bin/."
    exit 1
fi

# Point the dynamic linker at our bundled libraries.
# macOS uses DYLD_LIBRARY_PATH, Linux uses LD_LIBRARY_PATH.
# Prepend (not overwrite) so user-set values take precedence.
if [[ -d "${EXTERNAL_DIR}/libraries" ]]; then
    case "$(uname -s)" in
        Darwin)
            export DYLD_LIBRARY_PATH="${EXTERNAL_DIR}/libraries:${DYLD_LIBRARY_PATH:-}"
            ;;
        Linux)
            export LD_LIBRARY_PATH="${EXTERNAL_DIR}/libraries:${LD_LIBRARY_PATH:-}"
            ;;
    esac
fi

# Ensure writable directories exist.
mkdir -p "${EXTERNAL_DIR}/output" "${EXTERNAL_DIR}/logs"

# Sensible defaults if caller hasn't set them.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export LOG_LEVEL="${LOG_LEVEL:-WARNING}"
export LOG_METRICS_INTERVAL="${LOG_METRICS_INTERVAL:-5}"

# MODEL_PATH is intentionally NOT set here. The binary reads config/model.yaml
# and resolves relative paths via paths.resource(). Setting MODEL_PATH as an
# env var would override the config file, making customer edits to model.yaml
# silently ignored. Only set MODEL_PATH manually for command-line overrides:
#   MODEL_PATH=rf_trained/my_model.pt ./setup/run.sh

# VIDEO_SOURCE is intentionally NOT set here. The binary reads config/model.yaml
# for the source value. Setting it as an env var would override the config file,
# making customer edits to model.yaml silently ignored. Only set VIDEO_SOURCE
# manually when you want a command-line override, e.g.:
#   VIDEO_SOURCE=rtsp://... ./setup/run.sh

echo "Starting oad-pipeline..."
echo "  EXTERNAL_DIR      = ${EXTERNAL_DIR}"
echo "  VIDEO_SOURCE      = ${VIDEO_SOURCE:-<from config/model.yaml>}"
echo "  MODEL_PATH        = ${MODEL_PATH:-<from config/model.yaml>}"
echo "  LOG_LEVEL         = ${LOG_LEVEL}"
echo "  Dashboard         = http://localhost:8000"
echo ""
echo "  Tip: to override config, set env vars before ./setup/run.sh"
echo "       VIDEO_SOURCE=input/my_video.mp4 ./setup/run.sh"
echo ""

cd "${EXTERNAL_DIR}"
exec "${BIN_PATH}" "$@"
