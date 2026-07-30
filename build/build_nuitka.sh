#!/usr/bin/env bash
# Compile src/main.py to a standalone binary with Nuitka, then assemble the
# customer-facing deliverable in <repo>/External/.
#
# Usage:
#   cd <repo>/Internal
#   ./build/build_nuitka.sh
#
# Requirements:
#   - Python 3.10+ in a working environment (uv sync has been run from Internal/)
#   - nuitka installed (pip install nuitka)
#   - C compiler toolchain (clang on macOS, gcc on Linux, MSVC on Windows)
#
# Output:
#   <repo>/External/                    ← ship this folder
#   <repo>/External/bin/oad-pipeline    ← the binary (platform-named)
#
# Notes:
#   - The binary is platform-specific. Build on each target OS.
#   - The model is NOT bundled inside the binary; it lives at External/rf_trained/.
#   - Bundled .dylib/.so files are split out into External/libraries/ and loaded
#     via DYLD_LIBRARY_PATH / LD_LIBRARY_PATH by the run.sh launcher.

set -euo pipefail

# Resolve repo root from this script's location, regardless of caller cwd.
# build_nuitka.sh lives at Internal/build/build_nuitka.sh
#   → SCRIPT_DIR = <repo>/Internal/build
#   → INTERNAL_DIR = <repo>/Internal
#   → REPO_ROOT = <repo>
#   → EXTERNAL_DIR = <repo>/External
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERNAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${INTERNAL_DIR}/.." && pwd)"
EXTERNAL_DIR="${REPO_ROOT}/External"
# Keep the Nuitka build work in a short path. macOS codesign gets a single
# command line with every bundled .dylib/.so path, and the default TMPDIR on
# macOS (/var/folders/...) is long enough to push that command line past
# ARG_MAX when torch's 5000+ files are included. We force /tmp explicitly.
NUITKA_WORK="/tmp/oad-nuitka-build"

cd "${INTERNAL_DIR}"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PYTHON_BIN="${PYTHON_BIN:-python}"
if command -v uv >/dev/null 2>&1; then
    PYTHON_BIN="uv run python"
fi

BIN_STEM="oad-pipeline"

# Per-platform binary filename. Nuitka appends .exe on Windows automatically,
# so we name the source "oad-pipeline" and let it land as e.g. oad-pipeline.exe.
case "$(uname -s)" in
    Darwin|Linux) BIN_NAME="${BIN_STEM}" ;;
    *)            BIN_NAME="${BIN_STEM}" ;;  # Windows handled by Nuitka
esac

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

echo "==> Checking environment"
${PYTHON_BIN} -c "import nuitka" 2>/dev/null || {
    echo "nuitka not importable. Install with: ${PYTHON_BIN} -m pip install nuitka"
    exit 1
}
${PYTHON_BIN} -c "import ultralytics, fastapi, uvicorn, cv2, yaml" 2>/dev/null || {
    echo "One or more runtime deps missing. Run: uv sync"
    exit 1
}

# Confirm the model is present. We don't bundle it inside the binary; we copy
# it to External/rf_trained/ so it can be swapped per deployment via MODEL_PATH.
MODEL_PATH="${MODEL_PATH:-rf_trained/yolo26s_trained.mlpackage}"
if [[ ! -e "${MODEL_PATH}" ]]; then
    echo "Model not found at ${MODEL_PATH}"
    echo "Set MODEL_PATH=/path/to/model.mlpackage to override."
    exit 1
fi

# ---------------------------------------------------------------------------
# Clean previous External/ (preserve gitignored build scratch)
# ---------------------------------------------------------------------------

echo "==> Cleaning previous External/ at ${EXTERNAL_DIR}"
rm -rf "${EXTERNAL_DIR}"
mkdir -p "${EXTERNAL_DIR}/bin" \
         "${EXTERNAL_DIR}/libraries" \
         "${EXTERNAL_DIR}/rf_trained" \
         "${EXTERNAL_DIR}/config" \
         "${EXTERNAL_DIR}/templates" \
         "${EXTERNAL_DIR}/input" \
         "${EXTERNAL_DIR}/output" \
         "${EXTERNAL_DIR}/logs" \
         "${EXTERNAL_DIR}/setup"

# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------

echo "==> Compiling with Nuitka (this can take 2-5 minutes the first time)"
mkdir -p "${NUITKA_WORK}"

# Stage a customer-facing config into the Nuitka work directory and bake THAT
# into the binary. The Internal/config/ stays untouched (dev), but the
# External/ config the customer sees is generated from it. Patches the dev-only
# video source path to point at the bundled sample clip.
CUSTOMER_CONFIG="${NUITKA_WORK}/customer_config"
mkdir -p "${CUSTOMER_CONFIG}"
# Copy only the canonical config files; skip dev backups like kds_mock_v*.json
# or zones_bck.json that the dev team accumulates locally.
for f in "${INTERNAL_DIR}"/config/*.{yaml,yml,json}; do
    [[ -e "${f}" ]] || continue
    base="$(basename "${f}")"
    case "${base}" in
        *_v*.json|*_bck.json|*_backup.json) continue ;;  # dev backups
    esac
    cp "${f}" "${CUSTOMER_CONFIG}/"
done
if command -v yq >/dev/null 2>&1; then
    yq -i '.source = "input/sample.mp4"' "${CUSTOMER_CONFIG}/model.yaml"
else
    sed -i.bak 's|^source:.*$|source: "input/sample.mp4"|' \
        "${CUSTOMER_CONFIG}/model.yaml" && \
        rm -f "${CUSTOMER_CONFIG}/model.yaml.bak"
fi

${PYTHON_BIN} -m nuitka \
    --standalone \
    --assume-yes-for-downloads \
    --output-filename="${BIN_NAME}" \
    --output-dir="${NUITKA_WORK}" \
    --include-package=ultralytics \
    --include-package=cv2 \
    --include-package=fastapi \
    --include-package=uvicorn \
    --include-package=starlette \
    --include-package=jinja2 \
    --include-package-data=jinja2 \
    --include-package=yaml \
    --include-package=lap \
    --include-package=deep_sort_realtime \
    --include-package=src \
    --include-data-dir="${CUSTOMER_CONFIG}=config" \
    --include-data-dir="${INTERNAL_DIR}/templates=templates" \
    --module-parameter=torch-disable-jit=yes \
    --remove-output \
    src/main.py

# Nuitka emits everything inside <output-dir>/main.dist/
NUITKA_DIST="${NUITKA_WORK}/main.dist"
if [[ ! -x "${NUITKA_DIST}/${BIN_NAME}" ]]; then
    echo "Nuitka did not produce ${NUITKA_DIST}/${BIN_NAME}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Assemble deliverable into External/
# ---------------------------------------------------------------------------

echo "==> Assembling External/ deliverable"

# 1. Copy the binary itself.
cp "${NUITKA_DIST}/${BIN_NAME}" "${EXTERNAL_DIR}/bin/${BIN_NAME}"
chmod +x "${EXTERNAL_DIR}/bin/${BIN_NAME}"

# 2. Split out bundled libraries (.dylib on macOS, .so on Linux) into External/libraries/.
#    This lets run.sh set DYLD_LIBRARY_PATH / LD_LIBRARY_PATH cleanly, and keeps
#    External/bin/ as "just the binary".
echo "    - splitting bundled libraries into External/libraries/"
shopt -s nullglob
for lib in "${NUITKA_DIST}"/lib*.dylib "${NUITKA_DIST}"/lib*.so*; do
    cp -P "${lib}" "${EXTERNAL_DIR}/libraries/"
done
# .so Python extension modules stay with the binary (they're loaded by Python's
# import machinery, not the dynamic linker). Copy everything else from main.dist
# that isn't a .dylib/.so library or the binary itself.
for item in "${NUITKA_DIST}"/*; do
    name="$(basename "${item}")"
    case "${name}" in
        "${BIN_NAME}"|*.dylib|*.so|*.so.*) continue ;;  # already handled
    esac
    cp -R "${item}" "${EXTERNAL_DIR}/bin/"
done

# 3. Copy the model to External/rf_trained/.
if [[ -d "${MODEL_PATH}" ]]; then
    cp -R "${MODEL_PATH}" "${EXTERNAL_DIR}/rf_trained/"
else
    cp "${MODEL_PATH}" "${EXTERNAL_DIR}/rf_trained/"
fi

# 4. Copy the staged customer config to External/config/. The baked-in copy
#    inside the binary uses the same patched file (see compile step above).
cp -R "${CUSTOMER_CONFIG}/." "${EXTERNAL_DIR}/config/"
cp -R "${INTERNAL_DIR}/templates/." "${EXTERNAL_DIR}/templates/"

# 5. Copy a sample input clip if available.
SAMPLE_CANDIDATES=(
    "${REPO_ROOT}/videos/clips/v5/seg_seg004.mp4"
    "${INTERNAL_DIR}/input/sample.mp4"
)
for candidate in "${SAMPLE_CANDIDATES[@]}"; do
    if [[ -f "${candidate}" ]]; then
        cp "${candidate}" "${EXTERNAL_DIR}/input/sample.mp4"
        break
    fi
done

# 6. Drop in the customer-facing readme and launcher.
cp "${SCRIPT_DIR}/customer_readme.md" "${EXTERNAL_DIR}/readme.md"
cp "${SCRIPT_DIR}/run.sh"            "${EXTERNAL_DIR}/setup/run.sh"
chmod +x "${EXTERNAL_DIR}/setup/run.sh"

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

echo ""
echo "==> Release ready at ${EXTERNAL_DIR}"
echo ""
echo "Layout:"
( cd "${EXTERNAL_DIR}" && find . -maxdepth 2 -mindepth 1 | sort | sed 's|^|  |' )
echo ""
echo "Size:"
du -sh "${EXTERNAL_DIR}" "${EXTERNAL_DIR}/bin" "${EXTERNAL_DIR}/libraries" \
       "${EXTERNAL_DIR}/rf_trained" 2>/dev/null | sed 's|^|  |'
echo ""
echo "To test locally:"
echo "  cd ${EXTERNAL_DIR}"
echo "  ./setup/run.sh"
echo ""
echo "Then open http://localhost:8000 in a browser."
