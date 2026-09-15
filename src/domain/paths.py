"""Resolve filesystem paths correctly in dev and when frozen by Nuitka.

Dev layout (running from Internal/)::
    Internal/
    ├── src/paths.py        ← this file
    ├── config/             ← app config
    ├── templates/          ← dashboard HTML
    └── rf_trained/         ← model weights

Frozen layout (Nuitka --standalone, customer deliverable)::
    External/
    ├── bin/
    │   └── oad-pipeline    ← the binary (sys.executable)
    ├── libraries/          ← bundled .dylib/.so deps
    ├── config/             ← app config (sibling of bin/)
    ├── templates/          ← dashboard HTML
    ├── rf_trained/         ← model weights
    ├── input/              ← customer's video files
    ├── output/             ← reports land here
    └── logs/               ← runtime logs

``app_root()`` returns the directory that contains ``config/`` and
``templates/`` in both layouts. Dev: ``Internal/``. Frozen: ``External/``.
The binary itself sits one level below app_root, at ``External/bin/``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def app_root() -> Path:
    """Return the directory that contains ``config/`` and ``templates/``."""
    # Nuitka standalone: sys.frozen is NOT set, and sys.executable points to
    # the embedded python3 inside the .dist directory.  sys.argv[0] is the
    # actual binary path (e.g. External/bin/oad-pipeline).  We detect the
    # layout by checking whether the candidate directory contains both
    # config/ and templates/ (present in both dev and frozen layouts).
    candidate = Path(sys.argv[0]).resolve().parent.parent
    if (candidate / "config").is_dir() and (candidate / "templates").is_dir():
        return candidate
    # Dev: this file is at <repo>/Internal/src/domain/paths.py, so app_root is
    # three levels up.  It was two while the file lived at src/paths.py; after
    # the move that returned Internal/src, and every caller whose argv[0] sits
    # outside the repo (a scratch script, a probe) silently resolved config to
    # Internal/src/config and failed on a missing file.
    return Path(__file__).resolve().parents[2]


def resource(rel: str) -> str:
    """Resolve ``rel`` (e.g. ``"config/model.yaml"``) against the app root.

    Returns a string for ergonomic use with APIs that take paths-as-strings
    (Jinja2Templates, MockKDSClient, open(), ...).
    """
    return str(app_root() / rel)


def binary_dir() -> Path:
    """Return the directory containing the running binary (frozen only).

    Useful for locating bundled libraries (``External/libraries/``) that
    sit next to the binary, not next to the app root.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # In dev there is no binary; fall back to app_root for safety.
    return app_root()
