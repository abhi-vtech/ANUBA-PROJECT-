#!/usr/bin/env python3
"""Redraw the outgoing exit tripwire (config/exit_line.json) in a browser.

    uv run python scripts/edit_exit_line.py        # then open the URL it prints

``edit_rois.py`` draws in an OpenCV window, which needs a display on the
Jetson. Over SSH from another machine there is none, so this serves the editor
as a web page instead, reachable over ZeroTier, the office LAN, or a VS Code
forwarded port.

In the page
  click, click     place p1, then p2
  drag a handle    move an endpoint
  frame controls   step the background; the chips jump to useful moments
  Test             replay every hand detection in output/yolo_detections.log
                   through the detector's own crossing test, for the saved
                   line and the new one side by side
  Save             back up the current file, then write the new line
  Done             stop the editor

Saving needs the token in the printed URL, so someone who only finds the port
cannot change the line. The editor also stops by itself after --idle-min
minutes without a request.

Points are stored normalised 0..1 like the rest of config/, so a line drawn
here is correct whether the pipeline runs at 1280x720 or 1920x1080.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import secrets
import shutil
import statistics
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.exit_detector import _bbox_intersects_line  # noqa: E402  the pipeline's own test

W, H = 1280, 720          # the pipeline's processing size; the detection log uses it
SOURCE_FPS = 29.56        # measured average of the camA recording (it is variable-rate)
EPISODE_GAP = 15          # touches closer than this many frames count as one
MIN_LINE_PX = 20          # reject lines shorter than this at 1280x720

DEFAULT_VIDEO = ROOT / "Wienerschnitzel_Sacramento_CA_95818__camA__2026_09_06_11_to_12_PDT.mkv"
DEFAULT_RECORDING = ROOT / "output" / "recordings" / "detections_camA_pt.mp4"
DEFAULT_CONFIG = ROOT / "config" / "exit_line.json"
DEFAULT_LOG = ROOT / "output" / "yolo_detections.log"

# Backgrounds worth starting from, found in the 2026-09-10 check. The "reach"
# frames come from the annotated recording just after a wrapped order stopped
# being detected, when a hand reached up over the bins towards the pass shelf.
SUGGESTED = [
    {"label": "Empty station", "frame": 54000, "src": "raw"},
    {"label": "Reach over bins ~08:56", "frame": 15870, "src": "rec"},
    {"label": "Reach over bins ~08:57", "frame": 15877, "src": "rec"},
    {"label": "Reach over bins ~09:52", "frame": 17525, "src": "rec"},
]


def _mmss(frame: int) -> str:
    t = frame / SOURCE_FPS
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def _point(value) -> tuple[float, float]:
    if not (isinstance(value, (list, tuple)) and len(value) == 2):
        raise ValueError("each point must be [x, y]")
    x, y = float(value[0]), float(value[1])
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError("points must lie inside the frame")
    return x, y


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


class FrameSource:
    """One video file, read a single frame at a time and served as JPEG."""

    def __init__(self, path: Path):
        self.path = path
        self.count = 0
        self._lock = threading.Lock()
        self._cap = None
        if path.exists():
            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                self._cap = cap
                self.count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def available(self) -> bool:
        return self._cap is not None

    def jpeg(self, n: int) -> bytes | None:
        with self._lock:  # one decoder, several request threads
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(n, max(self.count - 1, 0))))
            ok, frame = self._cap.read()
        if not ok:
            return None
        ok, buf = cv2.imencode(".jpg", cv2.resize(frame, (W, H)), [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buf.tobytes() if ok else None


class DetectionReplay:
    """Hands and wrapped orders from the last run's per-frame detection log."""

    _FRAME = re.compile(r"^Frame (\d+):")
    _DET = re.compile(r"^\s+(hand|wrapped) \([\d.]+\) \[ID:(-?\d+)\] \((-?\d+), (-?\d+), (-?\d+), (-?\d+)\)")

    def __init__(self, path: Path):
        self.path = path
        self.frames = 0
        self.modified = None
        self.hands: list[tuple[int, tuple[int, int, int, int]]] = []
        self.wrapped_last: list[list[float]] = []
        if path.exists():
            self._load()

    def _load(self) -> None:
        self.modified = dt.datetime.fromtimestamp(self.path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        tracks: dict[int, list] = {}
        frame = None
        with open(self.path, errors="ignore") as fh:
            for line in fh:
                m = self._FRAME.match(line)
                if m:
                    frame = int(m.group(1))
                    self.frames += 1
                    continue
                d = self._DET.match(line)
                if d is None or frame is None:
                    continue
                box = tuple(int(d.group(i)) for i in range(3, 7))
                if d.group(1) == "hand":
                    self.hands.append((frame, box))
                else:
                    track = tracks.setdefault(int(d.group(2)), [0, None])
                    track[0] += 1
                    track[1] = [round((box[0] + box[2]) / 2 / W, 4), round((box[1] + box[3]) / 2 / H, 4)]
        self.wrapped_last = [t[1] for t in tracks.values() if t[0] >= 5]

    def test(self, p1: tuple[float, float], p2: tuple[float, float]) -> dict:
        a = (int(p1[0] * W), int(p1[1] * H))
        b = (int(p2[0] * W), int(p2[1] * H))
        touched, last = [], -1
        for frame, box in self.hands:
            if frame != last and _bbox_intersects_line(box, a, b):
                touched.append(frame)
                last = frame
        episodes: list[list[int]] = []
        for f in touched:
            if episodes and f - episodes[-1][1] <= EPISODE_GAP:
                episodes[-1][1] = f
            else:
                episodes.append([f, f])
        longest = sorted(episodes, key=lambda e: e[1] - e[0], reverse=True)[:12]
        return {
            "touch_frames": len(touched),
            "episodes": len(episodes),
            "median_len": statistics.median([e[1] - e[0] + 1 for e in episodes]) if episodes else 0,
            "longest": [{"start": s, "end": e, "frames": e - s + 1, "time": _mmss(s)} for s, e in sorted(longest)],
        }


class Editor:
    def __init__(self, args):
        self.config_path = Path(args.config)
        self.token = secrets.token_urlsafe(9)
        self.sources = {"raw": FrameSource(Path(args.video)), "rec": FrameSource(Path(args.recording))}
        self.replay = DetectionReplay(Path(args.log))
        self.log_path = ROOT / "bench" / "exit_line_editor.log"
        self.last_request = time.time()
        self.server: ThreadingHTTPServer | None = None

    def touch(self) -> None:
        self.last_request = time.time()

    def read_config(self) -> dict:
        try:
            return json.loads(self.config_path.read_text())
        except (OSError, ValueError):
            return {"p1": [0.8, 0.2], "p2": [0.8, 0.8], "line_name": "Exit_Tripwire_Line", "polygon": None}

    def state(self) -> dict:
        return {
            "config": self.read_config(),
            "config_path": _rel(self.config_path),
            "size": [W, H],
            "sources": {k: {"available": s.available, "frames": s.count, "name": s.path.name}
                        for k, s in self.sources.items()},
            "suggested": [c for c in SUGGESTED if self.sources[c["src"]].available],
            "log": {"available": bool(self.replay.hands), "frames": self.replay.frames,
                    "modified": self.replay.modified},
            "wrapped_last": self.replay.wrapped_last,
        }

    def save(self, p1: tuple[float, float], p2: tuple[float, float]) -> dict:
        if ((p1[0] - p2[0]) * W) ** 2 + ((p1[1] - p2[1]) * H) ** 2 < MIN_LINE_PX ** 2:
            raise ValueError(f"the two points are too close together (under {MIN_LINE_PX} px at 1280x720)")
        current = self.read_config()
        backup = None
        if self.config_path.exists():
            backup = self.config_path.with_name(f"{self.config_path.name}.bak.{dt.datetime.now():%Y%m%d_%H%M%S}")
            shutil.copyfile(self.config_path, backup)
        new = {
            "p1": [round(p1[0], 6), round(p1[1], 6)],
            "p2": [round(p2[0], 6), round(p2[1], 6)],
            "line_name": current.get("line_name", "Exit_Tripwire_Line"),
            "polygon": current.get("polygon"),
        }
        tmp = self.config_path.with_name(self.config_path.name + ".tmp")
        tmp.write_text(json.dumps(new, indent=2) + "\n")
        tmp.replace(self.config_path)  # atomic: a reader never sees a half-written file
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().isoformat(timespec="seconds")
        line = f"[{stamp}] saved {_rel(self.config_path)} p1={new['p1']} p2={new['p2']} backup={backup.name if backup else None}"
        with open(self.log_path, "a") as fh:
            fh.write(line + "\n")
        print(line, flush=True)
        return {"config": new, "backup": backup.name if backup else None}

    def watchdog(self, idle_min: float) -> None:
        while True:
            time.sleep(15)
            if time.time() - self.last_request > idle_min * 60:
                print(f"no requests for {idle_min:g} min, stopping", flush=True)
                self.server.shutdown()
                return


class Handler(BaseHTTPRequestHandler):
    server_version = "ExitLineEditor/1.0"

    def log_message(self, *_):  # keep the terminal for the URL and save events
        pass

    @property
    def editor(self) -> Editor:
        return self.server.editor  # type: ignore[attr-defined]

    def _reply(self, code: int, body, ctype: str = "application/json") -> None:
        if not isinstance(body, bytes):
            body = (json.dumps(body) if ctype == "application/json" else body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.editor.touch()
        url = urlparse(self.path)
        if url.path == "/":
            return self._reply(200, PAGE, "text/html; charset=utf-8")
        if url.path == "/api/state":
            return self._reply(200, self.editor.state())
        if url.path == "/frame":
            q = parse_qs(url.query)
            source = self.editor.sources.get(q.get("src", ["raw"])[0])
            try:
                n = int(q.get("n", ["0"])[0])
            except ValueError:
                return self._reply(400, {"error": "frame number must be a whole number"})
            if source is None or not source.available:
                return self._reply(404, {"error": "that video source is not available"})
            jpg = source.jpeg(n)
            if jpg is None:
                return self._reply(404, {"error": f"frame {n} could not be read"})
            return self._reply(200, jpg, "image/jpeg")
        return self._reply(404, {"error": "not found"})

    def do_POST(self):
        self.editor.touch()
        url = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._reply(400, {"error": "request body must be JSON"})
        if url.path in ("/api/save", "/api/quit") and not secrets.compare_digest(
                str(data.get("token", "")), self.editor.token):
            return self._reply(403, {"error": "Missing or wrong token. Open the exact URL the script printed."})
        try:
            if url.path == "/api/test":
                if not self.editor.replay.hands:
                    return self._reply(409, {"error": "no detection log to test against"})
                return self._reply(200, self.editor.replay.test(_point(data.get("p1")), _point(data.get("p2"))))
            if url.path == "/api/save":
                return self._reply(200, self.editor.save(_point(data.get("p1")), _point(data.get("p2"))))
            if url.path == "/api/quit":
                self._reply(200, {"stopping": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return None
        except (ValueError, TypeError) as exc:
            return self._reply(400, {"error": str(exc)})
        return self._reply(404, {"error": "not found"})


def _urls(port: int, token: str) -> list[tuple[str, str]]:
    urls = []
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show", "scope", "global"],
                             capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            parts = line.split()
            name, addr = parts[1], parts[3].split("/")[0]
            if not name.startswith(("docker", "l4tbr", "br-", "veth")):
                urls.append((name, f"http://{addr}:{port}/?t={token}"))
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    urls.append(("VS Code forwarded port", f"http://localhost:{port}/?t={token}"))
    return urls


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exit Line Editor</title>
<style>
:root{--bg:#15181b;--panel:#1e2226;--line:#2c3237;--ink:#e8ecee;--muted:#9aa4aa;--saved:#ffd60a;--new:#22d3ee;--bad:#f87171;--good:#4ade80}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:12px 18px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
header h1{font-size:16px;margin:0;font-weight:600}
header span{color:var(--muted);font-size:13px}
main{display:grid;grid-template-columns:minmax(0,1fr) 350px;gap:14px;padding:14px 18px}
@media (max-width:1100px){main{grid-template-columns:1fr}}
#stage{background:#000;border:1px solid var(--line);border-radius:4px;overflow:hidden}
canvas{display:block;width:100%;height:auto;cursor:crosshair;touch-action:none}
.bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:10px 0}
button,select,input{font:inherit;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:5px 10px}
button{cursor:pointer}
button:hover{border-color:#4a535a}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--new);outline-offset:1px}
button.primary{background:#0e7490;border-color:#0e7490}
button.chip{font-size:12.5px;padding:3px 10px;border-radius:999px}
input[type=number]{width:96px}
aside section{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:12px 14px;margin-bottom:12px}
aside h2{font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 8px;font-weight:600}
.kv{display:grid;grid-template-columns:auto 1fr;gap:6px 10px;margin:0}
.kv dt{color:var(--muted)} .kv dd{margin:0;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;font-variant-numeric:tabular-nums}
.sw{display:inline-block;width:16px;height:3px;vertical-align:middle;margin-right:6px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;margin-top:6px}
th,td{padding:5px 6px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:500;font-size:12px}
#msg{min-height:20px;color:var(--muted)} #msg.bad{color:var(--bad)} #msg.good{color:var(--good)}
.eps{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}
.hint{color:var(--muted);font-size:13px;margin:0}
</style></head><body>
<header><h1>Exit line editor</h1><span id="sub">loading…</span></header>
<main>
 <div>
  <div id="stage"><canvas id="cv" width="1280" height="720" aria-label="Camera frame. Click twice to place the exit line."></canvas></div>
  <div class="bar">
   <button data-step="-300">−10 s</button><button data-step="-30">−1 s</button>
   <button data-step="-1">−1 frame</button><button data-step="1">+1 frame</button>
   <button data-step="30">+1 s</button><button data-step="300">+10 s</button>
   <label>frame <input id="fn" type="number" min="0" step="1"></label><button id="go">Go</button>
   <select id="src" aria-label="Background"><option value="raw">Camera video</option><option value="rec">Annotated recording</option></select>
   <label><input id="wr" type="checkbox" checked> where wrapped orders vanished</label>
  </div>
  <div class="bar" id="chips"></div>
  <p class="hint">Click once for <b>p1</b> and again for <b>p2</b>, then drag either handle to adjust. The dashed yellow line is the one saved now; red dots mark where wrapped orders were last detected.</p>
 </div>
 <aside>
  <section><h2>Line</h2>
   <dl class="kv">
    <dt><span class="sw" style="background:var(--saved)"></span>saved</dt><dd id="kSaved">–</dd>
    <dt><span class="sw" style="background:var(--new)"></span>new</dt><dd id="kNew">click the frame for p1</dd>
   </dl>
   <div class="bar"><button id="clear">Start over</button><button id="fromSaved">Edit saved line</button></div>
   <div class="bar"><button id="test">Test on last run</button><button id="save" class="primary">Save</button></div>
   <div id="msg" role="status"></div>
  </section>
  <section><h2>Test against the recorded hour</h2>
   <p class="hint" id="logInfo"></p>
   <table id="tbl" hidden><thead><tr><th></th><th>saved</th><th>new</th></tr></thead><tbody></tbody></table>
   <div class="eps" id="eps"></div>
  </section>
  <section><h2>When you're finished</h2>
   <p class="hint">Save first, then stop the editor so the port closes.</p>
   <div class="bar"><button id="done">Done – stop editor</button></div>
  </section>
 </aside>
</main>
<script>
const T = new URLSearchParams(location.search).get('t') || '';
const $ = id => document.getElementById(id);
const cv = $('cv'), ctx = cv.getContext('2d'), img = new Image();
let S = null, src = 'raw', frame = 0, p1 = null, p2 = null, drag = null;

const fmt = p => `(${p[0].toFixed(4)}, ${p[1].toFixed(4)}) = ${Math.round(p[0] * 1280)}, ${Math.round(p[1] * 720)} px`;
function msg(text, cls) { const m = $('msg'); m.textContent = text; m.className = cls || ''; }

async function api(path, body) {
  const opts = body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)};
  const r = await fetch(path, opts);
  const j = await r.json().catch(() => ({error: `HTTP ${r.status}`}));
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

function setFrame(n, s) {
  if (s && S.sources[s] && S.sources[s].available) { src = s; $('src').value = s; }
  const max = Math.max(0, S.sources[src].frames - 1);
  frame = Math.min(max, Math.max(0, Math.round(n)));
  $('fn').value = frame;
  img.src = `/frame?src=${src}&n=${frame}`;
}
img.onload = draw;
img.onerror = () => msg('That frame could not be read from this video.', 'bad');

const P = p => [p[0] * cv.width, p[1] * cv.height];
function seg(a, b, color, dash, width) {
  const [x1, y1] = P(a), [x2, y2] = P(b);
  ctx.save(); ctx.setLineDash(dash); ctx.lineCap = 'round';
  for (const [c, w] of [['rgba(0,0,0,.75)', width + 4], [color, width]]) {
    ctx.strokeStyle = c; ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }
  ctx.restore();
}
function handle(p, label) {
  const [x, y] = P(p);
  ctx.beginPath(); ctx.arc(x, y, 9, 0, 2 * Math.PI); ctx.fillStyle = '#22d3ee'; ctx.fill();
  ctx.lineWidth = 2; ctx.strokeStyle = '#000'; ctx.stroke();
  ctx.font = '600 17px system-ui, sans-serif'; ctx.lineWidth = 4;
  ctx.strokeText(label, x + 12, y - 10); ctx.fillStyle = '#fff'; ctx.fillText(label, x + 12, y - 10);
}
function draw() {
  if (img.naturalWidth) ctx.drawImage(img, 0, 0, cv.width, cv.height);
  else { ctx.fillStyle = '#111'; ctx.fillRect(0, 0, cv.width, cv.height); }
  if (S && $('wr').checked) for (const q of S.wrapped_last) {
    const [x, y] = P(q); ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(255,59,48,.9)'; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = '#000'; ctx.stroke();
  }
  if (S) seg(S.config.p1, S.config.p2, '#ffd60a', [14, 9], 3);
  if (p1 && p2) seg(p1, p2, '#22d3ee', [], 4);
  if (p1) handle(p1, 'p1');
  if (p2) handle(p2, 'p2');
  $('kSaved').innerHTML = S ? `p1 ${fmt(S.config.p1)}<br>p2 ${fmt(S.config.p2)}` : '–';
  $('kNew').innerHTML = !p1 ? 'click the frame for p1' : `p1 ${fmt(p1)}<br>` + (p2 ? `p2 ${fmt(p2)}` : 'click again for p2');
}

function at(e) {
  const r = cv.getBoundingClientRect();
  return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))];
}
function near(p, q) {
  if (!p) return false;
  const r = cv.getBoundingClientRect();
  return Math.hypot((p[0] - q[0]) * r.width, (p[1] - q[1]) * r.height) <= 14;
}
cv.addEventListener('pointerdown', e => {
  const q = at(e);
  if (near(p1, q)) drag = 'p1';
  else if (near(p2, q)) drag = 'p2';
  else if (!p1) p1 = q;
  else if (!p2) p2 = q;
  else { msg('Drag a handle to adjust, or press "Start over" to redraw.'); return; }
  if (drag) cv.setPointerCapture(e.pointerId);
  draw();
});
cv.addEventListener('pointermove', e => { if (!drag) return; const q = at(e); if (drag === 'p1') p1 = q; else p2 = q; draw(); });
cv.addEventListener('pointerup', () => { drag = null; });

document.querySelectorAll('[data-step]').forEach(b => { b.onclick = () => setFrame(frame + Number(b.dataset.step)); });
$('go').onclick = () => setFrame(Number($('fn').value));
$('fn').onkeydown = e => { if (e.key === 'Enter') setFrame(Number($('fn').value)); };
$('src').onchange = () => setFrame(frame, $('src').value);
$('wr').onchange = draw;
$('clear').onclick = () => { p1 = p2 = null; draw(); msg(''); };
$('fromSaved').onclick = () => { p1 = [...S.config.p1]; p2 = [...S.config.p2]; draw(); msg('Editing a copy of the saved line.'); };

function showTest(a, b) {
  const rows = [['Frames a hand touches it', 'touch_frames'], ['Separate touches', 'episodes'], ['Median touch length (frames)', 'median_len']];
  $('tbl').tBodies[0].innerHTML = rows.map(([label, k]) =>
    `<tr><td>${label}</td><td>${a[k].toLocaleString()}</td><td>${b[k].toLocaleString()}</td></tr>`).join('');
  $('tbl').hidden = false;
  const box = $('eps'); box.innerHTML = '';
  if (!b.longest.length) { box.textContent = 'No hand touches the new line in the recorded hour.'; return; }
  const note = document.createElement('p'); note.className = 'hint'; note.style.width = '100%';
  note.textContent = 'Longest touches of the new line. Click one to see that moment:';
  box.append(note);
  for (const ep of b.longest) {
    const c = document.createElement('button'); c.className = 'chip';
    c.textContent = `~${ep.time} · ${ep.frames} frames`;
    c.onclick = () => setFrame(ep.start + Math.floor((ep.end - ep.start) / 2), S.sources.rec.available ? 'rec' : 'raw');
    box.append(c);
  }
}
$('test').onclick = async () => {
  if (!(p1 && p2)) return msg('Place both points first.', 'bad');
  if (!S.log.available) return msg('There is no detection log to test against.', 'bad');
  msg('Replaying hand detections…');
  try {
    const [a, b] = await Promise.all([api('/api/test', {p1: S.config.p1, p2: S.config.p2}), api('/api/test', {p1, p2})]);
    showTest(a, b); msg('Tested against the recorded hour.', 'good');
  } catch (err) { msg(err.message, 'bad'); }
};
$('save').onclick = async () => {
  if (!(p1 && p2)) return msg('Place both points first.', 'bad');
  if (!T) return msg('This page was opened without its token. Use the exact URL the script printed.', 'bad');
  try {
    const r = await api('/api/save', {token: T, p1, p2});
    S.config = r.config; draw();
    msg(`Saved to ${S.config_path}` + (r.backup ? `. The previous line is kept as ${r.backup}.` : '.'), 'good');
  } catch (err) { msg(err.message, 'bad'); }
};
$('done').onclick = async () => {
  try {
    await api('/api/quit', {token: T});
    document.body.innerHTML = '<p style="padding:24px">The editor has stopped. You can close this tab.</p>';
  } catch (err) { msg(err.message, 'bad'); }
};

(async () => {
  S = await api('/api/state');
  $('sub').textContent = `${S.config_path} · points are saved normalised, shown here at 1280×720`;
  $('src').querySelector('[value=rec]').disabled = !S.sources.rec.available;
  $('src').querySelector('[value=raw]').disabled = !S.sources.raw.available;
  if (!S.sources.raw.available) src = 'rec';
  $('logInfo').textContent = S.log.available
    ? `Replays ${S.log.frames.toLocaleString()} frames of hand detections from the run logged ${S.log.modified}, using the detector's own crossing test.`
    : 'No output/yolo_detections.log was found, so testing is unavailable.';
  for (const c of S.suggested) {
    const b = document.createElement('button'); b.className = 'chip'; b.textContent = c.label;
    b.onclick = () => setFrame(c.frame, c.src); $('chips').append(b);
  }
  const first = S.suggested[0];
  setFrame(first ? first.frame : 0, first ? first.src : src);
  draw();
  window.addEventListener('resize', draw);
})().catch(err => msg('Could not load the editor: ' + err.message, 'bad'));
</script>
</body></html>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0", help="address to listen on (default: all interfaces)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--video", default=str(DEFAULT_VIDEO), help="camera video to draw on")
    ap.add_argument("--recording", default=str(DEFAULT_RECORDING), help="annotated recording, same frame numbers as the log")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--log", default=str(DEFAULT_LOG), help="per-frame detection log used by Test")
    ap.add_argument("--idle-min", type=float, default=90, help="stop after this many minutes without a request")
    args = ap.parse_args(argv)

    print("reading the detection log ...", flush=True)
    editor = Editor(args)
    if not any(s.available for s in editor.sources.values()):
        print(f"error: no readable video at {args.video} or {args.recording}", file=sys.stderr)
        return 1
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        print(f"error: cannot listen on port {args.port} ({exc}); pass --port with a free one", file=sys.stderr)
        return 1
    server.daemon_threads = True
    server.editor = editor  # type: ignore[attr-defined]
    editor.server = server
    threading.Thread(target=editor.watchdog, args=(args.idle_min,), daemon=True).start()

    print(f"Exit line editor for {_rel(editor.config_path)}")
    print(f"  hand detections: {len(editor.replay.hands):,} across {editor.replay.frames:,} frames")
    print("Open one of these in your browser:")
    for name, url in _urls(args.port, editor.token):
        print(f"  {name:<24} {url}")
    print(f"It stops after {args.idle_min:g} min without requests, or when you press Done.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    print("editor stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
