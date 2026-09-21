#!/usr/bin/env python3
"""Parse a benchmark result directory into a metrics summary (JSON + text)."""
import json, re, sys, statistics as st
from pathlib import Path

def pct(v, p):
    if not v: return 0.0
    v = sorted(v); k = (len(v)-1)*p/100.0
    f = int(k); c = min(f+1, len(v)-1)
    return v[f] + (v[c]-v[f])*(k-f)

def parse_tegrastats(path):
    rows = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        r = {}
        m = re.search(r"RAM (\d+)/(\d+)MB", line)
        if not m: continue
        r["ram_used_mb"], r["ram_total_mb"] = int(m.group(1)), int(m.group(2))
        m = re.search(r"SWAP (\d+)/(\d+)MB", line)
        if m: r["swap_used_mb"] = int(m.group(1))
        m = re.search(r"CPU \[([^\]]+)\]", line)
        if m:
            cores = []
            for c in m.group(1).split(","):
                mm = re.match(r"(\d+)%@(\d+)", c.strip())
                if mm: cores.append((int(mm.group(1)), int(mm.group(2))))
                elif c.strip() == "off": cores.append((0, 0))
            if cores:
                r["cpu_util_avg"] = sum(u for u, _ in cores)/len(cores)
                r["cpu_util_max"] = max(u for u, _ in cores)
                r["cpu_freq_max"] = max(f for _, f in cores)
        m = re.search(r"GR3D_FREQ (\d+)%(?:@\[?(\d+))?", line)
        if m:
            r["gpu_util"] = int(m.group(1))
            if m.group(2): r["gpu_freq_mhz"] = int(m.group(2))
        for name, key in (("gpu", "t_gpu"), ("cpu", "t_cpu"), ("tj", "t_tj"), ("soc0", "t_soc0")):
            mm = re.search(rf"\b{name}@([\d.]+)C", line)
            if mm: r[key] = float(mm.group(1))
        for rail, key in (("VDD_IN", "p_in_mw"), ("VDD_CPU_GPU_CV", "p_cpugpu_mw"), ("VDD_SOC", "p_soc_mw")):
            mm = re.search(rf"{rail} (\d+)mW", line)
            if mm: r[key] = int(mm.group(1))
        rows.append(r)
    return rows

def parse_pipeline(path):
    ev = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        i = line.find('{"event": "metrics"')
        if i == -1: continue
        try: ev.append(json.loads(line[i:]))
        except Exception: pass
    return ev

def series(rows, key):
    return [r[key] for r in rows if key in r]

def summarize(vals, unit=""):
    if not vals: return None
    return {"min": round(min(vals),1), "mean": round(st.fmean(vals),1),
            "p50": round(pct(vals,50),1), "p95": round(pct(vals,95),1),
            "max": round(max(vals),1), "unit": unit, "n": len(vals)}

def main(outdir):
    d = Path(outdir)
    sysinfo = dict(l.split("=",1) for l in (d/"system.txt").read_text().splitlines() if "=" in l)
    tg = parse_tegrastats(d/"tegrastats.log")
    ev = parse_pipeline(d/"pipeline.log")

    # Steady state: drop the first 30 s (model load, CUDA context, warm-up).
    tg_ss = tg[30:] if len(tg) > 60 else tg
    ev_ss = [e for e in ev if e.get("elapsed_s",0) > 30] or ev

    inst_fps = []
    for a, b in zip(ev_ss, ev_ss[1:]):
        df = b["frame_count"]-a["frame_count"]; dt = b["elapsed_s"]-a["elapsed_s"]
        if dt > 0 and df >= 0: inst_fps.append(df/dt)

    det_counts = {}
    for e in ev_ss:
        for k, v in (e.get("detections") or {}).items():
            det_counts[k] = det_counts.get(k,0)+v
    frames_with = sum(1 for e in ev_ss if e.get("detections"))

    last = ev[-1] if ev else {}
    wall = int(sysinfo.get("wall_seconds", 0) or 0)
    vframes = int(sysinfo.get("video_frames", 0) or 0)
    vdur = float(sysinfo.get("video_duration_s", 0) or 0)
    done = last.get("frame_count", 0)

    out = {
        "system": sysinfo,
        "run": {
            "frames_processed": done,
            "frames_in_video": vframes,
            "completion_pct": round(100*done/vframes,2) if vframes else None,
            "wall_seconds": wall,
            "video_seconds": vdur,
            "realtime_factor": round(wall/vdur,3) if vdur and wall else None,
            "avg_fps_overall": round(done/wall,2) if wall else None,
            "exit_code": sysinfo.get("exit_code"),
        },
        "throughput": {
            "fps_instantaneous": summarize(inst_fps, "fps"),
            "loop_ms":     summarize([e["loop_ms"] for e in ev_ss if "loop_ms" in e], "ms"),
            "detect_ms":   summarize([e["detect_ms"] for e in ev_ss if "detect_ms" in e], "ms"),
            "flow_ms":     summarize([e["flow_ms"] for e in ev_ss if "flow_ms" in e], "ms"),
            "temporal_ms": summarize([e["temporal_ms"] for e in ev_ss if "temporal_ms" in e], "ms"),
        },
        "gpu":     {"util_pct": summarize(series(tg_ss,"gpu_util"),"%"),
                    "freq_mhz": summarize(series(tg_ss,"gpu_freq_mhz"),"MHz"),
                    "temp_c":   summarize(series(tg_ss,"t_gpu"),"C")},
        "cpu":     {"util_avg_pct": summarize(series(tg_ss,"cpu_util_avg"),"%"),
                    "util_max_pct": summarize(series(tg_ss,"cpu_util_max"),"%"),
                    "freq_max_mhz": summarize(series(tg_ss,"cpu_freq_max"),"MHz"),
                    "temp_c":       summarize(series(tg_ss,"t_cpu"),"C")},
        "memory":  {"ram_used_mb": summarize(series(tg_ss,"ram_used_mb"),"MB"),
                    "swap_used_mb": summarize(series(tg_ss,"swap_used_mb"),"MB"),
                    "ram_total_mb": tg[0].get("ram_total_mb") if tg else None},
        "power":   {"vdd_in_mw": summarize(series(tg_ss,"p_in_mw"),"mW"),
                    "cpu_gpu_cv_mw": summarize(series(tg_ss,"p_cpugpu_mw"),"mW"),
                    "soc_mw": summarize(series(tg_ss,"p_soc_mw"),"mW")},
        "thermal": {"tj_c": summarize(series(tg_ss,"t_tj"),"C")},
        "detections": {"total_by_class": dict(sorted(det_counts.items(), key=lambda x:-x[1])),
                       "sampled_windows": len(ev_ss),
                       "windows_with_detections": frames_with},
        "samples": {"tegrastats_rows": len(tg), "metric_events": len(ev)},
    }
    (d/"summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "bench/results/camA_full")
