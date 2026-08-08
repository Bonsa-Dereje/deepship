#!/usr/bin/env python3
"""
offr dashboard — live black & white system monitor for the offr server.

Serves a single-page dashboard (static/index.html) and a JSON stats API
(/api/stats) the page polls every couple of seconds. Everything reported
is read directly off THIS machine:

    - CPU / RAM / swap / load average / uptime  -> psutil
    - Disk usage per mount                        -> psutil
    - Network throughput (live rate, not totals)  -> psutil
    - GPU (NVIDIA GeForce 940M)                   -> `nvidia-smi`
    - Ollama models + what's currently loaded     -> local Ollama API (11434)
    - Plex Media Server                            -> process check + local Plex API (32400)

Run this directly ON the offr box (same machine Ollama/Plex run on) --
it talks to 127.0.0.1, not the Tailscale hostname. Open the dashboard
itself from any device on your Tailscale network by browsing to
http://<offr-tailscale-ip-or-name>:8420 (or set DASHBOARD_PORT).

Requirements:
    pip install flask psutil requests

Usage:
    python3 server.py                    # http://0.0.0.0:8420
    DASHBOARD_PORT=9000 python3 server.py

Optional env vars:
    OLLAMA_LOCAL_URL   default http://127.0.0.1:11434
    PLEX_LOCAL_URL     default http://127.0.0.1:32400
    PLEX_TOKEN         optional -- adds active-stream count if set
                        (Settings > General > Network in Plex, or see
                        https://support.plex.tv/articles/204059436)
"""
import os
import shutil
import subprocess
import time

import psutil
import requests
from flask import Flask, jsonify, send_from_directory

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, "static")

OLLAMA_URL = os.environ.get("OLLAMA_LOCAL_URL", "http://127.0.0.1:11434").rstrip("/")
PLEX_URL = os.environ.get("PLEX_LOCAL_URL", "http://127.0.0.1:32400").rstrip("/")
PLEX_TOKEN = os.environ.get("PLEX_TOKEN", "")
PORT = int(os.environ.get("DASHBOARD_PORT", "8420"))

app = Flask(__name__, static_folder=None)

_net_prev = {"t": time.time(), "sent": 0, "recv": 0}


def _net_rates():
    """Bytes/sec since the last poll, not just lifetime totals."""
    global _net_prev
    now = time.time()
    counters = psutil.net_io_counters()
    dt = max(now - _net_prev["t"], 0.001)
    up = max(0.0, (counters.bytes_sent - _net_prev["sent"]) / dt)
    down = max(0.0, (counters.bytes_recv - _net_prev["recv"]) / dt)
    _net_prev = {"t": now, "sent": counters.bytes_sent, "recv": counters.bytes_recv}
    return {
        "up_bps": up, "down_bps": down,
        "total_sent_gb": round(counters.bytes_sent / 1e9, 2),
        "total_recv_gb": round(counters.bytes_recv / 1e9, 2),
    }


def _f(v, default=None):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _gpu_stats():
    """nvidia-smi output for the GeForce 940M. Older laptop GPUs sometimes
    don't support power/fan/clock queries -- those come back as
    '[Not Supported]', which just becomes null for the frontend to show
    as '—' rather than a crash."""
    if not shutil.which("nvidia-smi"):
        return {"available": False, "reason": "nvidia-smi not found on PATH"}
    try:
        fields = ("name,utilization.gpu,utilization.memory,memory.used,memory.total,"
                   "temperature.gpu,power.draw,fan.speed,clocks.sm")
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return {"available": False, "reason": out.stderr.strip()[:200]}
        parts = [p.strip() for p in out.stdout.strip().split(",")]
        parts = (parts + [""] * 9)[:9]
        name, util_gpu, util_mem, mem_used, mem_total, temp, power, fan, clock = parts
        return {
            "available": True,
            "name": name,
            "utilization_pct": _f(util_gpu, 0),
            "mem_utilization_pct": _f(util_mem, 0),
            "mem_used_mb": _f(mem_used, 0),
            "mem_total_mb": _f(mem_total, 0),
            "temp_c": _f(temp),
            "power_w": _f(power),
            "fan_pct": _f(fan),
            "sm_clock_mhz": _f(clock),
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)[:200]}


def _ollama_stats():
    result = {"reachable": False, "models": [], "running": []}
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).json()
        result["reachable"] = True
        result["models"] = [
            {
                "name": m.get("name"),
                "size_gb": round((m.get("size") or 0) / 1e9, 2),
                "param_size": (m.get("details") or {}).get("parameter_size"),
                "quant": (m.get("details") or {}).get("quantization_level"),
                "family": (m.get("details") or {}).get("family"),
            }
            for m in tags.get("models", [])
        ]
    except Exception as exc:
        result["error"] = str(exc)[:200]
        return result
    try:
        ps = requests.get(f"{OLLAMA_URL}/api/ps", timeout=3).json()
        result["running"] = [
            {
                "name": m.get("name"),
                "vram_gb": round((m.get("size_vram") or 0) / 1e9, 2),
                "param_size": (m.get("details") or {}).get("parameter_size"),
                "quant": (m.get("details") or {}).get("quantization_level"),
                "expires_at": m.get("expires_at"),
            }
            for m in ps.get("models", [])
        ]
    except Exception:
        pass  # /api/ps failing just means "nothing loaded right now" on some versions
    return result


def _xml_attr(txt, name):
    marker = f'{name}="'
    i = txt.find(marker)
    if i == -1:
        return None
    j = txt.find('"', i + len(marker))
    return txt[i + len(marker):j]


def _plex_stats():
    result = {"running": False, "reachable": False}
    for proc in psutil.process_iter(["name"]):
        try:
            pname = (proc.info["name"] or "").lower()
            if "plex media server" in pname or pname == "plexmediaserver":
                result["running"] = True
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    try:
        headers = {"X-Plex-Token": PLEX_TOKEN} if PLEX_TOKEN else {}
        r = requests.get(f"{PLEX_URL}/identity", headers=headers, timeout=3)
        if r.ok:
            result["reachable"] = True
            result["version"] = _xml_attr(r.text, "version")
            result["platform"] = _xml_attr(r.text, "platform")
        if PLEX_TOKEN:
            s = requests.get(f"{PLEX_URL}/status/sessions", headers=headers, timeout=3)
            if s.ok:
                result["active_streams"] = s.text.count("<Video ") + s.text.count("<Track ")
    except Exception as exc:
        result["error"] = str(exc)[:200]
    return result


@app.route("/api/stats")
def api_stats():
    cpu_per_core = psutil.cpu_percent(percpu=True)
    cpu_overall = round(sum(cpu_per_core) / len(cpu_per_core), 1) if cpu_per_core else 0.0
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    try:
        load1, load5, load15 = os.getloadavg()
    except (AttributeError, OSError):
        load1 = load5 = load15 = 0.0

    disks = []
    for part in psutil.disk_partitions(all=False):
        if part.fstype in ("", "squashfs", "tmpfs", "devtmpfs"):
            continue
        try:
            u = psutil.disk_usage(part.mountpoint)
        except (PermissionError, FileNotFoundError):
            continue
        disks.append({
            "mount": part.mountpoint, "device": part.device,
            "used_gb": round(u.used / 1e9, 1), "total_gb": round(u.total / 1e9, 1),
            "pct": u.percent,
        })

    return jsonify({
        "ts": time.time(),
        "hostname": os.uname().nodename,
        "uptime_s": time.time() - psutil.boot_time(),
        "cpu": {
            "pct": cpu_overall, "per_core": cpu_per_core,
            "count_logical": psutil.cpu_count(), "count_physical": psutil.cpu_count(logical=False),
            "load1": load1, "load5": load5, "load15": load15,
        },
        "mem": {
            "used_gb": round(vm.used / 1e9, 2), "total_gb": round(vm.total / 1e9, 2), "pct": vm.percent,
            "swap_used_gb": round(sm.used / 1e9, 2), "swap_total_gb": round(sm.total / 1e9, 2), "swap_pct": sm.percent,
        },
        "disks": disks,
        "net": _net_rates(),
        "gpu": _gpu_stats(),
        "ollama": _ollama_stats(),
        "plex": _plex_stats(),
    })


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)