#!/usr/bin/env python3
"""Turn `docker inspect` output into per-container Prometheus metrics.

The shared helper for the homelab landing page's container inventory — used by
the mini (mini-metrics), the DGX (dgx-scrape, over SSH), and prod (this repo's
container-metrics). Reads the inspect stream on stdin, writes prometheus lines on
stdout; takes the box name as argv[1].

Input: one container per tab-separated line, produced by ``docker inspect`` with
``--format`` over these fields, in order:
    Name, State.Status, State.StartedAt, the compose-project label
    (``index .Config.Labels "com.docker.compose.project"``), and
    ``json .NetworkSettings.Ports``.

Emits (value = uptime seconds; 0 when not running):
    container_uptime_seconds{box,app,name,port,state} <seconds>

`state` drives the traffic light (running=green, restarting/created=amber,
exited/dead=red). `port` is the first published host port, else the first
exposed container port, else "" (host-network services publish nothing).
"""

import json
import re
import sys
from datetime import datetime, timezone

box = sys.argv[1] if len(sys.argv) > 1 else "unknown"
now = datetime.now(timezone.utc)


def esc(v):
    return str(v).replace("\\", "\\\\").replace('"', '\\"')


def pick_port(ports_json):
    try:
        ports = json.loads(ports_json) if ports_json and ports_json != "null" else {}
    except Exception:
        return ""
    # prefer a published host port (something bound to the host)
    for spec, binds in (ports or {}).items():
        if binds:
            hp = binds[0].get("HostPort")
            if hp:
                return hp
    # else the first exposed container port (e.g. "5432/tcp")
    for spec in ports or {}:
        return spec.split("/")[0]
    return ""


for line in sys.stdin:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 5:
        continue
    name, state, started, app, ports_json = parts[0], parts[1], parts[2], parts[3], parts[4]
    name = name.lstrip("/")
    up = 0
    if state == "running" and started and not started.startswith("0001"):
        try:
            # docker emits nanosecond precision (9 digits); py<3.11 fromisoformat
            # only accepts <=6, so truncate the fraction to microseconds.
            s = re.sub(r"\.(\d{6})\d*", r".\1", started.replace("Z", "+00:00"))
            st = datetime.fromisoformat(s)
            up = max(0, int((now - st).total_seconds()))
        except Exception:
            up = 0
    port = pick_port(ports_json)
    print(
        'container_uptime_seconds{box="%s",app="%s",name="%s",port="%s",state="%s"} %d'
        % (esc(box), esc(app), esc(name), esc(port), esc(state), up)
    )
