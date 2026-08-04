#!/usr/bin/env python3
"""Isolated load / perf harness for a single DGX inference service.

Drives ONE DGX service (diarization / Whisper / vLLM) at a concurrency ramp while
correlating client-side latency/throughput/errors with server-side memory + GPU
pulled from the homelab VictoriaMetrics stack. Produces a per-concurrency baseline
table and flags the memory/degradation ceiling.

Context: INCIDENT-2026-08-04 (#1397) — a batch drove concurrent diarization onto the
single-GPU, unified-memory DGX with no server-side guardrail; `/dev/shm` exhausted the
box. This harness characterises each service *in isolation* so we know its real
concurrency ceiling and memory-per-request before running them together again.

**Isolation is the operator's responsibility**: load only the target service on the
DGX (stop the others so the ~unified pool isn't already reserved) before running.

Usage (run once the DGX is back up):

    # diarization — ramp 1,2,4 concurrent, 60s each, against a real episode wav
    python scripts/perf/dgx_service_loadtest.py \
        --service diarize --host dgx-llm-1 --port 8001 \
        --concurrency 1,2,4 --duration 60 --audio /path/to/episode.wav \
        --vm-ssh homelab-claude --out reports/dgx-diarize-baseline.json

    # whisper
    python scripts/perf/dgx_service_loadtest.py --service whisper --port 8000 \
        --audio /path/to/episode.wav --concurrency 1,2,4 ...

    # vllm (Qwen) — prompt-driven
    python scripts/perf/dgx_service_loadtest.py --service vllm --port 8003 \
        --prompt "Summarise: ..." --max-tokens 256 --concurrency 1,4,8 ...

Server-side metrics are read from VictoriaMetrics (`:8428`) on the homelab. Because
that endpoint is LAN-only, `--vm-ssh <ssh-host>` shells the query through the homelab
(default). Pass `--vm-url http://host:8428` to query directly if reachable.

NOTE: only the observability half (VM queries) can be validated without the DGX; the
load half requires the target service to be up. Dry-run with `--dry-run` to print the
plan + confirm VM connectivity without sending load.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

try:
    import httpx
except ImportError:  # pragma: no cover
    print("httpx required: pip install httpx", file=sys.stderr)
    raise


# --------------------------------------------------------------------------- #
# Server-side observability (homelab VictoriaMetrics)
# --------------------------------------------------------------------------- #


def _vm_query_range(
    query: str, start: float, end: float, *, vm_ssh: Optional[str], vm_url: str, step: int = 15
) -> list[tuple[float, float]]:
    """Run a VictoriaMetrics range query, return [(ts, value)]. Routes through ssh
    when the VM endpoint is LAN-only (the default homelab deployment)."""
    params = f"query={query}&start={int(start)}&end={int(end)}&step={step}"
    url = f"{vm_url}/api/v1/query_range"
    if vm_ssh:
        cmd = [
            "ssh",
            vm_ssh,
            f"curl -s --get --data-urlencode {shlex.quote('query=' + query)} "
            f"--data 'start={int(start)}' --data 'end={int(end)}' "
            f"--data 'step={step}' {shlex.quote(url)}",
        ]
        raw = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    else:
        raw = httpx.get(url, params=params, timeout=30).text
    try:
        result = json.loads(raw).get("data", {}).get("result", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    if not result:
        return []
    return [(float(t), float(v)) for t, v in result[0]["values"]]


def _sample_server_window(
    instance: str, start: float, end: float, *, vm_ssh: Optional[str], vm_url: str
) -> dict[str, Any]:
    """Peak / minimum of the memory + GPU signals over [start, end] on the DGX."""

    def series(expr: str) -> list[float]:
        return [v for _, v in _vm_query_range(expr, start, end, vm_ssh=vm_ssh, vm_url=vm_url)]

    mem_avail = series(f'node_memory_MemAvailable_bytes{{instance="{instance}"}}/1e9')
    shmem = series(f'node_memory_Shmem_bytes{{instance="{instance}"}}/1e9')
    return {
        "mem_available_min_gb": round(min(mem_avail), 2) if mem_avail else None,
        "shmem_peak_gb": round(max(shmem), 2) if shmem else None,
        "shmem_delta_gb": round(max(shmem) - min(shmem), 2) if shmem else None,
        "samples": len(mem_avail),
    }


# --------------------------------------------------------------------------- #
# Load drivers (one request builder per service)
# --------------------------------------------------------------------------- #


def _build_request(service: str, base: str, args: argparse.Namespace) -> dict[str, Any]:
    """Return kwargs for an httpx request for the given service (one job)."""
    if service == "diarize":
        return {
            "method": "POST",
            "url": f"{base}/v1/diarize",
            "files": {"file": ("audio", open(args.audio, "rb"))},
            "data": {"min_speakers": 2, "max_speakers": 20},
        }
    if service == "whisper":
        return {
            "method": "POST",
            "url": f"{base}/v1/audio/transcriptions",
            "files": {"file": ("audio", open(args.audio, "rb"))},
            "data": {"model": args.model or "whisper-1"},
        }
    if service == "vllm":
        return {
            "method": "POST",
            "url": f"{base}/v1/completions",
            "json": {
                "model": args.model or "qwen",
                "prompt": args.prompt,
                "max_tokens": args.max_tokens,
            },
        }
    raise ValueError(f"unknown service {service}")


@dataclass
class LevelResult:
    concurrency: int
    requests: int
    errors: int
    p50_ms: Optional[float] = None
    p95_ms: Optional[float] = None
    throughput_rps: Optional[float] = None
    server: dict[str, Any] = field(default_factory=dict)


async def _run_level(
    service: str, base: str, args: argparse.Namespace, concurrency: int
) -> LevelResult:
    """Sustain `concurrency` in-flight requests for `args.duration` seconds."""
    latencies: list[float] = []
    errors = 0
    stop_at = time.monotonic() + args.duration
    started = time.time()

    async def worker() -> None:
        nonlocal errors
        async with httpx.AsyncClient(timeout=args.request_timeout) as client:
            while time.monotonic() < stop_at:
                req = _build_request(service, base, args)
                t0 = time.monotonic()
                try:
                    resp = await client.request(**req)
                    if resp.status_code >= 400:
                        errors += 1
                    else:
                        latencies.append((time.monotonic() - t0) * 1000)
                except Exception:  # noqa: BLE001 — load harness records, never crashes
                    errors += 1

    await asyncio.gather(*[worker() for _ in range(concurrency)])
    ended = time.time()

    res = LevelResult(concurrency=concurrency, requests=len(latencies) + errors, errors=errors)
    if latencies:
        res.p50_ms = round(statistics.median(latencies), 1)
        res.p95_ms = round(sorted(latencies)[int(len(latencies) * 0.95)], 1)
        res.throughput_rps = round(len(latencies) / max(ended - started, 1e-6), 3)
    res.server = _sample_server_window(
        args.instance, started, ended, vm_ssh=args.vm_ssh, vm_url=args.vm_url
    )
    return res


# --------------------------------------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--service", required=True, choices=["diarize", "whisper", "vllm"])
    p.add_argument("--host", default="dgx-llm-1", help="DGX tailnet host")
    p.add_argument("--port", type=int, required=True)
    p.add_argument(
        "--instance", default="dgx-llm-1", help="VictoriaMetrics instance label for the DGX"
    )
    p.add_argument("--concurrency", default="1,2,4", help="comma-separated ramp, e.g. 1,2,4,8")
    p.add_argument(
        "--duration", type=float, default=60.0, help="seconds of sustained load per level"
    )
    p.add_argument("--request-timeout", type=float, default=600.0)
    p.add_argument("--audio", help="audio file (diarize/whisper)")
    p.add_argument("--prompt", default="Summarise the following in one sentence: hello world.")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--model")
    p.add_argument(
        "--vm-ssh",
        default="homelab-claude",
        help="ssh host to route VM queries through (LAN-only VM)",
    )
    p.add_argument(
        "--vm-url",
        default="http://localhost:8428",
        help="VictoriaMetrics base URL (on the vm-ssh host)",
    )
    p.add_argument("--out", help="write JSON report here")
    p.add_argument(
        "--dry-run", action="store_true", help="print plan + check VM connectivity, send no load"
    )
    args = p.parse_args()

    if args.service in ("diarize", "whisper") and not args.audio and not args.dry_run:
        p.error(f"--audio is required for --service {args.service}")

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    base = f"http://{args.host}:{args.port}"
    print(f"# DGX load test: service={args.service} target={base} instance={args.instance}")
    print(f"# ramp={levels} duration={args.duration}s per level  (INCIDENT-2026-08-04 / #1397)")

    # Validate VM connectivity (this half works without the DGX).
    now = time.time()
    probe = _sample_server_window(
        args.instance, now - 120, now, vm_ssh=args.vm_ssh, vm_url=args.vm_url
    )
    print(
        f"# VM connectivity: {probe['samples']} samples in last 120s "
        f"(mem_avail_min={probe['mem_available_min_gb']}GB shmem_peak={probe['shmem_peak_gb']}GB)"
    )
    if args.dry_run:
        print("# --dry-run: plan validated, VM reachable; not sending load.")
        return 0

    results: list[LevelResult] = []
    for c in levels:
        print(f"\n## concurrency={c} …")
        res = asyncio.run(_run_level(args.service, base, args, c))
        results.append(res)
        print(
            f"   reqs={res.requests} errors={res.errors} p50={res.p50_ms}ms p95={res.p95_ms}ms "
            f"rps={res.throughput_rps} | mem_avail_min={res.server.get('mem_available_min_gb')}GB "
            f"shmem_peak={res.server.get('shmem_peak_gb')}GB "
            f"shmem_delta={res.server.get('shmem_delta_gb')}GB"
        )
        # Ceiling guard: stop ramping if the box is getting close to the OOM cliff.
        min_avail = res.server.get("mem_available_min_gb")
        if min_avail is not None and min_avail < 10:
            print(
                f"   !! mem_available dipped to {min_avail}GB — stopping ramp (OOM ceiling near)."
            )
            break

    report = {
        "service": args.service,
        "target": base,
        "instance": args.instance,
        "ramp": levels,
        "duration_sec": args.duration,
        "levels": [asdict(r) for r in results],
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n# wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
