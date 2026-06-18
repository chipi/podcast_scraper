#!/usr/bin/env python3
"""Poll the vLLM /metrics Prometheus endpoint and emit a compact one-line
sample per interval. Feeds #1016 / #1022 vLLM-on-GB10 tuning observations.

The original Round 2 collector wrote timestamps but no actual metric values —
either curl-grep mismatched the metric name prefix, or the parsed body was
empty for the time-windowed runs. This rewrite hits the endpoint cleanly via
urllib, parses Prometheus format, and reports:

- ``kv_cache_pct``    — `vllm:gpu_cache_usage_perc` × 100
- ``running``         — `vllm:num_requests_running`
- ``waiting``         — `vllm:num_requests_waiting`
- ``ttft_p50_ms``     — median TTFT from the histogram (approximation: pick
                         the bucket where cumulative count crosses the
                         midpoint, return the bucket's upper bound)
- ``tpot_p50_ms``     — same approximation for time-per-output-token
- ``gen_tps``         — generation throughput derived from
                         `vllm:request_generation_tokens_total` delta

Usage:

    PYTHONPATH=. .venv/bin/python scripts/eval/poll_vllm_metrics.py \\
        --base-url http://dgx-llm-1.tail6d0ed4.ts.net:8003 \\
        --interval 5 \\
        --model qwen3_5_35b \\
        --label phase2c_gi_round3_v1 \\
        >> docs/wip/EVAL_1016_metrics/vllm_kv_metrics.log

Exit on Ctrl-C; the loop is robust to transient HTTP errors (logs and
continues). Run this in a separate terminal / background process while an
experiment is making chat-completions calls.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

_METRICS_OF_INTEREST = (
    "vllm:gpu_cache_usage_perc",
    "vllm:cpu_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
    "vllm:time_to_first_token_seconds_bucket",
    "vllm:time_per_output_token_seconds_bucket",
    "vllm:request_generation_tokens_total",
    "vllm:request_prompt_tokens_total",
)


def _fetch(url: str, timeout: int = 5) -> str:
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def _parse_simple(body: str, metric: str) -> float | None:
    """Find the first non-comment line whose metric name is `metric` and return
    its float value. Ignores labels (uses the line's last whitespace-separated
    field as the value)."""
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith(metric) and (line[len(metric)] in (" ", "{")):
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                try:
                    return float(parts[1])
                except ValueError:
                    return None
    return None


def _parse_histogram_median(body: str, metric_base: str) -> float | None:
    """Approximate the histogram p50 from a ``<metric_base>_bucket{le="..."} N``
    series. Returns the upper bound of the bucket where cumulative count first
    crosses the midpoint, in the same unit as the bucket's ``le`` label
    (seconds for vLLM's TTFT/TPOT histograms).
    """
    bucket_prefix = f"{metric_base}_bucket{{"
    buckets: list[tuple[float, float]] = []  # (upper_bound, cumulative_count)
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        if not line.startswith(bucket_prefix):
            continue
        # Format: vllm:time_to_first_token_seconds_bucket{le="0.005",...} N
        # We need the le="..." value AND the trailing count.
        le_start = line.find('le="')
        if le_start < 0:
            continue
        le_end = line.find('"', le_start + 4)
        if le_end < 0:
            continue
        le_str = line[le_start + 4 : le_end]
        try:
            le_val = float(le_str) if le_str != "+Inf" else float("inf")
        except ValueError:
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        try:
            cum = float(parts[1])
        except ValueError:
            continue
        buckets.append((le_val, cum))
    if not buckets:
        return None
    buckets.sort()
    total = buckets[-1][1]
    if total <= 0:
        return None
    half = total / 2.0
    for le_val, cum in buckets:
        if cum >= half:
            return le_val if le_val != float("inf") else buckets[-2][0]
    return None


def sample_once(base_url: str) -> dict[str, float | None]:
    body = _fetch(f"{base_url.rstrip('/')}/metrics")
    return {
        # Confirmed metric names from vLLM 0.20.1+ /metrics endpoint:
        "kv_cache_pct": (_parse_simple(body, "vllm:kv_cache_usage_perc") or 0.0) * 100,
        "running": _parse_simple(body, "vllm:num_requests_running"),
        "waiting": _parse_simple(body, "vllm:num_requests_waiting"),
        "ttft_p50_s": _parse_histogram_median(body, "vllm:time_to_first_token_seconds"),
        "tpot_p50_s": _parse_histogram_median(body, "vllm:request_time_per_output_token_seconds"),
        "gen_tokens_total": _parse_simple(body, "vllm:generation_tokens_total"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-url",
        default="http://dgx-llm-1.tail6d0ed4.ts.net:8003",
        help="vLLM base URL (without /v1)",
    )
    p.add_argument("--interval", type=int, default=5, help="Poll interval in seconds")
    p.add_argument("--model", default="unknown", help="Short model label for log lines")
    p.add_argument("--label", default="", help="Run label (e.g. phase2c_gi_round3_v1)")
    p.add_argument("--max-samples", type=int, default=0, help="0 = forever")
    args = p.parse_args()

    print(
        f"# vllm-metrics poll start: base={args.base_url} interval={args.interval}s "
        f"model={args.model} label={args.label}",
        flush=True,
    )
    print(
        "# ts_utc, kv_pct, run, wait, ttft_p50_ms, tpot_p50_ms, " "gen_tok_total, gen_tps_5s",
        flush=True,
    )

    prev_total: float | None = None
    n = 0
    while True:
        try:
            s = sample_once(args.base_url)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"# poll error: {e}", file=sys.stderr, flush=True)
            time.sleep(args.interval)
            continue
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        cur_total = s["gen_tokens_total"] or 0.0
        if prev_total is None:
            gen_tps: float | None = None
        else:
            delta = cur_total - prev_total
            gen_tps = delta / args.interval if delta >= 0 else None
        prev_total = cur_total
        ttft_ms = s["ttft_p50_s"] * 1000 if s["ttft_p50_s"] is not None else None
        tpot_ms = s["tpot_p50_s"] * 1000 if s["tpot_p50_s"] is not None else None
        print(
            f"{now},{s['kv_cache_pct']:.1f},"
            f"{int(s['running'] or 0)},{int(s['waiting'] or 0)},"
            f"{ttft_ms if ttft_ms is None else f'{ttft_ms:.0f}'},"
            f"{tpot_ms if tpot_ms is None else f'{tpot_ms:.0f}'},"
            f"{int(cur_total)},"
            f"{gen_tps if gen_tps is None else f'{gen_tps:.0f}'}",
            flush=True,
        )
        n += 1
        if args.max_samples and n >= args.max_samples:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
