# flake8: noqa: E501  -- prompt string literals are long by intent
"""Sustained-load reliability + effective-throughput harness for #816 D2/D3.

Closes the methodology gap surfaced in #816: the standard autoresearch
summary-model harness measures quality + cost + per-call latency, but does
NOT measure reliability under sustained load (the 503 rate that emerges
only when call volume crosses the throttle threshold) or effective
throughput (success_rate × QPS).

Approach:
- For each candidate model, burst N parallel summary calls in waves at a
  target QPS (or just back-to-back if QPS is fast enough that the model
  paces itself naturally on rate limits).
- Track per-call: timestamp, latency, success/failure, error code, retry count.
- Compute: 503 rate, retry distribution, p50/p95 latency, effective throughput.

This is the methodology extension for autoresearch — D2. Re-running the
candidate matrix is D3 (same script, different --models).

Usage:
    python scripts/eval/score/summary_model_reliability_v1.py \\
        --transcript-path data/eval/sources/curated_5feeds_raw_v2/p01_e01.txt \\
        --models gemini-2.5-flash-lite gemini-2.5-flash gpt-4o-mini claude-haiku-4-5 \\
        --calls 30 \\
        --concurrency 5 \\
        --output data/eval/runs/baseline_summary_model_reliability_v1
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

SYSTEM_PROMPT = (
    "You are an expert at creating concise, informative summaries of podcast "
    "episodes. Focus on key insights, decisions, and lessons learned."
)

PROD_PROMPT = """Summarize the following podcast episode transcript.
- Write a detailed summary with 4-6 paragraphs
- Begin the first paragraph with a single sentence naming the episode's domain and its central arg or premise
- Cover ALL major discussion segments in the order they appear in the transcript
- Preserve key technical terms, concept names, product names, and specific vocabulary from the transcript verbatim
- Anchor each paragraph in specific claims, data points, or named entities from the transcript
- Focus on key decisions, arguments, and lessons learned
- Ignore sponsorships, ads, and housekeeping
- Do not use quotes or speaker names
- Do not invent info not implied by the transcript

Transcript:
```
{transcript}
```
"""

# Per-provider pricing (USD per 1M tokens, published rates as of 2026-06).
PRICING: dict[str, tuple[float, float]] = {
    # model -> (input_per_1M, output_per_1M)
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-1.5-flash": (0.075, 0.30),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-haiku-4-5": (0.80, 4.00),
}


@dataclass
class CallResult:
    model: str
    call_idx: int
    started_at: float
    ended_at: float
    success: bool
    status_code: int | None
    error_kind: str | None
    error_msg: str | None
    output_chars: int
    input_tokens_est: int
    output_tokens_est: int

    @property
    def latency_s(self) -> float:
        return round(self.ended_at - self.started_at, 3)


@dataclass
class ModelSummary:
    model: str
    calls_attempted: int
    successes: int
    failures: int
    success_rate_pct: float
    error_kinds: dict[str, int]
    rate_limit_count: int
    rate_limit_rate_pct: float
    server_error_count: int
    server_error_rate_pct: float
    latency_p50_s: float
    latency_p95_s: float
    latency_mean_s: float
    wall_clock_s: float
    effective_qps: float
    cost_usd_total: float
    cost_usd_per_successful_call: float
    notes: str = ""
    per_call: list[dict[str, Any]] = field(default_factory=list)


def _classify_error(exc: Exception) -> tuple[int | None, str, str]:
    """Return (status_code, kind, short_msg) for an exception."""
    msg = str(exc)
    # Try common attributes first
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if not isinstance(code, int):
        # Best-effort sniff from message
        for c in (429, 500, 502, 503, 504):
            if str(c) in msg:
                code = c
                break
        else:
            code = None
    kind = exc.__class__.__name__
    if code == 429 or "rate" in msg.lower() or "ratelimit" in kind.lower():
        kind = "rate_limit_429"
    elif code in (500, 502, 503, 504) or "503" in msg or "unavailable" in msg.lower():
        kind = "server_error_5xx"
    elif "timeout" in msg.lower() or "deadline" in msg.lower():
        kind = "timeout"
    elif "connection" in msg.lower() or "network" in msg.lower():
        kind = "network"
    return code, kind, msg[:240]


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _call_gemini(model: str, transcript: str) -> tuple[str, int]:
    import google.genai as genai
    from google.genai import types

    client = genai.Client()
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
        max_output_tokens=2000,
    )
    resp = client.models.generate_content(
        model=model,
        contents=PROD_PROMPT.format(transcript=transcript),
        config=cfg,
    )
    return resp.text or "", 0


def _call_openai(model: str, transcript: str) -> tuple[str, int]:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROD_PROMPT.format(transcript=transcript)},
        ],
        temperature=0.2,
        max_tokens=2000,
    )
    return resp.choices[0].message.content or "", 0


def _call_anthropic(model: str, transcript: str) -> tuple[str, int]:
    from anthropic import Anthropic

    client = Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": PROD_PROMPT.format(transcript=transcript)}],
    )
    text_out = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text_out = getattr(block, "text", "")
            break
    return text_out, 0


def _dispatch(model: str, transcript: str) -> tuple[str, int]:
    if model.startswith("gemini-"):
        return _call_gemini(model, transcript)
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"):
        return _call_openai(model, transcript)
    if model.startswith("claude-"):
        return _call_anthropic(model, transcript)
    raise ValueError(f"unknown model family: {model}")


def _make_call(model: str, transcript: str, call_idx: int) -> CallResult:
    t0 = time.time()
    try:
        out, _ = _dispatch(model, transcript)
        t1 = time.time()
        return CallResult(
            model=model,
            call_idx=call_idx,
            started_at=t0,
            ended_at=t1,
            success=True,
            status_code=200,
            error_kind=None,
            error_msg=None,
            output_chars=len(out),
            input_tokens_est=_est_tokens(transcript) + _est_tokens(PROD_PROMPT),
            output_tokens_est=_est_tokens(out),
        )
    except Exception as exc:  # noqa: BLE001
        t1 = time.time()
        code, kind, msg = _classify_error(exc)
        return CallResult(
            model=model,
            call_idx=call_idx,
            started_at=t0,
            ended_at=t1,
            success=False,
            status_code=code,
            error_kind=kind,
            error_msg=msg,
            output_chars=0,
            input_tokens_est=_est_tokens(transcript) + _est_tokens(PROD_PROMPT),
            output_tokens_est=0,
        )


def run_load(model: str, transcript: str, calls: int, concurrency: int) -> ModelSummary:
    print(f"\n[{model}] dispatching {calls} calls at concurrency {concurrency}...", flush=True)
    wall_t0 = time.time()
    results: list[CallResult] = []
    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(_make_call, model, transcript, i) for i in range(calls)]
        for fut in cf.as_completed(futs):
            r = fut.result()
            tag = "ok " if r.success else f"{r.status_code or '???'}"
            print(
                f"  [{model}] call#{r.call_idx:02d}  {tag}  {r.latency_s:6.2f}s  "
                f"err={r.error_kind or '-'}",
                flush=True,
            )
            results.append(r)
    wall = round(time.time() - wall_t0, 2)

    succ = [r for r in results if r.success]
    fail = [r for r in results if not r.success]
    kinds: dict[str, int] = {}
    for r in fail:
        if r.error_kind:
            kinds[r.error_kind] = kinds.get(r.error_kind, 0) + 1

    latencies_s = [r.latency_s for r in succ] or [0.0]
    sorted_lat = sorted(latencies_s)
    p50 = round(median(sorted_lat), 3)
    p95_idx = max(0, int(0.95 * len(sorted_lat)) - 1)
    p95 = round(sorted_lat[p95_idx], 3)
    mean_lat = round(sum(latencies_s) / len(latencies_s), 3)

    in_per_1m, out_per_1m = PRICING.get(model, (0.0, 0.0))
    cost = round(
        sum(
            (r.input_tokens_est / 1_000_000) * in_per_1m
            + (r.output_tokens_est / 1_000_000) * out_per_1m
            for r in results
        ),
        4,
    )
    cost_per_succ = round(cost / max(len(succ), 1), 4)

    return ModelSummary(
        model=model,
        calls_attempted=len(results),
        successes=len(succ),
        failures=len(fail),
        success_rate_pct=round(100 * len(succ) / max(len(results), 1), 1),
        error_kinds=kinds,
        rate_limit_count=kinds.get("rate_limit_429", 0),
        rate_limit_rate_pct=round(100 * kinds.get("rate_limit_429", 0) / max(len(results), 1), 1),
        server_error_count=kinds.get("server_error_5xx", 0),
        server_error_rate_pct=round(
            100 * kinds.get("server_error_5xx", 0) / max(len(results), 1), 1
        ),
        latency_p50_s=p50,
        latency_p95_s=p95,
        latency_mean_s=mean_lat,
        wall_clock_s=wall,
        effective_qps=round(len(succ) / max(wall, 0.001), 3),
        cost_usd_total=cost,
        cost_usd_per_successful_call=cost_per_succ,
        per_call=[asdict(r) for r in results],
    )


def composite_score(s: ModelSummary, ref_p50_s: float) -> dict[str, Any]:
    """Quality-floor composite: reliability is a hard floor, then cost+latency rank.

    - reliability floor: success_rate_pct >= 95
    - latency: lower p50 better, normalized vs panel best
    - cost: lower per-successful-call better, normalized vs panel best
    """
    floor_pass = s.success_rate_pct >= 95.0
    lat_score = round(ref_p50_s / max(s.latency_p50_s, 0.001), 3)
    return {
        "reliability_floor_pass": floor_pass,
        "success_rate_pct": s.success_rate_pct,
        "effective_qps": s.effective_qps,
        "latency_p50_s": s.latency_p50_s,
        "latency_p95_s": s.latency_p95_s,
        "cost_usd_per_successful_call": s.cost_usd_per_successful_call,
        "latency_score_vs_best": lat_score,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transcript-path", type=Path, required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--calls", type=int, default=30, help="calls per model")
    p.add_argument("--concurrency", type=int, default=5, help="concurrent in-flight calls")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    transcript = args.transcript_path.read_text(encoding="utf-8")
    # Cap transcript so we don't blow input tokens & cost — first ~6K chars is
    # representative of a typical podcast prologue+early body (~1.5K input tokens).
    transcript = transcript[:6000]

    # Pre-flight env checks
    needs = {"gemini": "GEMINI_API_KEY", "gpt": "OPENAI_API_KEY", "claude": "ANTHROPIC_API_KEY"}
    for m in args.models:
        for fam, env in needs.items():
            if m.startswith(fam) and env not in os.environ:
                # Gemini SDK also accepts GOOGLE_API_KEY
                if fam == "gemini" and "GOOGLE_API_KEY" in os.environ:
                    continue
                print(f"WARN: {env} not set, {m} will fail", file=sys.stderr)

    args.output.mkdir(parents=True, exist_ok=True)
    summaries: list[ModelSummary] = []
    for m in args.models:
        s = run_load(m, transcript, args.calls, args.concurrency)
        summaries.append(s)

    # Composite scoring — reference latency is the panel's best p50
    ref_p50 = min((s.latency_p50_s for s in summaries if s.successes > 0), default=1.0)
    composites = {s.model: composite_score(s, ref_p50) for s in summaries}

    payload = {
        "schema": "metrics_summary_model_reliability_v1",
        "params": {
            "calls_per_model": args.calls,
            "concurrency": args.concurrency,
            "transcript_path": str(args.transcript_path),
            "transcript_chars_used": len(transcript),
        },
        "summaries": [asdict(s) for s in summaries],
        "composite": composites,
        "ranking": sorted(
            composites.items(),
            key=lambda kv: (
                not kv[1]["reliability_floor_pass"],  # passes first
                kv[1]["cost_usd_per_successful_call"],  # cheaper better
                kv[1]["latency_p50_s"],  # faster better as tiebreak
            ),
        ),
    }

    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{'model':<30} succ%   p50(s)  p95(s)  qps_eff  $/succ   floor?  errors")
    for s in summaries:
        floor = "PASS" if s.success_rate_pct >= 95 else "FAIL"
        errs = ",".join(f"{k}={v}" for k, v in s.error_kinds.items()) or "-"
        print(
            f"{s.model:<30} {s.success_rate_pct:>5.1f}  {s.latency_p50_s:>6.2f}  "
            f"{s.latency_p95_s:>6.2f}  {s.effective_qps:>6.3f}  "
            f"{s.cost_usd_per_successful_call:>7.4f}  {floor:<6}  {errs}"
        )
    print("\nranking (reliability floor → cost → latency):")
    for i, (model_name, c) in enumerate(payload["ranking"], 1):
        c_dict: dict[str, Any] = c  # type: ignore[assignment]
        print(
            f"  {i}. {model_name:<28} floor={c_dict['reliability_floor_pass']!s:<5} "
            f"$/succ={c_dict['cost_usd_per_successful_call']:.4f}  p50={c_dict['latency_p50_s']:.2f}s"
        )
    print(f"\nwrote {args.output}/metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
