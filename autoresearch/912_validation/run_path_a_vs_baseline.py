"""#912 Path A vs baseline parse-rate validation.

Two-arm A/B against qwen3.5:9b on Ollama at localhost:11434.

- **Baseline**: production code path (`response_format={"type":"json_object"}`)
- **Path A**: proposed fix (`extra_body={"format": SCHEMA}` — Ollama native
  grammar-constrained decoding via the OpenAI-compat endpoint)

Same prompts, same params, same fixture, N trials per arm. Captures
parse success, parse error, wall-clock, response shape. Early-exit
on the Path A arm if median latency on the first 3 trials exceeds
60s (the prior session's 175s tax shouldn't be re-experienced silently).

Run:
    .venv/bin/python autoresearch/912_validation/run_path_a_vs_baseline.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Template
from openai import OpenAI

REPO = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = REPO / "tests/fixtures/transcripts/v2/p01_e01_fast.txt"
SYS_TMPL = (
    REPO / "src/podcast_scraper/prompts/ollama/summarization/bundled_clean_summary_system_v1.j2"
)
USER_TMPL = (
    REPO / "src/podcast_scraper/prompts/ollama/summarization/bundled_clean_summary_user_v1.j2"
)

# JSON shape the prompt requests: {"title": null|str, "summary": str, "bullets": [str]}
SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": ["string", "null"]},
        "summary": {"type": "string"},
        "bullets": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": ["title", "summary", "bullets"],
}

# Mirror production-call params from OllamaProvider.summarize_bundled
PARAMS = dict(
    paragraphs_min=4,
    paragraphs_max=6,
    max_words_per_bullet=45,
    bullet_min=3,
)
TEMPERATURE = 0.2
MAX_TOKENS = 16384
MODEL = "qwen3.5:9b"
DEFAULT_ENDPOINT = "http://localhost:11434/v1"


def render_prompts(fixture: Path) -> tuple[str, str]:
    sys_text = Template(SYS_TMPL.read_text()).render(**PARAMS)
    user_text = Template(USER_TMPL.read_text()).render(
        transcript=fixture.read_text(),
        title="Building Trails That Last",
        **PARAMS,
    )
    return sys_text, user_text


def call_once(
    client: OpenAI,
    sys_text: str,
    user_text: str,
    *,
    arm: str,
) -> Dict[str, Any]:
    """Run one trial. arm in {"baseline", "path_a"}."""
    messages = [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": user_text},
    ]
    kwargs: Dict[str, Any] = dict(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    # Ollama-specific options (matching production OllamaProvider construction)
    extra_body: Dict[str, Any] = {
        "options": {"num_ctx": 32768},
        "reasoning_effort": "none",  # qwen3.5 reasoning suppression
    }
    if arm == "baseline":
        kwargs["response_format"] = {"type": "json_object"}
    elif arm == "path_a":
        extra_body["format"] = SCHEMA
    else:
        raise ValueError(arm)
    kwargs["extra_body"] = extra_body

    t0 = time.perf_counter()
    err: Optional[str] = None
    raw: str = ""
    finish_reason: Optional[str] = None
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    try:
        resp = client.chat.completions.create(**kwargs)
        raw = (resp.choices[0].message.content or "").strip()
        finish_reason = resp.choices[0].finish_reason
        usage = getattr(resp, "usage", None)
        if usage is not None:
            completion_tokens = getattr(usage, "completion_tokens", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - t0

    parse_ok = False
    parse_error: Optional[str] = None
    parsed_shape: Dict[str, Any] = {}
    if raw and not err:
        try:
            obj = json.loads(raw, strict=False)
            parse_ok = True
            if isinstance(obj, dict):
                parsed_shape = {
                    "has_title": "title" in obj,
                    "has_summary": isinstance(obj.get("summary"), str)
                    and len(obj.get("summary") or "") > 0,
                    "n_bullets": (
                        len(obj.get("bullets") or []) if isinstance(obj.get("bullets"), list) else 0
                    ),
                    "summary_chars": len(obj.get("summary") or ""),
                }
        except json.JSONDecodeError as e:
            parse_error = f"{type(e).__name__}: {e}"

    return dict(
        arm=arm,
        elapsed_s=round(elapsed, 3),
        request_error=err,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        response_chars=len(raw),
        parse_ok=parse_ok,
        parse_error=parse_error,
        parsed_shape=parsed_shape,
        raw_response_preview=raw[:500] if raw else "",
    )


def summarize(trials: List[Dict[str, Any]], arm: str) -> Dict[str, Any]:
    arm_trials = [t for t in trials if t["arm"] == arm]
    n = len(arm_trials)
    if n == 0:
        return {"arm": arm, "n": 0}
    parse_ok = sum(1 for t in arm_trials if t["parse_ok"])
    has_summary = sum(
        1 for t in arm_trials if t["parse_ok"] and t["parsed_shape"].get("has_summary")
    )
    latencies = [t["elapsed_s"] for t in arm_trials if t["request_error"] is None]
    return dict(
        arm=arm,
        n=n,
        parse_ok=parse_ok,
        parse_rate=round(parse_ok / n, 3),
        has_summary=has_summary,
        median_latency_s=round(statistics.median(latencies), 2) if latencies else None,
        p95_latency_s=(
            round(statistics.quantiles(latencies, n=20)[-1], 2) if len(latencies) >= 20 else None
        ),
        max_latency_s=round(max(latencies), 2) if latencies else None,
        median_completion_tokens=(
            round(
                statistics.median(
                    [
                        t["completion_tokens"]
                        for t in arm_trials
                        if t.get("completion_tokens") is not None
                    ]
                ),
                0,
            )
            if any(t.get("completion_tokens") is not None for t in arm_trials)
            else None
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Trials per arm (default 10).",
    )
    parser.add_argument(
        "--early-exit-latency-s",
        type=float,
        default=60.0,
        help="Abort Path A arm if median of first 3 trials exceeds this (default 60s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "autoresearch/912_validation/trials.jsonl",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE,
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=DEFAULT_ENDPOINT,
        help=(
            "OpenAI-compat base URL (default: laptop localhost). "
            "Use http://dgx-llm-1:11434/v1 for DGX."
        ),
    )
    args = parser.parse_args()

    sys_text, user_text = render_prompts(args.fixture)
    print(f"[setup] sys={len(sys_text)} chars, user={len(user_text)} chars, model={MODEL}")
    print(f"[setup] fixture={args.fixture.name} ({args.fixture.stat().st_size} bytes)")
    print(f"[setup] trials/arm={args.trials}, output={args.output}")

    print(f"[setup] endpoint={args.endpoint}")
    client = OpenAI(base_url=args.endpoint, api_key="ollama-local")
    trials: List[Dict[str, Any]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("")  # truncate

    for arm in ("baseline", "path_a"):
        print(f"\n[{arm}] running {args.trials} trials...")
        arm_latencies: List[float] = []
        for i in range(args.trials):
            t = call_once(client, sys_text, user_text, arm=arm)
            t["trial_idx"] = i
            trials.append(t)
            with args.output.open("a") as f:
                f.write(json.dumps(t) + "\n")
            parse_marker = "✓" if t["parse_ok"] else "✗"
            err_note = (
                f" parse_err={t['parse_error'][:80]}"
                if t["parse_error"]
                else f" req_err={t['request_error'][:80]}" if t["request_error"] else ""
            )
            print(
                f"  [{arm} #{i:02d}] {parse_marker} {t['elapsed_s']}s "
                f"({t.get('response_chars', 0)} chars){err_note}"
            )
            arm_latencies.append(t["elapsed_s"])

            if arm == "path_a" and i == 2:
                med3 = statistics.median(arm_latencies[:3])
                if med3 > args.early_exit_latency_s:
                    print(
                        f"  [{arm}] EARLY EXIT — median first-3 = {med3:.1f}s "
                        f"> {args.early_exit_latency_s}s (Path A still slow)"
                    )
                    break

    print("\n=== SUMMARY ===")
    for arm in ("baseline", "path_a"):
        s = summarize(trials, arm)
        print(f"  {s}")
    print(f"\n[done] trials written to {args.output}")


if __name__ == "__main__":
    main()
