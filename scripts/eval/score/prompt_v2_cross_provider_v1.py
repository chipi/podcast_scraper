# flake8: noqa: E501
"""Cross-provider port of the #906 v1↔v2 pairwise validation (#985).

Same shape as ``prompt_v2_validation_v1.py`` but parameterized over provider.
Loads ``<provider>/summarization/long_v1.j2`` and ``long_v2.j2`` from the
shipped prompt store (not inlined strings), generates two summaries per
episode against the chosen provider's SDK, and lets Sonnet 4.6 judge.

Judge is held constant across providers (Sonnet 4.6) so per-provider
verdicts are comparable to the #906 Anthropic result (5-0-0 for v2).

Acceptance gate per #985: v2 wins ≥ 4/5 episodes → flip default for that
provider; else keep v1 + document the per-provider verdict.

Usage:
    python scripts/eval/score/prompt_v2_cross_provider_v1.py \\
        --provider openai \\
        --sources data/eval/sources/curated_5feeds_raw_v2 \\
        --episodes p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \\
        --output  data/eval/runs/prompt_v2_cross_provider/openai
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

SYSTEM_PROMPT = (
    "You are an expert at creating concise, informative summaries of podcast "
    "episodes. Focus on key insights, decisions, and lessons learned."
)

JUDGE_SYSTEM = (
    "You are evaluating two candidate summaries (A and B) of the same podcast "
    "episode transcript. Better = covers main claims faithfully, preserves "
    "named entities, captures position changes if present, mentions recurring "
    "guests if cited by the host, and does NOT invent or paraphrase. "
    'Reply STRICT JSON: {"winner": "A" | "B" | "TIE", "reason": "<short>"}'
)


def _load_dotenv_if_present() -> None:
    """Mirror the pattern used by other scoring scripts — read repo-root .env
    if the runtime hasn't already exported the keys.
    """
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def _render(provider: str, version: str, transcript: str, title: str | None) -> str:
    """Render ``<provider>/summarization/long_v<version>`` with transcript."""
    from podcast_scraper.prompts.store import render_prompt

    return render_prompt(
        f"{provider}/summarization/long_v{version}",
        transcript=transcript,
        title=title or "",
        paragraphs_min=4,
        paragraphs_max=6,
    )


def _make_summarizer(provider: str) -> Callable[[str], str]:
    """Return a single-arg callable: prompt -> generated summary."""
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI()

        def go(prompt: str) -> str:
            resp = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.0,
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return (resp.choices[0].message.content or "").strip()

        return go

    if provider == "gemini":
        from google import genai

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        # gemini-2.5-flash-lite is the production default per
        # PROD_DEFAULT_GEMINI_SUMMARY_MODEL + cloud_balanced/cloud_thin
        # profiles. Plain "flash" has thinking-mode token accounting that
        # makes max_output_tokens=2000 produce inconsistent short outputs.
        def go(prompt: str) -> str:
            resp = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
                config={"temperature": 0.0, "max_output_tokens": 2000},
            )
            return (resp.text or "").strip()

        return go

    if provider == "deepseek":
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )

        def go(prompt: str) -> str:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                temperature=0.0,
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return (resp.choices[0].message.content or "").strip()

        return go

    if provider == "ollama":
        # qwen3.5:35b is the production default per cloud_with_dgx_primary;
        # honor the same DGX endpoint as the runtime. Use the *native*
        # /api/chat endpoint (not the /v1 OpenAI shim) because qwen3.5 is
        # a thinking model and the OpenAI shim has no `think: false` knob —
        # without disabling thinking, the model burns the entire num_predict
        # budget on hidden reasoning and returns empty content (caught in
        # EVAL_REAL90_2026_06.md; same trap that bit the #959 smoke).
        import httpx

        base = os.environ.get(
            "OLLAMA_API_BASE_NATIVE",
            "http://dgx-llm-1.tail6d0ed4.ts.net:11434",
        )
        model = os.environ.get("OLLAMA_SUMMARY_MODEL", "qwen3.5:35b")
        timeout = float(os.environ.get("OLLAMA_TIMEOUT_S", "300"))

        def go(prompt: str) -> str:
            with httpx.Client(timeout=timeout) as http:
                resp = http.post(
                    f"{base}/api/chat",
                    json={
                        "model": model,
                        "stream": False,
                        "think": False,
                        "options": {"temperature": 0.0, "num_predict": 2000},
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                    },
                )
                resp.raise_for_status()
                return (resp.json()["message"]["content"] or "").strip()

        return go

    raise SystemExit(f"unknown provider: {provider}")


def judge(client: Any, transcript: str, a: str, b: str) -> dict[str, Any]:
    msg = (
        f"Transcript (truncated to 4000 chars):\n```\n{transcript[:4000]}\n```\n\n"
        f"Summary A:\n```\n{a}\n```\n\n"
        f"Summary B:\n```\n{b}\n```\n\n"
        "Which is better? STRICT JSON only."
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        temperature=0.0,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": msg}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"winner": "PARSE_ERROR", "reason": text}


def main() -> int:
    _load_dotenv_if_present()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--provider",
        required=True,
        choices=["openai", "gemini", "deepseek", "ollama"],
    )
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ANTHROPIC_API_KEY required (judge)", file=sys.stderr)
        return 1

    summarize = _make_summarizer(args.provider)
    from anthropic import Anthropic

    judge_client = Anthropic()
    args.output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    t0 = time.time()

    for ep in args.episodes:
        srcs = list(args.sources.rglob(f"{ep}.txt"))
        if not srcs:
            print(f"  SKIP {ep}: no source", file=sys.stderr)
            continue
        transcript = srcs[0].read_text(encoding="utf-8")
        prompt_v1 = _render(args.provider, "1", transcript, ep)
        prompt_v2 = _render(args.provider, "2", transcript, ep)
        try:
            summ_a = summarize(prompt_v1)
            summ_b = summarize(prompt_v2)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERR {ep}: {exc}", file=sys.stderr)
            continue
        verdict = judge(judge_client, transcript, summ_a, summ_b)
        results.append(
            {
                "episode": ep,
                "summary_v1_chars": len(summ_a),
                "summary_v2_chars": len(summ_b),
                "winner": verdict.get("winner"),
                "reason": verdict.get("reason"),
            }
        )
        print(
            f"  [{time.time()-t0:6.1f}s] {ep}  "
            f"v1={len(summ_a)}c  v2={len(summ_b)}c  -> {verdict.get('winner')}"
        )

    wins = {"A_v1": 0, "B_v2": 0, "TIE": 0, "OTHER": 0}
    for r in results:
        w = r.get("winner")
        if w == "A":
            wins["A_v1"] += 1
        elif w == "B":
            wins["B_v2"] += 1
        elif w == "TIE":
            wins["TIE"] += 1
        else:
            wins["OTHER"] += 1

    if wins["B_v2"] >= 4:
        verdict_str = (
            f"v2 wins {wins['B_v2']}/{len(results)} — flip default to long_v2 for {args.provider}"
        )
    elif wins["A_v1"] > wins["B_v2"]:
        verdict_str = (
            f"v1 still wins ({wins['A_v1']} vs {wins['B_v2']}) — keep {args.provider} on long_v1"
        )
    else:
        verdict_str = f"mixed ({wins['A_v1']} v1 / {wins['B_v2']} v2 / {wins['TIE']} tie) — hold {args.provider} on long_v1 pending further evidence"

    payload = {
        "schema": "metrics_prompt_v2_cross_provider_v1",
        "provider": args.provider,
        "judge_model": "claude-sonnet-4-6",
        "episodes": len(results),
        "wins": wins,
        "results": results,
        "verdict": verdict_str,
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nv1 wins: {wins['A_v1']}")
    print(f"v2 wins: {wins['B_v2']}")
    print(f"TIE: {wins['TIE']}")
    print(f"verdict: {verdict_str}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
