"""Cleaning autoresearch sweep (#594).

For each (provider, model, temperature) cell, clean each input transcript via
direct API call using the canonical cleaning system + user prompt, and save
the output under `<output>/<provider>/<model>__t<temp>/<episode>.cleaned.txt`.

Computes two cheap per-cell metrics that don't need an LLM judge:
- `similarity_to_silver` — SequenceMatcher ratio of cleaned vs silver (higher = better)
- `sponsor_pattern_hits_cleaned` — residual SPONSOR_PATTERNS hits (lower = better)

Pairwise LLM judge runs separately (`scripts/eval/score/cleaning_judge_v1.py`).

Direct API calls (not provider class) so the sweep is decoupled from
LLMBasedCleaner's length-guard fallback — we're tuning the underlying model
choice, not the wrapper.

Usage:
    python scripts/eval/score/cleaning_sweep_v1.py \\
        --sources data/eval/sources/curated_5feeds_raw_v2 \\
        --silver  data/eval/references/silver/cleaning_v1 \\
        --episodes p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \\
        --output data/eval/runs/baseline_cleaning_autoresearch_v1
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from podcast_scraper.cleaning.commercial.patterns import SPONSOR_PATTERNS

CLEANING_SYSTEM = (
    "You are a transcript cleaning assistant. "
    "Remove sponsors, ads, intros, outros, and meta-commentary. "
    "Preserve all substantive content and speaker information. "
    "Return only the cleaned text, no explanations."
)

CLEANING_USER_TMPL = (
    "Clean the following podcast transcript. Apply the rules in the system "
    "prompt. Return ONLY the cleaned transcript text (no preamble, no markdown "
    "fences, no commentary).\n\n"
    "Transcript:\n```\n{transcript}\n```"
)

# Matrix — cloud only (Ollama daemon is user-managed per project convention).
MATRIX: list[tuple[str, str]] = [
    ("openai", "gpt-4o-mini"),
    ("openai", "gpt-4o"),
    ("anthropic", "claude-haiku-4-5"),
    ("gemini", "gemini-2.5-flash-lite"),
    ("gemini", "gemini-2.0-flash"),
    ("gemini", "gemini-2.5-flash"),
    ("deepseek", "deepseek-chat"),
]

TEMPS: list[float] = [0.0, 0.2, 0.4]


# ----- per-provider direct API callers ---------------------------------------


def _clean_openai(client: Any, model: str, transcript: str, temperature: float) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=8000,
        messages=[
            {"role": "system", "content": CLEANING_SYSTEM},
            {"role": "user", "content": CLEANING_USER_TMPL.format(transcript=transcript)},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def _clean_anthropic(client: Any, model: str, transcript: str, temperature: float) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=8000,
        temperature=temperature,
        system=CLEANING_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": CLEANING_USER_TMPL.format(transcript=transcript),
            }
        ],
    )
    return resp.content[0].text.strip()


def _clean_gemini(client: Any, model: str, transcript: str, temperature: float) -> str:
    from google.genai.types import GenerateContentConfig

    resp = client.models.generate_content(
        model=model,
        contents=CLEANING_USER_TMPL.format(transcript=transcript),
        config=GenerateContentConfig(
            system_instruction=CLEANING_SYSTEM,
            temperature=temperature,
            max_output_tokens=8000,
        ),
    )
    return (resp.text or "").strip()


def _clean_deepseek(client: Any, model: str, transcript: str, temperature: float) -> str:
    # OpenAI-compatible API
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=8000,
        messages=[
            {"role": "system", "content": CLEANING_SYSTEM},
            {"role": "user", "content": CLEANING_USER_TMPL.format(transcript=transcript)},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def build_clients() -> dict[str, Any]:
    clients: dict[str, Any] = {}
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI

        clients["openai"] = OpenAI()
    if os.environ.get("ANTHROPIC_API_KEY"):
        from anthropic import Anthropic

        clients["anthropic"] = Anthropic()
    if os.environ.get("GEMINI_API_KEY"):
        from google.genai import Client as GenaiClient

        clients["gemini"] = GenaiClient(api_key=os.environ["GEMINI_API_KEY"])
    if os.environ.get("DEEPSEEK_API_KEY"):
        from openai import OpenAI

        clients["deepseek"] = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"
        )
    return clients


CALLERS = {
    "openai": _clean_openai,
    "anthropic": _clean_anthropic,
    "gemini": _clean_gemini,
    "deepseek": _clean_deepseek,
}


# ----- metrics ---------------------------------------------------------------


def _sponsor_hits(text: str) -> int:
    return sum(sum(1 for _ in pat.pattern.finditer(text)) for pat in SPONSOR_PATTERNS)


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


# ----- main ------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--silver", type=Path, required=True)
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit-cells", type=int, default=0, help="0 = full matrix, N = first N cells")
    args = p.parse_args()

    clients = build_clients()
    missing = [prov for prov, _ in MATRIX if prov not in clients]
    if missing:
        print(
            f"WARNING: missing API keys for {sorted(set(missing))} — those cells skip",
            file=sys.stderr,
        )

    transcripts: dict[str, str] = {}
    silvers: dict[str, str] = {}
    for ep in args.episodes:
        srcs = list(args.sources.rglob(f"{ep}.txt"))
        if not srcs:
            print(f"  SKIP {ep}: no source", file=sys.stderr)
            continue
        sil = args.silver / f"{ep}.silver.txt"
        if not sil.exists():
            print(f"  SKIP {ep}: no silver", file=sys.stderr)
            continue
        transcripts[ep] = srcs[0].read_text(encoding="utf-8")
        silvers[ep] = sil.read_text(encoding="utf-8")

    cells = MATRIX[: args.limit_cells] if args.limit_cells else MATRIX
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, Any]] = []

    t0 = time.time()
    for provider, model in cells:
        if provider not in clients:
            continue
        for temp in TEMPS:
            slug = f"{provider}__{model.replace('/', '-').replace(':', '-')}__t{temp}"
            cell_dir = args.output / slug
            cell_dir.mkdir(parents=True, exist_ok=True)
            caller = CALLERS[provider]
            for ep in transcripts:
                t_cell = time.time()
                try:
                    cleaned = caller(clients[provider], model, transcripts[ep], temp)
                except Exception as exc:  # noqa: BLE001
                    metrics_rows.append(
                        {
                            "provider": provider,
                            "model": model,
                            "temperature": temp,
                            "episode_id": ep,
                            "error": str(exc),
                        }
                    )
                    print(f"  ERR {slug}/{ep}: {exc}", file=sys.stderr)
                    continue
                cleaned_path = cell_dir / f"{ep}.cleaned.txt"
                cleaned_path.write_text(cleaned, encoding="utf-8")
                raw = transcripts[ep]
                silver = silvers[ep]
                metrics_rows.append(
                    {
                        "provider": provider,
                        "model": model,
                        "temperature": temp,
                        "episode_id": ep,
                        "raw_chars": len(raw),
                        "cleaned_chars": len(cleaned),
                        "silver_chars": len(silver),
                        "cleaned_to_raw_pct": round(100 * len(cleaned) / max(len(raw), 1), 1),
                        "cleaned_to_silver_pct": round(100 * len(cleaned) / max(len(silver), 1), 1),
                        "similarity_to_silver": round(_similarity(cleaned, silver), 4),
                        "sponsor_hits_raw": _sponsor_hits(raw),
                        "sponsor_hits_cleaned": _sponsor_hits(cleaned),
                        "sponsor_hits_silver": _sponsor_hits(silver),
                        "latency_s": round(time.time() - t_cell, 2),
                    }
                )
            elapsed = time.time() - t0
            print(f"  [{elapsed:7.1f}s] {slug:55s} done")

    out_metrics = args.output / "metrics.jsonl"
    with out_metrics.open("w") as f:
        for row in metrics_rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nwrote {len(metrics_rows)} cell-episode rows -> {out_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
