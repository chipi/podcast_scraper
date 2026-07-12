#!/usr/bin/env python3
"""Corpus battle — judge two corpora's summaries/insights head-to-head, grounded in the transcript.

Built for the cloud-vs-DGX question (prod-v2 vs prod-v3), but takes any two corpora. Where
``compare_corpora_v1.py`` scores what a machine can check without an opinion, this scores the part
that needs one: is the summary *faithful* to the episode, does it *cover* it, does it read well.

Three things this harness takes seriously, because each has burned us before:

**Vendor bias is enforced, not documented.** Paired/graded judging over-indexes on stylistic
similarity, so a judge from a candidate's own vendor hands it a free boost (#939: a Qwen judge
crowned a Qwen candidate at rank 1 while cloud judges put it at rank 8). The candidates here are
Gemini (v2) and Qwen (v3), so a Qwen or Gemma judge is disqualified. ``_assert_cross_vendor``
refuses to run rather than emit a biased number.

**Each summary is judged against its own transcript.** v2 and v3 read *different* transcripts (the
v2 one is drifted and from a different ASR). Grading v3's summary against v2's transcript would
punish it for content it correctly reported. Faithfulness is only meaningful against the text the
summariser actually saw.

**The result is confounded, and says so.** A head-to-head of the shipped corpora measures
``transcript x LLM`` together. It answers "is the v3 corpus better?" — a real question — but NOT
"is the local LLM as good as the cloud one?". For that, run both LLMs over the *same* transcripts
(``--arm controlled``): same input, only the model differs.

Usage
-----

    # Arm A — as-shipped: is the v3 corpus a better artifact than v2?
    python scripts/eval/corpus_battle_v1.py \
        --a .test_outputs/manual/prod-v2/corpus --a-label cloud-gemini \
        --b .test_outputs/manual/prod-v3/corpus --b-label dgx-qwen35 \
        --episodes 20 --out data/eval/corpus_battle/runs/<run_id>

Exit codes: 0 ok · 2 input/config error (including a vendor-bias violation)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

# Reasoning models emit their chain-of-thought before the answer, and those digits match a naive
# score regex. Always cut everything up to the end of the thinking block before parsing.
_THINK = re.compile(r"(?is).*?</think\s*>")
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)

# vendor -> the model-name fragments that belong to it
_VENDORS: Dict[str, Tuple[str, ...]] = {
    "alibaba": ("qwen",),
    "google": ("gemini", "gemma"),
    "openai": ("gpt", "openai", "o1", "o3"),
    "anthropic": ("claude", "sonnet", "haiku", "opus"),
    "meta": ("llama",),
    "nvidia": ("nemotron",),
    "mistral": ("mistral", "ministral", "magistral"),
    "deepseek": ("deepseek",),
}

_RUBRIC = """You are grading a podcast episode summary against the transcript it was written from.

Score each dimension from 1 (terrible) to 5 (excellent):
- faithfulness: every claim is supported by the transcript; no invented facts, names, or numbers
- coverage: the summary captures the episode's main points, not just its opening
- coherence: it reads as a structured whole, not a list of disconnected fragments
- fluency: clear, well-formed English

Respond with ONLY a JSON object, no other text:
{"faithfulness": <1-5>, "coverage": <1-5>, "coherence": <1-5>, "fluency": <1-5>,
 "note": "<one short sentence>"}"""


def _vendor_of(model: str) -> Optional[str]:
    m = model.lower()
    for vendor, frags in _VENDORS.items():
        if any(f in m for f in frags):
            return vendor
    return None


def _assert_cross_vendor(judges: List[str], candidates: List[str]) -> None:
    """Refuse to run a judge that shares a vendor with any candidate (see #939)."""
    cand_vendors = {v for v in (_vendor_of(c) for c in candidates) if v}
    for judge in judges:
        jv = _vendor_of(judge)
        if jv and jv in cand_vendors:
            raise SystemExit(
                f"error: judge {judge!r} is vendor '{jv}', which is also a candidate's vendor "
                f"({sorted(cand_vendors)}). Same-vendor judging hands that candidate a free style "
                f"boost — pick a judge from a disjoint vendor (see autoresearch/JUDGING.md)."
            )


def _parse_scores(raw: str) -> Optional[Dict[str, float]]:
    """Pull the score object out of a judge reply, tolerating reasoning blocks and prose."""
    text = _THINK.sub("", raw or "", count=1)
    match = _JSON_OBJ.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    out: Dict[str, float] = {}
    for dim in ("faithfulness", "coverage", "coherence", "fluency"):
        val = obj.get(dim)
        if isinstance(val, (int, float)) and 1 <= float(val) <= 5:
            out[dim] = float(val)
    return out or None


def _call_ollama(host: str, model: str, prompt: str, num_ctx: int) -> str:
    resp = httpx.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": num_ctx},
        },
        timeout=900.0,
    )
    resp.raise_for_status()
    return str(resp.json().get("response", ""))


def _call_openai(model: str, prompt: str) -> str:
    from openai import OpenAI

    reply = OpenAI().chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return reply.choices[0].message.content or ""


def _call_anthropic(model: str, prompt: str) -> str:
    from anthropic import Anthropic

    reply = Anthropic().messages.create(
        model=model, max_tokens=1024, messages=[{"role": "user", "content": prompt}]
    )
    return "".join(block.text for block in reply.content if getattr(block, "type", "") == "text")


def _judge(
    judge: str, transcript: str, summary: str, *, ollama_host: str, num_ctx: int
) -> Optional[Dict[str, float]]:
    """Grade one summary with one judge. ``judge`` is ``provider:model``."""
    provider, _, model = judge.partition(":")
    prompt = (
        f"{_RUBRIC}\n\n=== TRANSCRIPT ===\n{transcript}\n\n=== SUMMARY TO GRADE ===\n{summary}\n"
    )
    try:
        if provider == "ollama":
            raw = _call_ollama(ollama_host, model, prompt, num_ctx)
        elif provider == "openai":
            raw = _call_openai(model, prompt)
        elif provider == "anthropic":
            raw = _call_anthropic(model, prompt)
        else:
            raise ValueError(f"unknown judge provider {provider!r} (use ollama|openai|anthropic)")
        return _parse_scores(raw)
    except Exception as exc:  # noqa: BLE001 — one bad judge call must not kill the run
        print(f"    judge {judge} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def _latest_runs(corpus: Path) -> List[Path]:
    runs = []
    for feed in sorted(corpus.glob("feeds/*")):
        feed_runs = sorted(feed.glob("run_*"))
        if feed_runs:
            runs.append(feed_runs[-1])
    return runs


def _episodes(corpus: Path) -> Dict[str, Dict[str, Any]]:
    """guid -> {title, summary, transcript} from the newest run of each feed."""
    out: Dict[str, Dict[str, Any]] = {}
    for run in _latest_runs(corpus):
        for meta_path in sorted((run / "metadata").glob("*.metadata.json")):
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            episode = meta.get("episode") or {}
            guid = episode.get("guid")
            if not guid:
                continue
            summary = (meta.get("content") or {}).get("summary") or meta.get("summary")
            stem = meta_path.name[: -len(".metadata.json")]
            transcript_path = run / "transcripts" / f"{stem}.txt"
            if not summary or not transcript_path.is_file():
                continue
            out[guid] = {
                "title": episode.get("title", ""),
                "summary": str(summary),
                "transcript": transcript_path.read_text(encoding="utf-8", errors="replace"),
            }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--a-label", default="A")
    ap.add_argument("--b-label", default="B")
    ap.add_argument("--a-model", required=True, help="Model that wrote A's summaries (bias check)")
    ap.add_argument("--b-model", required=True, help="Model that wrote B's summaries (bias check)")
    ap.add_argument(
        "--judges",
        nargs="+",
        default=["anthropic:claude-sonnet-4-6", "openai:gpt-5.4"],
        help=(
            "Judges as provider:model (ollama|openai|anthropic). Default is the cloud flagship "
            "panel — vendor-disjoint from both Gemini and Qwen, and the same pair that decided "
            "#928. Cloud judges cost money but do not touch the GPU, so they can run while the "
            "DGX is still building the corpus."
        ),
    )
    ap.add_argument("--ollama-host", default="http://dgx-llm-1:11434")
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--episodes", type=int, default=20, help="How many shared episodes to judge")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, help="Directory to write results into")
    args = ap.parse_args(argv)

    # The rule, enforced before a single token is spent.
    _assert_cross_vendor(args.judges, [args.a_model, args.b_model])

    eps_a, eps_b = _episodes(args.a), _episodes(args.b)
    shared = sorted(set(eps_a) & set(eps_b))
    if not shared:
        print("error: no shared episodes with both a summary and a transcript", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    rng.shuffle(shared)
    picked = shared[: args.episodes]
    print(f"judging {len(picked)} of {len(shared)} shared episodes")
    print(f"  candidates: {args.a_label} ({args.a_model}) vs {args.b_label} ({args.b_model})")
    print(f"  judges:     {', '.join(args.judges)} (cross-vendor: verified)")

    results: List[Dict[str, Any]] = []
    for i, guid in enumerate(picked, 1):
        a, b = eps_a[guid], eps_b[guid]
        print(f"[{i}/{len(picked)}] {a['title'][:56]}", flush=True)
        row: Dict[str, Any] = {"guid": guid, "title": a["title"], "judges": {}}
        for judge in args.judges:
            # Each summary is graded against the transcript ITS OWN pipeline produced.
            sa = _judge(args.ollama_host, judge, a["transcript"], a["summary"], args.num_ctx)
            sb = _judge(args.ollama_host, judge, b["transcript"], b["summary"], args.num_ctx)
            row["judges"][judge] = {args.a_label: sa, args.b_label: sb}
        results.append(row)

    # Aggregate: mean per dimension, per judge, per side.
    print("\n=== scores (1-5, higher is better) ===")
    summary_rows = []
    for judge in args.judges:
        for label in (args.a_label, args.b_label):
            dims: Dict[str, List[float]] = {}
            for row in results:
                scores = (row["judges"].get(judge) or {}).get(label)
                if not scores:
                    continue
                for dim, val in scores.items():
                    dims.setdefault(dim, []).append(val)
            if not dims:
                continue
            means = {d: statistics.fmean(v) for d, v in dims.items()}
            overall = statistics.fmean(means.values())
            n = len(next(iter(dims.values())))
            summary_rows.append((judge, label, means, overall, n))
            dim_str = "  ".join(f"{d[:4]}={m:.2f}" for d, m in sorted(means.items()))
            print(f"  {judge:26} {label:16} {dim_str}   overall={overall:.2f}  (n={n})")

    payload = {
        "candidates": {args.a_label: args.a_model, args.b_label: args.b_model},
        "judges": args.judges,
        "episodes_judged": len(picked),
        "per_episode": results,
        "aggregate": [
            {"judge": j, "side": s, "dims": m, "overall": o, "n": n}
            for j, s, m, o, n in summary_rows
        ],
        "caveat": (
            "As-shipped arm: each side's summary was written from its own transcript, so this "
            "measures transcript x LLM together, not the LLM alone. To isolate the model, run both "
            "LLMs over the SAME transcripts."
        ),
    }
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "battle.json").write_text(json.dumps(payload, indent=2))
        print(f"\nwritten: {args.out / 'battle.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
