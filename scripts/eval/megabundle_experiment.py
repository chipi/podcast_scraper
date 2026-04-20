"""Mega-bundle experiment: single LLM call for summary + GI + KG (#632).

Tests whether one structured JSON call can produce all extraction outputs
(title, summary, bullets, insights, topics, entities) without unacceptable
quality degradation vs three standalone calls.

Silver references (curated_5feeds_benchmark_v2):
  - Summary paragraph: data/eval/references/silver/silver_sonnet46_benchmark_v2_paragraph
  - GI insights (n=12): data/eval/references/silver/silver_sonnet46_gi_benchmark_v2
  - KG topics + entities: data/eval/references/silver/silver_sonnet46_kg_benchmark_v2

Scoring per field:
  - Summary paragraph: ROUGE-L F1 vs silver paragraph
  - GI insights: embedding coverage @ 0.65 threshold vs silver insight texts
  - KG topics: embedding coverage @ 0.65 threshold vs silver topic labels
  - KG entities: set F1 on normalized entity names

Success criteria (from #632 / MEGA_BUNDLE_EXPERIMENT_PLAN.md):
  Summary:    gpt-4o-mini standalone 0.469 -> acceptable >= 0.440
  GI (n=12):  82% coverage -> acceptable >= 75%
  KG topics:  71% coverage -> acceptable >= 65%
  Entities:   1.000 F1 -> acceptable >= 0.900

Keys (autoresearch convention; falls back to production env var):
  AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY   | OPENAI_API_KEY
  AUTORESEARCH_EXPERIMENT_GEMINI_API_KEY   | GEMINI_API_KEY
  AUTORESEARCH_EXPERIMENT_ANTHROPIC_API_KEY| ANTHROPIC_API_KEY

Usage:
    export AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY=sk-...
    python scripts/eval/megabundle_experiment.py \\
        --transcripts-dir data/eval/materialized/curated_5feeds_benchmark_v2 \\
        --silver-dir data/eval/references/silver \\
        --episodes p01_e03,p02_e03,p03_e03,p04_e03,p05_e03 \\
        --provider openai \\
        --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _REPO_ROOT / "data" / "eval" / "runs" / "_megabundle"

# Success criteria per field (baselines from v2 held-out eval on same 5 eps).
SUMMARY_BASELINE = 0.469  # gpt-4o-mini standalone paragraph, v2 held-out
SUMMARY_ACCEPTABLE = 0.440  # <6% drop
BULLETS_BASELINE = 0.396  # gpt-4o-mini standalone bullets, v2 held-out
BULLETS_ACCEPTABLE = 0.370  # <6% drop
GI_BASELINE_PCT = 82.0
GI_ACCEPTABLE_PCT = 75.0
KG_TOPIC_BASELINE_PCT = 71.0
KG_TOPIC_ACCEPTABLE_PCT = 65.0
ENTITY_ACCEPTABLE_F1 = 0.900

EMBED_THRESHOLD = 0.65


def _resolve_key(provider: str) -> Optional[str]:
    mapping = {
        "openai": ("AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY", "OPENAI_API_KEY"),
        "gemini": ("AUTORESEARCH_EXPERIMENT_GEMINI_API_KEY", "GEMINI_API_KEY"),
        "anthropic": ("AUTORESEARCH_EXPERIMENT_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
        "deepseek": ("AUTORESEARCH_EXPERIMENT_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY"),
        "mistral": ("AUTORESEARCH_EXPERIMENT_MISTRAL_API_KEY", "MISTRAL_API_KEY"),
        "grok": ("AUTORESEARCH_EXPERIMENT_GROK_API_KEY", "GROK_API_KEY"),
    }
    ar, prod = mapping[provider]
    return os.environ.get(ar) or os.environ.get(prod)


_OPENAI_COMPATIBLE_BASE_URLS: Dict[str, str] = {
    "deepseek": "https://api.deepseek.com",
    "grok": "https://api.x.ai/v1",
}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def build_megabundle_prompt(transcript: str) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt) for the mega-bundle call."""
    system = (
        "You are a podcast content analyzer. Given a podcast episode transcript, "
        "produce a SINGLE JSON object with exactly the fields specified below. "
        "Output valid JSON only — no commentary, no code fences."
    )
    user = (
        "From the transcript below, extract the following fields into one JSON "
        "object:\n\n"
        '  "title": string — concise episode title (10-15 words).\n'
        '  "summary": string — 4-6 paragraph prose summary, covering main '
        "arguments, guests, and conclusions.\n"
        '  "bullets": array of 4-6 strings — key takeaways as standalone sentences.\n'
        '  "insights": array of EXACTLY 12 objects, each '
        '{"text": string, "insight_type": "claim"|"fact"|"opinion"}. '
        "Insights must be grounded factual claims or strong opinions from the "
        "transcript, not summaries or filler.\n"
        '  "topics": array of EXACTLY 10 strings — distinct 2-8 word noun phrases '
        "capturing the episode's subject matter. Noun phrases only "
        "(e.g. 'passive index investing', NOT 'passive investing is better'). "
        "Topics must be unique.\n"
        '  "entities": array of up to 15 objects, each '
        '{"name": string, "kind": "person"|"org"|"place", '
        '"role": "host"|"guest"|"mentioned"}. '
        'Use "host"/"guest" only when the transcript clearly identifies the '
        'person as such; otherwise use "mentioned".\n\n'
        "Transcript:\n"
        "---\n"
        f"{transcript}\n"
        "---\n\n"
        "Output ONLY the JSON object."
    )
    return system, user


# ---------------------------------------------------------------------------
# Provider call
# ---------------------------------------------------------------------------


def call_megabundle(
    provider: str, model: str, api_key: str, transcript: str
) -> Tuple[Dict[str, Any], float, Optional[float]]:
    """Returns (parsed_json, wall_s, estimated_cost_usd_or_None)."""
    system, user = build_megabundle_prompt(transcript)
    t0 = time.time()
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        text = resp.choices[0].message.content or "{}"
        # gpt-4o-mini pricing approx: $0.15/1M input, $0.60/1M output
        usage = resp.usage
        cost = None
        if usage:
            cost = (
                usage.prompt_tokens / 1_000_000 * 0.15 + usage.completion_tokens / 1_000_000 * 0.60
            )
    elif provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=0.0,
        )
        text = resp.content[0].text if resp.content else "{}"
        # claude-haiku-4-5 pricing approx: $1/1M input, $5/1M output
        cost = None
        if hasattr(resp, "usage"):
            cost = (
                resp.usage.input_tokens / 1_000_000 * 1.0
                + resp.usage.output_tokens / 1_000_000 * 5.0
            )
    elif provider == "gemini":
        import google.genai as genai

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents=f"{system}\n\n{user}",
            config={
                "response_mime_type": "application/json",
                "temperature": 0.0,
                "max_output_tokens": 16384,
            },
        )
        text = resp.text or "{}"
        # gemini-2.5-flash-lite pricing very approx
        cost = None
    elif provider in ("deepseek", "grok", "mistral"):
        # OpenAI-compatible chat endpoint (DeepSeek/Grok via base_url swap;
        # Mistral via its SDK).
        if provider == "mistral":
            from mistralai import Mistral

            m_client = Mistral(api_key=api_key)
            resp = m_client.chat.complete(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            text = resp.choices[0].message.content or "{}"
            cost = None  # Mistral cost derivation TBD
        else:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=_OPENAI_COMPATIBLE_BASE_URLS[provider])
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.0,
            }
            if provider == "deepseek":
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or "{}"
            cost = None
    else:
        raise ValueError(f"Unknown provider: {provider}")
    dt = time.time() - t0

    # Parse JSON; strip code fences if present.
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Provider returned non-JSON: {e}\n---\n{text[:500]}") from e
    return parsed, round(dt, 1), round(cost, 6) if cost is not None else None


# ---------------------------------------------------------------------------
# Silver loaders
# ---------------------------------------------------------------------------


def load_silver_paragraph(silver_dir: Path, episode_id: str) -> Optional[str]:
    """Load silver paragraph summary. Silver uses key 'summary_final'."""
    pred_path = silver_dir / "silver_sonnet46_benchmark_v2_paragraph" / "predictions.jsonl"
    if not pred_path.exists():
        return None
    for line in pred_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("episode_id") == episode_id:
            out = row.get("output") or {}
            if isinstance(out, dict):
                return out.get("summary_final") or out.get("summary") or out.get("paragraph") or ""
            if isinstance(out, str):
                return out
    return None


def _load_silver_bullets_row(silver_dir: Path, episode_id: str) -> Optional[Dict[str, Any]]:
    """Return the decoded {title, bullets, ...} dict from the bullets silver row,
    or None. The bullets silver stores a JSON STRING at output.summary_final."""
    path = silver_dir / "silver_sonnet46_benchmark_v2_bullets" / "predictions.jsonl"
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("episode_id") == episode_id:
            out = row.get("output") or {}
            nested = out.get("summary_final")
            if isinstance(nested, str):
                try:
                    return json.loads(nested)
                except json.JSONDecodeError:
                    return None
            if isinstance(nested, dict):
                return nested
    return None


def load_silver_bullets(silver_dir: Path, episode_id: str) -> List[str]:
    inner = _load_silver_bullets_row(silver_dir, episode_id)
    if not inner:
        return []
    return [str(b).strip() for b in (inner.get("bullets") or []) if b]


def load_silver_title(silver_dir: Path, episode_id: str) -> str:
    inner = _load_silver_bullets_row(silver_dir, episode_id)
    if not inner:
        return ""
    return str(inner.get("title") or "").strip()


def load_silver_gi_insights(silver_dir: Path, episode_id: str) -> List[str]:
    path = silver_dir / "silver_sonnet46_gi_benchmark_v2" / "predictions.jsonl"
    if not path.exists():
        return []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("episode_id") == episode_id:
            insights = (row.get("output") or {}).get("insights") or []
            return [str(ins.get("text", "")).strip() for ins in insights if ins.get("text")]
    return []


def load_silver_kg(silver_dir: Path, episode_id: str) -> Tuple[List[str], List[str]]:
    """Return (topic_labels, entity_names)."""
    path = silver_dir / "silver_sonnet46_kg_benchmark_v2" / "predictions.jsonl"
    if not path.exists():
        return [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("episode_id") == episode_id:
            out = row.get("output") or {}
            topics = [
                str(t.get("label", t) if isinstance(t, dict) else t).strip()
                for t in (out.get("topics") or [])
            ]
            entities = [
                str(e.get("name", e) if isinstance(e, dict) else e).strip()
                for e in (out.get("entities") or [])
            ]
            return [t for t in topics if t], [e for e in entities if e]
    return [], []


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


def rouge_l_f1(hypothesis: str, reference: str) -> float:
    """Word-level ROUGE-L F1 using LCS."""
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    if not hyp or not ref:
        return 0.0
    # LCS length
    dp = [[0] * (len(ref) + 1) for _ in range(len(hyp) + 1)]
    for i, h in enumerate(hyp, 1):
        for j, r in enumerate(ref, 1):
            if h == r:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[len(hyp)][len(ref)]
    if lcs == 0:
        return 0.0
    p = lcs / len(hyp)
    r = lcs / len(ref)
    return 2 * p * r / (p + r)


def entity_set_f1(hyp_names: List[str], ref_names: List[str]) -> float:
    def norm(s: str) -> str:
        return " ".join(s.strip().lower().split())

    hyp = {norm(n) for n in hyp_names if n}
    ref = {norm(n) for n in ref_names if n}
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    tp = len(hyp & ref)
    if tp == 0:
        return 0.0
    p = tp / len(hyp)
    r = tp / len(ref)
    return 2 * p * r / (p + r)


def embedding_coverage(
    hyp_texts: List[str], ref_texts: List[str], threshold: float = EMBED_THRESHOLD
) -> Tuple[float, int, int]:
    """Fraction of refs matched by at least one hyp above cosine threshold.

    Returns (coverage_pct, matched_count, ref_count).
    """
    if not ref_texts:
        return 0.0, 0, 0
    if not hyp_texts:
        return 0.0, 0, len(ref_texts)
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("all-MiniLM-L6-v2")
    hyp_emb = model.encode(hyp_texts, convert_to_tensor=True, show_progress_bar=False)
    ref_emb = model.encode(ref_texts, convert_to_tensor=True, show_progress_bar=False)
    sim = util.cos_sim(ref_emb, hyp_emb)  # (n_ref, n_hyp)
    matched = 0
    for i in range(len(ref_texts)):
        if sim[i].max().item() >= threshold:
            matched += 1
    return matched / len(ref_texts) * 100.0, matched, len(ref_texts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Mega-bundle experiment (#632)")
    parser.add_argument("--transcripts-dir", required=True)
    parser.add_argument("--silver-dir", required=True, help="Root dir for silver references")
    parser.add_argument("--episodes", required=True)
    parser.add_argument(
        "--provider",
        required=True,
        choices=["openai", "anthropic", "gemini", "deepseek", "mistral", "grok"],
    )
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    key = _resolve_key(args.provider)
    if not key:
        sys.exit(f"No API key for {args.provider}")

    transcripts_dir = Path(args.transcripts_dir)
    silver_dir = Path(args.silver_dir)
    episodes = [e.strip() for e in args.episodes.split(",") if e.strip()]

    print(f"Provider/Model: {args.provider}/{args.model}")
    print(f"Episodes:       {episodes}\n", flush=True)

    results: List[Dict[str, Any]] = []
    total_cost = 0.0

    for ep in episodes:
        tpath = transcripts_dir / f"{ep}.txt"
        if not tpath.exists():
            print(f"  {ep}: SKIP (no transcript)", flush=True)
            continue
        transcript = tpath.read_text(encoding="utf-8")
        # Cap transcript input to avoid going over context (rough 25k-char cap).
        transcript = transcript[:25000]

        ref_para = load_silver_paragraph(silver_dir, ep) or ""
        ref_bullets = load_silver_bullets(silver_dir, ep)
        ref_title = load_silver_title(silver_dir, ep)
        ref_insights = load_silver_gi_insights(silver_dir, ep)
        ref_topics, ref_entities = load_silver_kg(silver_dir, ep)

        print(
            f"  {ep}: transcript={len(transcript)} chars, "
            f"refs: para={len(ref_para)} chars, bullets={len(ref_bullets)}, "
            f"gi={len(ref_insights)}, kg={len(ref_topics)} topics + {len(ref_entities)} ents",
            flush=True,
        )
        print(f"  {ep}: calling {args.provider}/{args.model}...", flush=True)
        try:
            parsed, wall_s, cost = call_megabundle(args.provider, args.model, key, transcript)
        except Exception as e:
            print(f"  {ep}: FAIL ({type(e).__name__}: {e})", flush=True)
            continue
        if cost:
            total_cost += cost

        # Extract fields (defensive on missing keys)
        hyp_summary = str(parsed.get("summary", "")).strip()
        hyp_bullets = parsed.get("bullets") or []
        hyp_insights = [
            str((i or {}).get("text", i) if isinstance(i, dict) else i).strip()
            for i in (parsed.get("insights") or [])
        ]
        hyp_topics = [str(t).strip() for t in (parsed.get("topics") or [])]
        hyp_entities = [
            str((e or {}).get("name", e) if isinstance(e, dict) else e).strip()
            for e in (parsed.get("entities") or [])
        ]
        hyp_insights = [x for x in hyp_insights if x]
        hyp_topics = [x for x in hyp_topics if x]
        hyp_entities = [x for x in hyp_entities if x]

        # Score
        hyp_title = str(parsed.get("title", "")).strip()
        title_rouge = rouge_l_f1(hyp_title, ref_title) if ref_title and hyp_title else None
        summ_rouge = rouge_l_f1(hyp_summary, ref_para) if ref_para else None
        # Bullets: ROUGE-L on joined-newline strings (matches v2 bullets scoring).
        if ref_bullets and hyp_bullets:
            bul_rouge = rouge_l_f1(
                "\n".join(str(b) for b in hyp_bullets),
                "\n".join(str(b) for b in ref_bullets),
            )
        else:
            bul_rouge = None
        gi_cov, gi_hit, gi_total = embedding_coverage(hyp_insights, ref_insights)
        kg_cov, kg_hit, kg_total = embedding_coverage(hyp_topics, ref_topics)
        ent_f1 = entity_set_f1(hyp_entities, ref_entities)

        row = {
            "episode": ep,
            "provider": args.provider,
            "model": args.model,
            "wall_s": wall_s,
            "cost_usd": cost,
            "counts": {
                "bullets": len(hyp_bullets),
                "insights": len(hyp_insights),
                "topics": len(hyp_topics),
                "entities": len(hyp_entities),
            },
            "scores": {
                "title_rouge_l": round(title_rouge, 4) if title_rouge is not None else None,
                "summary_rouge_l": round(summ_rouge, 4) if summ_rouge is not None else None,
                "bullets_rouge_l": round(bul_rouge, 4) if bul_rouge is not None else None,
                "gi_coverage_pct": round(gi_cov, 1),
                "gi_matched": f"{gi_hit}/{gi_total}",
                "kg_topic_coverage_pct": round(kg_cov, 1),
                "kg_topic_matched": f"{kg_hit}/{kg_total}",
                "entity_f1": round(ent_f1, 3),
            },
            "outputs": {
                "title": parsed.get("title", ""),
                "summary": hyp_summary,
                "bullets": hyp_bullets,
                "insights": hyp_insights,
                "topics": hyp_topics,
                "entities": hyp_entities,
            },
        }
        results.append(row)

        def _fmt(v: Optional[float], spec: str = ".3f") -> str:
            return format(v, spec) if v is not None else "—"

        print(
            f"  {ep}: title={_fmt(title_rouge)} summary={_fmt(summ_rouge)} "
            f"bullets={_fmt(bul_rouge)} gi={gi_cov:.1f}% kg_topics={kg_cov:.1f}% "
            f"ent={ent_f1:.3f} wall={wall_s}s cost=${cost or 0:.4f}",
            flush=True,
        )

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY vs success criteria")
    print("=" * 70)
    titles = [
        r["scores"]["title_rouge_l"] for r in results if r["scores"]["title_rouge_l"] is not None
    ]
    summ = [
        r["scores"]["summary_rouge_l"]
        for r in results
        if r["scores"]["summary_rouge_l"] is not None
    ]
    buls = [
        r["scores"]["bullets_rouge_l"]
        for r in results
        if r["scores"]["bullets_rouge_l"] is not None
    ]
    gis = [r["scores"]["gi_coverage_pct"] for r in results]
    kgs = [r["scores"]["kg_topic_coverage_pct"] for r in results]
    ents = [r["scores"]["entity_f1"] for r in results]

    if titles:
        tm = statistics.mean(titles)
        # Titles are short strings; ROUGE-L baseline noise band is wide. Report
        # without a pass/fail verdict — interpret in context.
        print(f"  Title (ROUGE-L):    avg {tm:.3f}  (short-string ROUGE-L — compare relative)")
    if summ:
        sm = statistics.mean(summ)
        verdict = "PASS" if sm >= SUMMARY_ACCEPTABLE else "BORDERLINE" if sm >= 0.420 else "FAIL"
        print(
            f"  Summary (ROUGE-L):  avg {sm:.3f}  "
            f"(bar {SUMMARY_ACCEPTABLE}; baseline {SUMMARY_BASELINE}) -> {verdict}"
        )
    if buls:
        bm = statistics.mean(buls)
        verdict = "PASS" if bm >= BULLETS_ACCEPTABLE else "FAIL"
        print(
            f"  Bullets (ROUGE-L):  avg {bm:.3f}  "
            f"(bar {BULLETS_ACCEPTABLE}; baseline {BULLETS_BASELINE}) -> {verdict}"
        )
    if gis:
        gm = statistics.mean(gis)
        verdict = "PASS" if gm >= GI_ACCEPTABLE_PCT else "FAIL"
        print(
            f"  GI coverage:        avg {gm:.1f}%  "
            f"(bar {GI_ACCEPTABLE_PCT}%; baseline {GI_BASELINE_PCT}%) -> {verdict}"
        )
    if kgs:
        km = statistics.mean(kgs)
        verdict = "PASS" if km >= KG_TOPIC_ACCEPTABLE_PCT else "FAIL"
        print(
            f"  KG topic coverage:  avg {km:.1f}%  "
            f"(bar {KG_TOPIC_ACCEPTABLE_PCT}%; baseline {KG_TOPIC_BASELINE_PCT}%) -> {verdict}"
        )
    if ents:
        em = statistics.mean(ents)
        verdict = "PASS" if em >= ENTITY_ACCEPTABLE_F1 else "FAIL"
        print(f"  Entity F1:          avg {em:.3f}  " f"(bar {ENTITY_ACCEPTABLE_F1}) -> {verdict}")
    print(f"\nTotal API cost (est.): ${total_cost:.4f}")

    # Write
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = _RESULTS_DIR / f"megabundle_{args.provider}_{args.model}_{ts}.json"
    out.write_text(
        json.dumps(
            {
                "provider": args.provider,
                "model": args.model,
                "episodes": episodes,
                "total_cost_usd": round(total_cost, 6),
                "results": results,
            },
            indent=2,
        )
    )
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
