"""Semantic A/B eval for GI insights: does #1191 salience-ranking surface better insights?

Reconstructs BOTH arms from a SINGLE run's artifacts (no baseline re-run needed):
  - "without ranking" = top-N insights in EXTRACTION order (what the old cap kept)
  - "with ranking"    = top-N by SALIENCE (the #1191 view)

A cross-vendor Claude judge (Gemini generated the insights) scores each insight 1-5 on
SUBSTANCE — the thing the value-gate `tier` claims to measure — so it independently
validates the tier/salience that powers ranking.

Usage:
  python scripts/eval/insight_salience_eval.py <corpus_dir> \
      [--limit N] [--model claude-sonnet-5] [--out out.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

# Direct Anthropic call (the repo's Sonnet46Judge hardcodes temperature=0.0, which the
# claude-sonnet-5 reasoning model rejects with a 400). Keep the same shape, drop temperature.
_PRICE_IN, _PRICE_OUT = 3.00, 15.00  # $/Mtok, Sonnet-tier


class ClaudeJudge:
    def __init__(self, *, api_key: str, model: str) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self.model = model

    def score(self, prompt: str, *, max_tokens: int = 1200) -> "JudgeReply":
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            timeout=120.0,
        )
        parts = [getattr(b, "text", "") for b in (getattr(msg, "content", []) or [])]
        text = "".join(p for p in parts if p).strip()
        u = getattr(msg, "usage", None)
        pt = int(getattr(u, "input_tokens", 0) or 0) if u else 0
        ct = int(getattr(u, "output_tokens", 0) or 0) if u else 0
        cost = pt / 1e6 * _PRICE_IN + ct / 1e6 * _PRICE_OUT
        return JudgeReply(text=text, model=self._model, cost_usd=cost)


class JudgeReply:
    def __init__(self, *, text: str, model: str, cost_usd: float) -> None:
        self.text, self.model, self.cost_usd = text, model, cost_usd


RUBRIC = """
You are grading the SUBSTANCE of individual "insights" auto-extracted from a podcast episode.
An insight scores high when it is SPECIFIC, INFORMATIVE, and NON-OBVIOUS — it teaches a reader
something concrete about the episode's subject. It scores low when it is vague, generic, obvious,
or filler ("the guest shared their views", "technology is changing fast").

Score each insight from 1 to 5:
  5 = sharp, specific, non-obvious claim a knowledgeable reader would find genuinely informative
  4 = solid, specific, informative
  3 = real but somewhat generic or partly obvious
  2 = vague or mostly obvious; little information
  1 = filler / contentless / not an insight

Judge ONLY the insight text on its own terms. Do not reward length. Reply with a SINGLE JSON array,
one object per insight, in the SAME ORDER, no prose:
[{"id": <int>, "score": <1-5>, "reason": "<=12 words"}]
"""


def load_insights(corpus_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for f in sorted(glob.glob(os.path.join(corpus_dir, "**", "*.gi.json"), recursive=True)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        nodes = d.get("nodes", [])
        title = next(
            (n["properties"].get("title") for n in nodes if n.get("type") == "Episode"), None
        ) or d.get("episode_id", "?")
        ep = d.get("episode_id", f)
        idx = 0
        for n in nodes:
            if n.get("type") != "Insight":
                continue
            p = n.get("properties", {})
            rows.append(
                {
                    "episode": ep,
                    "title": title,
                    "extraction_idx": idx,  # node order == extraction order
                    "text": p.get("text", ""),
                    "tier": p.get("tier"),
                    "salience": p.get("salience"),
                    "routing_tag": p.get("routing_tag"),
                    "grounded": p.get("grounded"),
                }
            )
            idx += 1
    return rows


def _parse_scores(text: str, n: int) -> List[Optional[int]]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return [None] * n
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return [None] * n
    by_id = {}
    for o in arr:
        if isinstance(o, dict) and "id" in o and "score" in o:
            try:
                by_id[int(o["id"])] = int(o["score"])
            except Exception:
                pass
    return [by_id.get(i) for i in range(n)]


def judge_batch(batch: List[Dict[str, Any]], judge: ClaudeJudge) -> None:
    title = batch[0]["title"]
    lines = "\n".join(f'{i}. "{r["text"]}"' for i, r in enumerate(batch))
    prompt = f'{RUBRIC}\n\nEpisode: "{title}"\n\nInsights:\n{lines}\n'
    res = judge.score(prompt, max_tokens=1200)
    scores = _parse_scores(res.text, len(batch))
    for r, s in zip(batch, scores):
        r["judge_score"] = s
    batch[0]["_cost"] = res.cost_usd
    batch[0]["_model"] = res.model


def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None

    def ranks(vals: List[float]) -> List[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        rk = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk

    xr, yr = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    n = len(pairs)
    d2 = sum((a - b) ** 2 for a, b in zip(xr, yr))
    return 1 - (6 * d2) / (n * (n * n - 1))


def analyze(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in rows if r.get("judge_score") is not None]
    out: Dict[str, Any] = {"n_insights": len(rows), "n_scored": len(scored)}
    if not scored:
        return out

    # 1) does ranking track judge quality? correlate salience & tier vs judge
    out["spearman_salience_vs_judge"] = _spearman(
        [r["salience"] for r in scored], [r["judge_score"] for r in scored]
    )
    out["spearman_tier_vs_judge"] = _spearman(
        [r["tier"] for r in scored], [r["judge_score"] for r in scored]
    )

    # 2) tier vs judge: mean judge score per value-gate tier (does 3>2>1?)
    by_tier: Dict[Any, List[int]] = defaultdict(list)
    for r in scored:
        by_tier[r["tier"]].append(r["judge_score"])
    out["mean_judge_by_tier"] = {
        str(t): round(statistics.mean(v), 3)
        for t, v in sorted(by_tier.items(), key=lambda x: (x[0] is None, x[0]))
    }

    # 3) A/B: top-N by salience vs top-N by extraction order, per episode, averaged
    per_ep: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in scored:
        per_ep[r["episode"]].append(r)
    ab: Dict[str, Any] = {}
    for N in (3, 5, 8, 10):
        rank_scores, extr_scores, eps_used = [], [], 0
        for ep, items in per_ep.items():
            if len(items) < N:
                continue
            eps_used += 1
            by_sal = sorted(items, key=lambda r: (-(r["salience"] or 0)))[:N]
            by_ext = sorted(items, key=lambda r: r["extraction_idx"])[:N]
            rank_scores.append(statistics.mean(r["judge_score"] for r in by_sal))
            extr_scores.append(statistics.mean(r["judge_score"] for r in by_ext))
        if eps_used:
            ab[f"top{N}"] = {
                "episodes": eps_used,
                "with_ranking_avg": round(statistics.mean(rank_scores), 3),
                "without_ranking_avg": round(statistics.mean(extr_scores), 3),
                "delta": round(statistics.mean(rank_scores) - statistics.mean(extr_scores), 3),
            }
    out["ab_topN_salience_vs_extraction"] = ab

    # 4) de-truncation value: judge score of insights BEYOND a nominal old cap (idx>=N) vs within
    for N in (5, 8):
        within = [r["judge_score"] for r in scored if r["extraction_idx"] < N]
        beyond = [r["judge_score"] for r in scored if r["extraction_idx"] >= N]
        out[f"detrunc_beyond_idx{N}"] = {
            "within_n": len(within),
            "beyond_n": len(beyond),
            "within_avg": round(statistics.mean(within), 3) if within else None,
            "beyond_avg": round(statistics.mean(beyond), 3) if beyond else None,
        }
    out["overall_mean_judge"] = round(statistics.mean(r["judge_score"] for r in scored), 3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir")
    ap.add_argument("--limit", type=int, default=None, help="cap insights (for a cheap smoke test)")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 2
    judge = ClaudeJudge(api_key=key, model=args.model)

    rows = load_insights(args.corpus_dir)
    if args.limit:
        rows = rows[: args.limit]
    print(f"loaded {len(rows)} insights from {args.corpus_dir}", file=sys.stderr)
    if not rows:
        return 1

    # batch within an episode to keep the title stable per prompt
    per_ep: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        per_ep[r["episode"]].append(r)
    total_cost = 0.0
    for ep, items in per_ep.items():
        for i in range(0, len(items), args.batch):
            batch = items[i : i + args.batch]
            try:
                judge_batch(batch, judge)
                total_cost += batch[0].get("_cost", 0.0) or 0.0
            except Exception as e:  # noqa: BLE001
                print(f"  judge error ep={ep[:20]} batch@{i}: {e}", file=sys.stderr)
    report = analyze(rows)
    report["judge_model"] = args.model
    report["judge_cost_usd"] = round(total_cost, 4)
    print(json.dumps(report, indent=2))
    if args.out:
        json.dump({"report": report, "rows": rows}, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
