#!/usr/bin/env python3
"""Offline eval for personalized discovery ranking — the #1139 gate.

``rank_discover`` (``app_discover_view``) re-orders the shared catalog by a
signed-in user's interests when ``APP_PERSONALIZED_RANKING`` is on. The flag
ships **off** ("gated until the score is tuned"): there was no offline eval to
say personalization actually beats recency, so it reached no users. This is that
eval — the single gate that unblocks flipping the flag (LEARNING-PLATFORM gap 1
/ Pivot 2).

It runs the REAL path per seeded persona
(``tests/fixtures/ground-truth/v3/seeded_users/*.json``): seed a throwaway
per-user state dir with the persona's ``heard`` episodes (playback) + captured
topics (explicit interests), derive interests exactly as ``/discover`` does
(``get_interests`` ∪ ``derive_interests``), then rank the catalog two ways —

* **personalized** — ``rank_discover(root, interests, pool)``
* **recency** — ``rank_discover(root, [], pool)`` (the shipped default)

and score both against the persona's gold ``expected_relevant_shows`` (binary
relevance: an episode is relevant iff its show is in the gold set). The gate
metric is the **nDCG@K uplift** — how much personalization lifts relevant
episodes above the recency baseline. Uplift ≤ 0 means personalization isn't
earning its flag on this corpus (honest-negative); a clear positive uplift is
the evidence to turn it on.

Deterministic, no ML, no network. Writes ``gate_metrics.json`` under
``data/eval/rank_discover/<run_id>/`` mirroring the enrichment eval loop.

Run::

    python scripts/eval/score/rank_discover_v1.py
"""

from __future__ import annotations

import json
import math
import re
import tempfile
from pathlib import Path

from podcast_scraper.server import app_user_state
from podcast_scraper.server.app_content_source import build_catalog_rows_cumulative
from podcast_scraper.server.app_discover_view import rank_discover
from podcast_scraper.server.app_slugs import slug_for_row
from podcast_scraper.server.app_user_corpus import derive_interests

_ROOT = Path(__file__).resolve().parents[3]
_CORPUS = _ROOT / "tests" / "fixtures" / "app-validation-corpus" / "v3"
_USERS_DIR = _ROOT / "tests" / "fixtures" / "ground-truth" / "v3" / "seeded_users"

# top-K the /discover feed surfaces; the gate scores the head of the ranking.
_K = 10
# A fixed instant for the seeded playback rows (Date.now() is unavailable and
# would break determinism); only used for recency-of-update ordering, not ranking.
_SEED_TS = 1_700_000_000
# Personalization must lift relevant episodes at least this much (mean nDCG@K
# uplift over the personas) to justify flipping APP_PERSONALIZED_RANKING on.
_UPLIFT_MIN = 0.05
_NDCG_FLOOR = 0.5  # and the personalized ranking itself must clear this bar.

_LABEL_RE = re.compile(r"(p\d+_e\d+)")


def _label_to_slug(rows) -> dict[str, str]:
    """``p05_e04`` (fixture label) → catalog slug, via the KG artifact path."""
    out: dict[str, str] = {}
    for r in rows:
        m = _LABEL_RE.search(getattr(r, "kg_relative_path", "") or "")
        if m:
            out[m.group(1)] = slug_for_row(r)
    return out


def _show_of(slug: str) -> str:
    """Show prefix (``p05``) from a catalog slug (``p05-25cb576153``)."""
    return slug.split("-", 1)[0]


def _dcg(relevances: list[int]) -> float:
    """Discounted cumulative gain over a ranked list of binary relevances."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def _ndcg_at_k(ranked_shows: list[str], gold_shows: set[str], k: int) -> float:
    """nDCG@k with binary relevance (episode relevant iff its show ∈ gold)."""
    rels = [1 if s in gold_shows else 0 for s in ranked_shows[:k]]
    ideal = sorted(rels, reverse=True)
    idcg = _dcg(ideal)
    return _dcg(rels) / idcg if idcg > 0 else 0.0


def _precision_at_k(ranked_shows: list[str], gold_shows: set[str], k: int) -> float:
    rels = [1 for s in ranked_shows[:k] if s in gold_shows]
    return len(rels) / k if k else 0.0


def _seed_user_state(data_dir: Path, user: dict, label_to_slug: dict[str, str]) -> str:
    """Write the persona's heard playback + captured-topic interests; return user_id."""
    uid = str(user["user_id"])
    # Heard → a playback row past the "heard" threshold (position >> any duration).
    for label in user.get("heard", []):
        slug = label_to_slug.get(str(label))
        if slug:
            app_user_state.set_playback(data_dir, uid, slug, 10_000_000.0, _SEED_TS)
    # Captured topics → explicit interests (the picker/entity-card follow surface).
    captured = [str(t) for t in user.get("captured_topics", []) if t]
    if captured:
        app_user_state.set_interests(data_dir, uid, captured)
    return uid


def _interests_for(root: Path, data_dir: Path, uid: str) -> list[str]:
    """Explicit ∪ derived, exactly as the /discover route folds them (#1139)."""
    interests = list(app_user_state.get_interests(data_dir, uid))
    derived = derive_interests(root, data_dir, uid)
    return list(dict.fromkeys([*interests, *derived]))


def _score_user(corpus: Path, user: dict, rows, label_to_slug: dict[str, str], k: int) -> dict:
    """One persona's personalized-vs-recency scores against its gold shows."""
    gold_shows = {str(s) for s in user.get("expected_relevant_shows", [])}
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td)
        uid = _seed_user_state(data_dir, user, label_to_slug)
        interests = _interests_for(corpus, data_dir, uid)
        personalized = rank_discover(corpus, interests, rows, limit=k)
        recency = rank_discover(corpus, [], rows, limit=k)
    p_shows = [_show_of(s.slug) for s in personalized]
    r_shows = [_show_of(s.slug) for s in recency]
    row = {
        "user_id": uid,
        "persona": user.get("persona", ""),
        "gold_shows": sorted(gold_shows),
        "n_interests": len(interests),
        "personalized_ndcg": round(_ndcg_at_k(p_shows, gold_shows, k), 4),
        "recency_ndcg": round(_ndcg_at_k(r_shows, gold_shows, k), 4),
        "personalized_precision": round(_precision_at_k(p_shows, gold_shows, k), 4),
        "recency_precision": round(_precision_at_k(r_shows, gold_shows, k), 4),
    }
    row["ndcg_uplift"] = round(row["personalized_ndcg"] - row["recency_ndcg"], 4)
    return row


def evaluate(corpus: Path = _CORPUS, users_dir: Path = _USERS_DIR, k: int = _K) -> dict:
    """Pure eval: score every seeded persona and return ``{metrics, per_user}``.

    No printing, no file writes — the reusable core the CLI and the CI gate test
    both call. ``metrics.gate.pass`` is the flip-the-flag verdict.
    """
    rows = build_catalog_rows_cumulative(corpus)
    label_to_slug = _label_to_slug(rows)
    users = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(users_dir.glob("*.json"))]
    per_user = [
        _score_user(corpus, u, rows, label_to_slug, k)
        for u in users
        if u.get("expected_relevant_shows")
    ]
    n = len(per_user) or 1
    agg = {
        "k": k,
        "n_users": len(per_user),
        "mean_personalized_ndcg": round(sum(r["personalized_ndcg"] for r in per_user) / n, 4),
        "mean_recency_ndcg": round(sum(r["recency_ndcg"] for r in per_user) / n, 4),
        "mean_ndcg_uplift": round(sum(r["ndcg_uplift"] for r in per_user) / n, 4),
        "mean_personalized_precision": round(
            sum(r["personalized_precision"] for r in per_user) / n, 4
        ),
    }
    agg["gate"] = {
        "uplift_min": _UPLIFT_MIN,
        "ndcg_floor": _NDCG_FLOOR,
        "pass": (
            agg["mean_ndcg_uplift"] >= _UPLIFT_MIN and agg["mean_personalized_ndcg"] >= _NDCG_FLOOR
        ),
    }
    return {"metrics": agg, "per_user": per_user}


def main() -> int:
    result = evaluate()
    agg, per_user = result["metrics"], result["per_user"]
    print(f"=== rank_discover eval (K={_K}) — personalized vs recency ===")
    for row in per_user:
        print(
            f"  {row['user_id']:9} {row['persona']:28} gold={row['gold_shows']} "
            f"nDCG {row['recency_ndcg']:.3f}→{row['personalized_ndcg']:.3f} "
            f"(uplift {row['ndcg_uplift']:+.3f})  P@{_K} "
            f"{row['recency_precision']:.2f}→{row['personalized_precision']:.2f}"
        )
    gate_pass = agg["gate"]["pass"]
    print(
        f"\naggregate: mean nDCG {agg['mean_recency_ndcg']:.3f}→"
        f"{agg['mean_personalized_ndcg']:.3f} (uplift {agg['mean_ndcg_uplift']:+.3f}); "
        f"gate {'PASS' if gate_pass else 'FAIL'} "
        f"(need uplift ≥ {_UPLIFT_MIN}, nDCG ≥ {_NDCG_FLOOR})"
    )

    out_dir = _ROOT / "data" / "eval" / "rank_discover" / "rank_discover_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gate_metrics.json"
    out_path.write_text(
        json.dumps({"metrics": agg, "per_user": per_user}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
