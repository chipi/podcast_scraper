"""Harvest cross-Person Insight pairs from a corpus for NLI-contradiction silver labeling.

Part of #1106 (RFC-088 ``nli_contradiction`` accuracy eval). Produces a stratified
JSONL of candidate pairs for offline silver labeling (two-lens: ``label`` +
``contradiction_type``). Deterministic given ``--seed``.

Two strata (see #1106 / the epic-1101 audit):

* ``precision`` — sampled from pairs the enricher **flagged**
  (``contradiction_score >= threshold``); measures "of what DeBERTa flags, how
  many are real".
* ``recall`` — sampled from cross-Person pairs the enricher did **not** flag;
  measures "of real contradictions, how many DeBERTa missed" — especially the
  Option-B *competing-claim* disagreements a strict-NLI model structurally can't
  catch.

No model is loaded here: flagged/unflagged is read from the committed
``enrichments/nli_contradiction.json`` envelope, so this runs on a ``.[dev]``
install (no ``[ml]`` extra, no re-scoring). Threshold-sweep scores come in a
later step once the ``[ml]`` env is confirmed.

Usage:
    python scripts/eval/score/enrichment_nli_harvest_v1.py \\
        --corpus .test_outputs/manual/prod-v2/corpus \\
        --out data/eval/enrichment/nli_contradiction/gold/harvest_prodv2_v1.jsonl \\
        --n-flagged 60 --n-unflagged 90 --seed 1106

Exit codes:
    0 — harvest written
    1 — corpus / enrichment output missing
    2 — invocation error
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.enrichers.nli_contradiction import (
    _episode_topic_insight_speaker_index,
)
from podcast_scraper.enrichment.paths import discover_episode_bundles


def _pair_key(topic_id: str, iid_a: str, iid_b: str) -> tuple[str, frozenset[str]]:
    """Order-independent key for a (topic, insight-pair)."""
    return (topic_id, frozenset({iid_a, iid_b}))


def _load_flagged(corpus_root: Path) -> dict[tuple[str, frozenset[str]], float]:
    """Read the enricher's committed output → {(topic, {ids}) -> contradiction_score}."""
    out = corpus_root / "enrichments" / "nli_contradiction.json"
    if not out.is_file():
        raise FileNotFoundError(f"missing enrichment output: {out}")
    envelope = json.loads(out.read_text(encoding="utf-8"))
    data = envelope.get("data") or {}
    flagged: dict[tuple[str, frozenset[str]], float] = {}
    for rec in data.get("contradictions", []):
        key = _pair_key(rec["topic_id"], rec["insight_a_id"], rec["insight_b_id"])
        flagged[key] = float(rec.get("contradiction_score", 0.0))
    return flagged


def _all_cross_person_pairs(
    corpus_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Rebuild every cross-Person Insight pair per topic (mirrors the enricher)."""
    bundles = discover_episode_bundles(corpus_root)
    by_topic, person_label = _episode_topic_insight_speaker_index(bundles)
    pairs: list[dict[str, Any]] = []
    for tid, entries in by_topic.items():
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                iid_a, pid_a, text_a = entries[i]
                iid_b, pid_b, text_b = entries[j]
                if pid_a == pid_b:
                    continue
                if not (text_a.strip() and text_b.strip()):
                    continue  # can't label an empty insight
                pairs.append(
                    {
                        "topic_id": tid,
                        "person_a_id": pid_a,
                        "person_a_name": person_label.get(pid_a, pid_a),
                        "person_b_id": pid_b,
                        "person_b_name": person_label.get(pid_b, pid_b),
                        "insight_a_id": iid_a,
                        "insight_b_id": iid_b,
                        "insight_a_text": text_a,
                        "insight_b_text": text_b,
                    }
                )
    return pairs, person_label


def _topic_diverse_sample(
    rng: random.Random, rows: list[dict[str, Any]], n: int, *, per_topic_cap: int = 2
) -> list[dict[str, Any]]:
    """Sample up to ``n`` rows, capping per-topic to keep topic diversity."""
    shuffled = rows[:]
    rng.shuffle(shuffled)
    taken: list[dict[str, Any]] = []
    per_topic: dict[str, int] = {}
    # First pass: respect the per-topic cap.
    for r in shuffled:
        if len(taken) >= n:
            break
        t = r["topic_id"]
        if per_topic.get(t, 0) >= per_topic_cap:
            continue
        per_topic[t] = per_topic.get(t, 0) + 1
        taken.append(r)
    # Backfill if the cap left us short.
    if len(taken) < n:
        remaining = [r for r in shuffled if r not in taken]
        taken.extend(remaining[: n - len(taken)])
    return taken


def _stratify_flagged_by_score(
    rng: random.Random,
    flagged_rows: list[dict[str, Any]],
    n: int,
) -> list[dict[str, Any]]:
    """Sample flagged pairs across the logit range, oversampling the 0.5 boundary."""
    boundary = [r for r in flagged_rows if r["deberta_score"] < 1.0]
    mid = [r for r in flagged_rows if 1.0 <= r["deberta_score"] < 2.5]
    high = [r for r in flagged_rows if r["deberta_score"] >= 2.5]
    # 50% boundary (that's where precision is decided), 30% mid, 20% high.
    quota = [(boundary, round(n * 0.5)), (mid, round(n * 0.3)), (high, n)]
    out: list[dict[str, Any]] = []
    for bucket, target in quota:
        need = min(target, n) - len(out) if bucket is quota[-1][0] else target
        need = max(0, min(need, len(bucket)))
        out.extend(_topic_diverse_sample(rng, bucket, need))
    return out[:n]


def main() -> int:
    """Harvest + stratify candidate pairs; write JSONL for silver labeling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-flagged", type=int, default=60)
    parser.add_argument("--n-unflagged", type=int, default=90)
    parser.add_argument("--seed", type=int, default=1106)
    args = parser.parse_args()

    corpus_root: Path = args.corpus
    if not corpus_root.is_dir():
        print(f"corpus not found: {corpus_root}", file=sys.stderr)
        return 1
    try:
        flagged = _load_flagged(corpus_root)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    all_pairs, _ = _all_cross_person_pairs(corpus_root)
    for r in all_pairs:
        key = _pair_key(r["topic_id"], r["insight_a_id"], r["insight_b_id"])
        score = flagged.get(key)
        r["deberta_flagged"] = score is not None
        r["deberta_score"] = score  # None for unflagged (below-threshold, not re-scored yet)

    flagged_rows = [r for r in all_pairs if r["deberta_flagged"]]
    unflagged_rows = [r for r in all_pairs if not r["deberta_flagged"]]

    rng = random.Random(args.seed)
    precision_sample = _stratify_flagged_by_score(rng, flagged_rows, args.n_flagged)
    recall_sample = _topic_diverse_sample(rng, unflagged_rows, args.n_unflagged)

    for r in precision_sample:
        r["stratum"] = "precision"
    for r in recall_sample:
        r["stratum"] = "recall"

    sample = precision_sample + recall_sample
    rng.shuffle(sample)  # de-bias labeler against stratum ordering

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(sample):
            r["pair_id"] = f"{args.seed}-{i:04d}"
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"harvested {len(sample)} pairs → {args.out}\n"
        f"  corpus cross-person pairs: {len(all_pairs)}  "
        f"(flagged {len(flagged_rows)} / unflagged {len(unflagged_rows)})\n"
        f"  precision stratum: {len(precision_sample)}  recall stratum: {len(recall_sample)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
