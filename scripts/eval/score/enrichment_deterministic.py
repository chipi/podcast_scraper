"""Enrichment scoring — chunk-2 deterministic enrichers (RFC-088).

Exact-match against gold envelopes under
``data/eval/enrichment/deterministic/gold/``. The gold layout:

    data/eval/enrichment/deterministic/gold/
        topic_cooccurrence_corpus.gold.json
        temporal_velocity.gold.json
        ...

Each ``*.gold.json`` IS the expected envelope (same shape the executor
writes to disk). The script runs the named enricher against
``--corpus`` and byte-compares envelope.data against gold.data after
sorting any list rows so dict-key ordering doesn't trip the diff.

Direct-Python entry point (no Make wrapper) per REPLAN-O6.

Usage:
    python scripts/eval/score/enrichment_deterministic.py \\
        --corpus path/to/corpus \\
        --gold data/eval/enrichment/deterministic/gold \\
        [--enricher topic_cooccurrence_corpus,grounding_rate]

Exit codes:
    0 — every gold envelope present in corpus matches exactly (or no gold)
    1 — at least one mismatch (or corpus output missing for a gold envelope)
    2 — invocation error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _canonicalise(value: Any) -> Any:
    """Recursively sort list rows by their JSON-serialised form so the
    gold and corpus envelopes compare equal regardless of insertion order
    on the producer side. Dict ordering is normalised by json.dumps with
    ``sort_keys=True`` at the leaf comparison."""
    if isinstance(value, dict):
        return {k: _canonicalise(v) for k, v in value.items()}
    if isinstance(value, list):
        normalised = [_canonicalise(v) for v in value]
        try:
            return sorted(normalised, key=lambda r: json.dumps(r, sort_keys=True))
        except TypeError:
            return normalised
    return value


def _load_envelope(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _list_gold(gold_dir: Path) -> list[Path]:
    if not gold_dir.is_dir():
        return []
    return sorted(gold_dir.glob("*.gold.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=False)
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("data/eval/enrichment/deterministic/gold"),
    )
    parser.add_argument(
        "--enricher",
        type=str,
        default="",
        help="Comma-separated subset of enricher ids to score.",
    )
    args = parser.parse_args()

    gold_files = _list_gold(args.gold)
    if not gold_files:
        print(
            json.dumps(
                {
                    "status": "no_gold",
                    "gold_dir": str(args.gold),
                    "message": (
                        "No gold files yet. Drop *.gold.json fixtures under "
                        f"{args.gold} (one per enricher_id) to enable "
                        "exact-match scoring."
                    ),
                },
                indent=2,
            )
        )
        return 0

    if not args.corpus or not args.corpus.is_dir():
        print(
            json.dumps(
                {
                    "status": "no_corpus",
                    "message": "Pass --corpus pointing at the operator's enriched corpus.",
                }
            )
        )
        return 2

    only = {p.strip() for p in args.enricher.split(",") if p.strip()}
    per_enricher: list[dict[str, Any]] = []
    any_mismatch = False
    for gold_path in gold_files:
        enricher_id = (
            gold_path.stem[: -len(".gold")] if gold_path.stem.endswith(".gold") else gold_path.stem
        )
        if only and enricher_id not in only:
            continue
        gold = _load_envelope(gold_path)
        if gold is None or "data" not in gold:
            per_enricher.append({"enricher_id": enricher_id, "status": "gold_malformed"})
            any_mismatch = True
            continue
        corpus_path = args.corpus / "enrichments" / f"{enricher_id}.json"
        corpus = _load_envelope(corpus_path)
        if corpus is None:
            per_enricher.append(
                {
                    "enricher_id": enricher_id,
                    "status": "corpus_envelope_missing",
                    "expected": str(corpus_path),
                }
            )
            any_mismatch = True
            continue
        gold_data = _canonicalise(gold.get("data") or {})
        corpus_data = _canonicalise(corpus.get("data") or {})
        match = gold_data == corpus_data
        per_enricher.append(
            {
                "enricher_id": enricher_id,
                "status": "match" if match else "mismatch",
            }
        )
        if not match:
            any_mismatch = True

    print(
        json.dumps(
            {
                "status": "scored",
                "gold_dir": str(args.gold),
                "corpus": str(args.corpus),
                "per_enricher": per_enricher,
            },
            indent=2,
        )
    )
    return 1 if any_mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
