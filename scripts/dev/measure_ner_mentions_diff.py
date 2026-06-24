#!/usr/bin/env python3
"""#1076 chunk 4-A validation — measure the NER pass's MENTIONS_PERSON
delta against a real corpus + produce a sample JSON for operator
TP/FP labelling.

Runs the typed-MENTIONS post-pass twice on a deep copy of each
``.gi.json`` (regex-only baseline + regex+NER), compares the edge sets,
and prints a summary table. For every NER-only edge, dumps the
``(insight_text_excerpt, kg_entity_name, spaCy_span)`` triple to a
JSON file so the operator can spot-check and label TP / FP.

DOES NOT mutate the on-disk artifacts. The corpus is read-only.

Usage:

    .venv/bin/python scripts/dev/measure_ner_mentions_diff.py \
        --corpus .test_outputs/manual/prod-v2/corpus \
        --sample-out /tmp/ner_fp_sample.json \
        --sample-size 30

The sample JSON has the shape:

    [
      {
        "episode_path": "feeds/.../metadata/0001 - foo.gi.json",
        "insight_id": "insight:abc",
        "insight_text": "...truncated...",
        "kg_entity_name": "Maya Hutchinson",
        "kg_entity_id": "person:maya-hutchinson",
        "spacy_span": "Maya",
        "operator_label": null
      },
      ...
    ]

Operator fills ``operator_label`` with "TP" / "FP" / "ambiguous" for
each row, then we tabulate the rate.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from podcast_scraper.gi.relational_edges import (  # noqa: E402
    add_insight_entity_edges,
    kg_entity_index,
)


def _gi_kg_pairs_under(corpus_root: Path):
    """Yield (gi_path, kg_path) for every .gi.json that has a sibling .kg.json."""
    for gi_path in corpus_root.rglob("*.gi.json"):
        kg_path = gi_path.with_suffix("").with_suffix(".kg.json")
        # The above strips .gi then .json; rebuild properly:
        kg_path = Path(str(gi_path).replace(".gi.json", ".kg.json"))
        if kg_path.is_file():
            yield gi_path, kg_path


def _flatten_artifact(raw: dict) -> dict:
    """Unwrap the legacy ``{"data": {"nodes": ..., "edges": ...}}`` envelope so
    the helpers see the flat shape they expect."""
    if "data" in raw and isinstance(raw["data"], dict):
        flat = {**raw, **raw["data"]}
        flat.pop("data", None)
        return flat
    return raw


def _mp_edge_keys(artifact: dict) -> set:
    """Return the set of (insight_id, person_id) tuples for MENTIONS_PERSON edges."""
    edges = artifact.get("edges") or []
    return {
        (e.get("from"), e.get("to"))
        for e in edges
        if isinstance(e, dict) and e.get("type") == "MENTIONS_PERSON"
    }


def _insight_text(artifact: dict, insight_id: str) -> str:
    for n in artifact.get("nodes") or []:
        if not isinstance(n, dict):
            continue
        if n.get("id") == insight_id and n.get("type") == "Insight":
            return ((n.get("properties") or {}).get("text") or "")[:300]
    return ""


def _person_name_from_kg(kg_artifact: dict, person_id: str) -> str:
    for n in kg_artifact.get("nodes") or []:
        if not isinstance(n, dict):
            continue
        if n.get("id") == person_id:
            return (n.get("properties") or {}).get("name") or ""
    return ""


def _spacy_span_for_insight(nlp, insight_text: str, entity_name: str) -> str:
    """Return the spaCy PERSON span text that triggered the match against
    ``entity_name`` (token-subset). Best-effort — used for operator
    spot-checking."""
    if not nlp or not insight_text:
        return ""
    name_tokens = {t.lower() for t in entity_name.split() if t}
    doc = nlp(insight_text)
    for ent in getattr(doc, "ents", []) or []:
        if getattr(ent, "label_", None) != "PERSON":
            continue
        span_text = (ent.text or "").strip()
        if len(span_text) < 3:
            continue
        span_tokens = {t for t in span_text.lower().replace(".", " ").split() if t}
        if span_tokens and span_tokens.issubset(name_tokens):
            return span_text
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--sample-out", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--max-files", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    if not args.corpus.is_dir():
        print(f"corpus not found: {args.corpus}", file=sys.stderr)
        return 2

    try:
        import spacy
    except ImportError:
        print("spaCy not installed in the venv", file=sys.stderr)
        return 2

    print("Loading en_core_web_sm...", flush=True)
    nlp = spacy.load("en_core_web_sm")

    total_files = 0
    total_regex = 0
    total_ner = 0
    files_with_delta = 0
    ner_only_rows = []

    for gi_path, kg_path in _gi_kg_pairs_under(args.corpus):
        if args.max_files and total_files >= args.max_files:
            break
        total_files += 1
        raw_gi = json.loads(gi_path.read_text(encoding="utf-8"))
        raw_kg = json.loads(kg_path.read_text(encoding="utf-8"))
        flat_gi = _flatten_artifact(raw_gi)
        flat_kg = _flatten_artifact(raw_kg)
        entity_index = kg_entity_index(flat_kg)
        if not entity_index:
            continue
        # Regex baseline on a deep copy so the disk artifact stays read-only.
        gi_regex = copy.deepcopy(flat_gi)
        add_insight_entity_edges(gi_regex, entity_index)
        gi_ner = copy.deepcopy(flat_gi)
        add_insight_entity_edges(gi_ner, entity_index, nlp=nlp)
        keys_regex = _mp_edge_keys(gi_regex)
        keys_ner = _mp_edge_keys(gi_ner)
        ner_only = keys_ner - keys_regex
        total_regex += len(keys_regex)
        total_ner += len(keys_ner)
        if ner_only:
            files_with_delta += 1
            for insight_id, person_id in ner_only:
                entity_name = _person_name_from_kg(flat_kg, person_id)
                insight_text = _insight_text(flat_gi, insight_id)
                ner_only_rows.append(
                    {
                        "episode_path": str(gi_path.relative_to(args.corpus)),
                        "insight_id": insight_id,
                        "insight_text": insight_text,
                        "kg_entity_id": person_id,
                        "kg_entity_name": entity_name,
                        "spacy_span": _spacy_span_for_insight(nlp, insight_text, entity_name),
                        "operator_label": None,
                    }
                )
        if total_files % 20 == 0:
            print(
                f"  scanned {total_files} files, NER-only delta so far: {len(ner_only_rows)}",
                flush=True,
            )

    # Sample for operator labelling — random subset.
    sample = (
        random.sample(ner_only_rows, args.sample_size)
        if len(ner_only_rows) > args.sample_size
        else ner_only_rows
    )
    args.sample_out.parent.mkdir(parents=True, exist_ok=True)
    args.sample_out.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")

    delta = total_ner - total_regex
    pct = (100 * delta / total_regex) if total_regex else 0.0
    print("\n=== Summary ===")
    print(f"Files scanned:                {total_files}")
    print(f"Files with NER-only delta:    {files_with_delta}")
    print(f"Regex baseline MP edges:      {total_regex}")
    print(f"Regex + NER MP edges:         {total_ner}")
    if total_regex:
        print(f"NER-only delta:               +{delta} ({pct:.1f}% over baseline)")
    print(f"Sampled {len(sample)} rows for operator labelling → {args.sample_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
