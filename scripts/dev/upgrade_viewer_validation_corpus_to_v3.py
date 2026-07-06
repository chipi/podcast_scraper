#!/usr/bin/env python3
"""#1075 chunk 2 — upgrade ``tests/fixtures/viewer-validation-corpus/v2/`` to v3.

Today every ``.gi.json`` in the validation corpus has Person nodes but
NO ``MENTIONS_PERSON`` / ``ABOUT`` edges and Insights without
``insight_type`` / ``position_hint``. The shipped Person Profile /
Position Tracker viewer surfaces read all four, so the fixture set
needs upgrading before it can exercise the new code paths.

This script walks every ``*.gi.json`` under the validation corpus and
does an in-place upgrade:

1. Bumps ``schema_version`` to ``"3.0"``.
2. Calls ``migrate_gi_document_v3`` to normalise existing ``MENTIONS``
   edges + ``insight_type`` vocab, fills required v3 defaults
   (``insight_type='unknown'``, ``position_hint=0.5``).
3. Calls ``add_insight_entity_edges`` against the Person nodes already
   in the artifact so each Insight gets ``MENTIONS_PERSON`` edges to
   every Person whose name appears in its text.
4. Adds ``ABOUT(Insight → Topic)`` edges using a deterministic
   rule: every Insight gets an ``ABOUT`` edge to every Topic already
   in the artifact (one episode, few topics, full coverage). Real
   pipelines use a topic encoder; for test fixtures we want predictable
   coverage so the viewer panels render every section.

Run from the repo root:

    .venv/bin/python scripts/dev/upgrade_viewer_validation_corpus_to_v3.py

Idempotent: re-running produces no further changes (the migration calls
de-dup edges and the v3 defaults survive the round-trip).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from podcast_scraper.gi.relational_edges import add_insight_entity_edges  # noqa: E402
from podcast_scraper.migrations.gil_kg_identity_migrations import (  # noqa: E402
    migrate_gi_document_v3,
)

_FIXTURES_VERSION = (
    (_REPO_ROOT / "tests" / "fixtures" / "FIXTURES_VERSION").read_text(encoding="utf-8").strip()
)
_CORPUS_DIR = _REPO_ROOT / "tests" / "fixtures" / "viewer-validation-corpus" / _FIXTURES_VERSION


def _build_entity_index(artifact: dict) -> dict[str, tuple[str, str]]:
    """Build the entity_index ``add_insight_entity_edges`` expects from
    Person nodes already present in the artifact.

    The function matches ``name`` against each Insight's text as a
    whole-word phrase; for our fixtures the LLM-extracted speaker names
    (``Maya``, ``Liam``, ...) appear verbatim in the source transcript
    slices we use as Insight text.
    """
    out: dict[str, tuple[str, str]] = {}
    nodes = artifact.get("data", {}).get("nodes") or artifact.get("nodes") or []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        if str(n.get("type")) != "Person":
            continue
        nid = n.get("id")
        props = n.get("properties") or {}
        name = props.get("name")
        if isinstance(nid, str) and isinstance(name, str) and name.strip():
            out[nid] = (name, "person")
    return out


def _add_about_edges_to_all_topics(artifact: dict) -> int:
    """Add ``ABOUT(Insight → Topic)`` edges connecting every Insight to
    every Topic already in the artifact. Deduplicates by (from, to).

    Returns the number of new edges added.
    """
    data = artifact.get("data") or {}
    nodes = data.get("nodes") if "data" in artifact else artifact.get("nodes", [])
    edges = data.get("edges") if "data" in artifact else artifact.get("edges", [])
    nodes_list = nodes if isinstance(nodes, list) else []
    edges_list = edges if isinstance(edges, list) else []

    insights = [
        n["id"]
        for n in nodes_list
        if isinstance(n, dict) and n.get("type") == "Insight" and isinstance(n.get("id"), str)
    ]
    topics = [
        n["id"]
        for n in nodes_list
        if isinstance(n, dict) and n.get("type") == "Topic" and isinstance(n.get("id"), str)
    ]
    existing = {
        (e.get("from"), e.get("to"))
        for e in edges_list
        if isinstance(e, dict) and e.get("type") == "ABOUT"
    }
    added = 0
    for insight_id in insights:
        for topic_id in topics:
            if (insight_id, topic_id) in existing:
                continue
            edges_list.append(
                {
                    "type": "ABOUT",
                    "from": insight_id,
                    "to": topic_id,
                    # Deterministic confidence — real pipelines use
                    # cosine similarity; for fixtures we set 0.5 so
                    # filtering / ranking still has a numeric to work
                    # with and ordering is stable.
                    "properties": {"confidence": 0.5},
                }
            )
            existing.add((insight_id, topic_id))
            added += 1
    return added


def upgrade_one(path: Path, nlp=None) -> tuple[int, int]:
    """Upgrade a single .gi.json file in place.

    When *nlp* is provided, ``add_insight_entity_edges`` runs the
    #1076 chunk 4-A spaCy NER pass in addition to the literal whole-
    word regex match. Used by the validation script to measure how
    many extra MENTIONS_PERSON edges the NER pass catches vs the
    regex baseline.

    Returns ``(mentions_person_added, about_added)``.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    # Stage 1 — wrap legacy "data" envelope to top-level shape that
    # ``migrate_gi_document_v3`` expects. The validation corpus uses
    # ``{"schema_version": ..., "data": {"nodes": [...], "edges": [...]}}``
    # while migrations work on the flat shape.
    wrapped = False
    if "data" in raw and isinstance(raw["data"], dict):
        flat = {**raw, **raw["data"]}
        # Drop the envelope key so the migration sees nodes/edges at top level.
        flat.pop("data", None)
        wrapped = True
    else:
        flat = raw

    # Stage 2 — strict schema_version normalization. Existing fixtures
    # store the integer-looking "2", which the strict v3 validator
    # rejects; the migration only handles "1.0" / "2.0" strings.
    if flat.get("schema_version") in ("2", 2):
        flat["schema_version"] = "2.0"
    if flat.get("schema_version") == "3" or flat.get("schema_version") == 3:
        flat["schema_version"] = "3.0"

    migrated = migrate_gi_document_v3(flat)

    # Stage 3 — MENTIONS_PERSON from Person names in Insight text.
    entity_index = _build_entity_index(migrated)
    mp_added = add_insight_entity_edges(migrated, entity_index, nlp=nlp)

    # Stage 4 — ABOUT edges across the cartesian Insight × Topic for
    # this episode (small, deterministic, exercises every viewer
    # rail-panel section).
    about_added = _add_about_edges_to_all_topics(migrated)

    # Re-wrap if input had the legacy envelope.
    if wrapped:
        nodes = migrated.pop("nodes", [])
        edges = migrated.pop("edges", [])
        migrated["data"] = {"nodes": nodes, "edges": edges}

    path.write_text(json.dumps(migrated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return mp_added, about_added


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--use-ner",
        action="store_true",
        help=(
            "Augment the regex-based typed-MENTIONS post-pass with the "
            "#1076 chunk 4-A spaCy NER pass. Catches BART-paraphrased "
            "name fragments (KG 'Maya Hutchinson' matched by Insight "
            "text 'Maya')."
        ),
    )
    args = parser.parse_args()

    if not _CORPUS_DIR.is_dir():
        print(f"Corpus dir not found: {_CORPUS_DIR}", file=sys.stderr)
        return 2
    files = sorted(_CORPUS_DIR.rglob("*.gi.json"))
    if not files:
        print(f"No .gi.json files under {_CORPUS_DIR}", file=sys.stderr)
        return 2

    nlp = None
    if args.use_ner:
        try:
            import spacy

            nlp = spacy.load("en_core_web_sm")
            print("spaCy en_core_web_sm loaded — NER pass ON")
        except Exception as exc:
            print(f"spaCy load failed: {exc}", file=sys.stderr)
            return 2

    total_mp = 0
    total_about = 0
    for path in files:
        mp, about = upgrade_one(path, nlp=nlp)
        total_mp += mp
        total_about += about
        print(f"  {path.relative_to(_REPO_ROOT)}  +{mp} MP  +{about} ABOUT")
    print(
        f"\nUpgraded {len(files)} files: +{total_mp} MENTIONS_PERSON edges, "
        f"+{total_about} ABOUT edges (NER {'ON' if nlp is not None else 'OFF'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
