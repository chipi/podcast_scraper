"""Regenerate the app validation corpus's storylines (``enrichments/topic_theme_clusters.json``).

The builder authors the storylines (``build_app_validation_corpus.STORYLINES``) from each topic's
episodes, but a rebuild does not reproduce the committed corpus (ADR-165), so this applies the
builder's OWN functions to the committed files: each topic's episode ids from the episodes'
knowledge graphs, then ``_theme_clusters_data`` and ``_enrichment_envelope``. Dry run by
default; ``--write`` writes. It refuses when an existing storyline would change, so adding
one never moves another.

Usage:  .venv/bin/python scripts/tools/regen_fixture_storylines.py [--write]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "tests" / "fixtures" / "app-validation-corpus" / "v3"
TARGET = CORPUS / "enrichments" / "topic_theme_clusters.json"


def _builder():
    spec = importlib.util.spec_from_file_location(
        "build_app_validation_corpus", REPO / "scripts" / "build_app_validation_corpus.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_app_validation_corpus"] = mod
    spec.loader.exec_module(mod)
    return mod


def topic_episodes(corpus: Path) -> dict[str, list[str]]:
    """``topic id → episode ids`` from the committed episodes' knowledge graphs."""
    out: dict[str, set[str]] = {}
    for kg_path in sorted(corpus.rglob("*.kg.json")):
        meta_path = kg_path.with_name(kg_path.name.replace(".kg.json", ".metadata.json"))
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        episode_id = str(
            (meta.get("episode") or {}).get("episode_id") or meta.get("episode_id") or ""
        )
        if not episode_id:
            continue
        kg = json.loads(kg_path.read_text(encoding="utf-8"))
        for node in kg.get("nodes") or []:
            if isinstance(node, dict) and node.get("type") == "Topic" and node.get("id"):
                out.setdefault(str(node["id"]), set()).add(episode_id)
    return {t: sorted(e) for t, e in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="write the file (default: dry run)")
    args = ap.parse_args()
    builder = _builder()
    data = builder._theme_clusters_data(topic_episodes(CORPUS))
    envelope = builder._enrichment_envelope("topic_theme_clusters", data, version="1.1.0")
    old = json.loads(TARGET.read_text(encoding="utf-8"))
    old_by_id = {c["graph_compound_parent_id"]: c for c in old["data"]["clusters"]}
    for cluster in data["clusters"]:
        cid = cluster["graph_compound_parent_id"]
        before = old_by_id.get(cid)
        if before is not None and before != cluster:
            print(f"REFUSED: {cid} would change — adding storylines must not move existing ones")
            return 1
        members = ", ".join(
            f"{m['topic_id']} ({len(m['episode_ids'])} eps)" for m in cluster["members"]
        )
        print(f"{'kept ' if before else 'NEW  '} {cid} — {cluster['canonical_label']}: {members}")
    text = json.dumps(envelope, indent=2, sort_keys=True) + "\n"
    if args.write:
        TARGET.write_text(text, encoding="utf-8")
        print(f"wrote {TARGET.relative_to(REPO)}")
    else:
        print("dry run — pass --write to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
