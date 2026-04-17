#!/usr/bin/env python3
"""Re-parse ``summary`` in ``*.metadata.json`` after fixing markdown JSON fences.

Some LLMs (e.g. Gemini) return `` ```json { ... } ``` `` on a single line. Older
parsers left the opening fence in place, produced ``schema_status: degraded``,
and polluted downstream GI/KG. After upgrading ``parse_summary_output``, run this
on an existing corpus output tree to refresh ``summary`` without re-summarizing.

With ``--patch-graphs``, also rewrites adjacent ``*.gi.json`` and (when safe)
``*.kg.json`` from the **current** ``summary.bullets`` in memory, then rebuilds
``*.bridge.json``. No transcript re-download and no new LLM summarization calls;
GI Quote nodes and grounding edges are left unchanged (only Insight text and
Topic/ABOUT are synced for GI; KG Topic nodes are replaced only for bullet-derived
or visibly corrupt topic labels).

Usage (from repo root)::

    python scripts/tools/repair_fenced_summary_metadata.py path/to/run/root
    python scripts/tools/repair_fenced_summary_metadata.py path/to/run --dry-run
    python scripts/tools/repair_fenced_summary_metadata.py path/to/run --force
    python scripts/tools/repair_fenced_summary_metadata.py path/to/run --include-degraded
    python scripts/tools/repair_fenced_summary_metadata.py path/to/run --patch-graphs
    python scripts/tools/repair_fenced_summary_metadata.py path/to/run --patch-graphs --dry-run

``--force`` attempts ``parse_summary_output`` on every file that has a
``summary`` object (use with care).

Default mode only touches summaries whose ``raw_text`` or ``bullets`` contain a
markdown code fence (`` ``` ``), i.e. the Gemini-style fenced JSON bug.

``--include-degraded`` additionally re-parses any ``schema_status`` of
``degraded`` or ``invalid`` that still has non-empty ``raw_text`` (broader;
use on a copy of the corpus).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple
from unittest.mock import Mock

try:
    from podcast_scraper.builders.bridge_builder import build_bridge
    from podcast_scraper.builders.rfc072_artifact_paths import (
        bridge_json_path_adjacent_to_metadata,
    )
    from podcast_scraper.gi.io import write_artifact as gi_write_artifact
    from podcast_scraper.kg.io import write_artifact as kg_write_artifact
    from podcast_scraper.schemas.summary_schema import parse_summary_output
    from podcast_scraper.utils.corpus_graph_bullet_sync import (
        bullet_labels_from_summary_bullets,
        patch_gi_for_bullet_labels,
        patch_kg_for_bullet_labels,
    )
except ImportError:
    _root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(_root / "src"))
    from podcast_scraper.builders.bridge_builder import build_bridge
    from podcast_scraper.builders.rfc072_artifact_paths import (
        bridge_json_path_adjacent_to_metadata,
    )
    from podcast_scraper.gi.io import write_artifact as gi_write_artifact
    from podcast_scraper.kg.io import write_artifact as kg_write_artifact
    from podcast_scraper.schemas.summary_schema import parse_summary_output
    from podcast_scraper.utils.corpus_graph_bullet_sync import (
        bullet_labels_from_summary_bullets,
        patch_gi_for_bullet_labels,
        patch_kg_for_bullet_labels,
    )


def _short_summary_from_bullets(bullets: list[str], raw_fallback: str | None) -> str:
    """Match ``SummaryMetadata.short_summary`` computed field."""
    if bullets:
        if len(bullets) == 1:
            return bullets[0]
        return " ".join(bullets[:2])
    return raw_fallback or ""


def _summary_source_text(summ: Dict[str, Any]) -> str:
    raw = summ.get("raw_text")
    if isinstance(raw, str) and raw.strip():
        return raw
    bullets = summ.get("bullets")
    if isinstance(bullets, list) and bullets:
        return "\n".join(str(b) for b in bullets)
    return ""


def _should_attempt_repair(summ: Dict[str, Any], *, force: bool, include_degraded: bool) -> bool:
    if force:
        return True
    raw = summ.get("raw_text")
    if isinstance(raw, str) and "```" in raw:
        return True
    bullets = summ.get("bullets")
    if isinstance(bullets, list) and any(isinstance(b, str) and "```" in b for b in bullets):
        return True
    if include_degraded:
        if (
            summ.get("schema_status") in ("degraded", "invalid")
            and isinstance(raw, str)
            and raw.strip()
        ):
            return True
    return False


def _repair_one_summary(
    summ: Dict[str, Any],
    episode_title: str | None,
    *,
    force: bool,
    include_degraded: bool,
) -> Tuple[bool, str]:
    if not isinstance(summ, dict):
        return False, "skip: summary not an object"
    if "bullets" not in summ and not force:
        return False, "skip: no bullets key"
    if not _should_attempt_repair(summ, force=force, include_degraded=include_degraded):
        return False, "skip: no ``` fence (try --include-degraded or --force)"

    source = _summary_source_text(summ).strip()
    if not source:
        return False, "skip: empty source text"

    result = parse_summary_output(source, Mock(), episode_title=episode_title)
    if not result.success or result.schema is None:
        return False, f"skip: parse failed ({result.error or 'unknown'})"

    schema = result.schema
    summ["title"] = schema.title
    summ["bullets"] = schema.bullets
    summ["key_quotes"] = schema.key_quotes
    summ["named_entities"] = schema.named_entities
    summ["timestamps"] = schema.timestamps
    summ["schema_status"] = schema.status
    summ["raw_text"] = schema.raw_text
    summ["short_summary"] = _short_summary_from_bullets(schema.bullets, schema.raw_text)
    return True, "updated"


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _artifact_paths(metadata_path: Path) -> tuple[Path, Path, Path]:
    s = str(metadata_path)
    if not s.endswith(".metadata.json"):
        raise ValueError(f"expected *.metadata.json, got {metadata_path!s}")
    base = s[: -len(".metadata.json")]
    gi = Path(base + ".gi.json")
    kg = Path(base + ".kg.json")
    bridge = Path(bridge_json_path_adjacent_to_metadata(s))
    return gi, kg, bridge


def sync_downstream_graphs(
    metadata_path: Path,
    doc: Dict[str, Any],
    *,
    dry_run: bool,
) -> Dict[str, int]:
    """Patch GI/KG and rebuild bridge from ``doc`` summary bullets. Returns counts."""
    counts = {"gi": 0, "kg": 0, "bridge": 0}
    summ = doc.get("summary")
    labels = bullet_labels_from_summary_bullets(
        summ.get("bullets") if isinstance(summ, dict) else None
    )
    if not labels:
        return counts

    ep = doc.get("episode")
    episode_id = ep.get("episode_id") if isinstance(ep, dict) else None
    if not isinstance(episode_id, str) or not episode_id.strip():
        print(f"{metadata_path}: skip-graphs: missing episode.episode_id", file=sys.stderr)
        return counts

    gi_path, kg_path, bridge_path = _artifact_paths(metadata_path)

    gi_payload: Dict[str, Any] | None = None
    kg_payload: Dict[str, Any] | None = None

    if gi_path.is_file():
        try:
            gi_payload = json.loads(gi_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"{metadata_path}: skip-gi read: {e}", file=sys.stderr)
            gi_payload = None

    if kg_path.is_file():
        try:
            kg_payload = json.loads(kg_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"{metadata_path}: skip-kg read: {e}", file=sys.stderr)
            kg_payload = None

    if gi_payload is None and kg_payload is None:
        return counts

    gi_out = gi_payload
    if gi_payload is not None:
        patched, msg = patch_gi_for_bullet_labels(gi_payload, labels)
        if patched is not None:
            if _canonical_json(patched) != _canonical_json(gi_payload):
                gi_out = patched
                print(f"{metadata_path}: {msg}")
                if not dry_run:
                    gi_write_artifact(gi_path, patched, validate=True)
                counts["gi"] = 1
        else:
            print(f"{metadata_path}: {msg}")

    kg_out = kg_payload
    if kg_payload is not None:
        patched_kg, msg_k = patch_kg_for_bullet_labels(kg_payload, labels)
        if patched_kg is not None:
            if _canonical_json(patched_kg) != _canonical_json(kg_payload):
                kg_out = patched_kg
                print(f"{metadata_path}: {msg_k}")
                if not dry_run:
                    kg_write_artifact(kg_path, patched_kg, validate=True)
                counts["kg"] = 1
        else:
            print(f"{metadata_path}: {msg_k}")

    if gi_out is not None or kg_out is not None:
        bridge_doc = build_bridge(episode_id.strip(), gi_out, kg_out)
        if dry_run:
            print(f"{metadata_path}: would write bridge -> {bridge_path}")
        else:
            bridge_path.parent.mkdir(parents=True, exist_ok=True)
            bridge_path.write_text(
                json.dumps(bridge_doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            counts["bridge"] = 1

    return counts


def iter_metadata_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.metadata.json"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-parse summary fields in *.metadata.json (markdown JSON fence repair)."
    )
    parser.add_argument(
        "corpus_root",
        type=Path,
        help="Root directory to scan (recursive) for *.metadata.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only; do not write files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Try parse_summary_output on every document that has a summary object",
    )
    parser.add_argument(
        "--include-degraded",
        action="store_true",
        help="Also re-parse degraded/invalid summaries with non-empty raw_text (broad)",
    )
    parser.add_argument(
        "--patch-graphs",
        action="store_true",
        help="Sync *.gi.json / *.kg.json / *.bridge.json from summary bullets (no LLM)",
    )
    args = parser.parse_args()
    root: Path = args.corpus_root
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2

    updated = 0
    skipped = 0
    errors = 0
    patched_gi = patched_kg = patched_bridge = 0
    for path in iter_metadata_files(root):
        try:
            text = path.read_text(encoding="utf-8")
            doc = json.loads(text)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"ERR {path}: {e}", file=sys.stderr)
            errors += 1
            continue

        summ = doc.get("summary")
        if summ is None:
            skipped += 1
            continue

        ep = doc.get("episode")
        episode_title = ep.get("title") if isinstance(ep, dict) else None

        changed, reason = _repair_one_summary(
            summ,
            episode_title,
            force=args.force,
            include_degraded=args.include_degraded,
        )
        if not changed:
            skipped += 1
            if args.force or "skip: no ``` fence" not in reason:
                print(f"{path}: {reason}")
        else:
            print(f"{path}: {reason}")
            updated += 1
            if not args.dry_run:
                path.write_text(
                    json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

        if args.patch_graphs:
            c = sync_downstream_graphs(path, doc, dry_run=args.dry_run)
            patched_gi += c["gi"]
            patched_kg += c["kg"]
            patched_bridge += c["bridge"]

    print(
        f"Done. metadata_updated={updated} skipped={skipped} errors={errors} "
        f"dry_run={args.dry_run} force={args.force} "
        f"include_degraded={args.include_degraded} "
        f"patch_graphs={args.patch_graphs} "
        f"(gi_files={patched_gi} kg_files={patched_kg} bridge_files={patched_bridge})"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
