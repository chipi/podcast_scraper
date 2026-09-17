"""Is every episode's people surface in sync with its speaker record? (#2075)

Operator rule 2026-09-17: ONE speaker record per episode (`content.speakers`) and every surface that
shows people is written from it — the transcript's speaker names, who said each quote (gi.json), the
people and roles in kg.json, and the operator graph that reads both. An episode whose surfaces
disagree is the defect, however right each file looks on its own.

For each episode the app would show (newest run per episode, the catalog's own membership rule),
this loads metadata.json, kg.json, gi.json, both segments files and the speakers diagnostics, and
runs `kg.speaker_coherence.check_episode_in_sync`. Every violation carries a code:

  QUOTE_NOT_PLACED      a quote credited to someone no voice was matched to
  QUOTE_FIELDS_VS_EDGE  a quote's own speaker_id disagrees with its SPOKEN_BY edge
  CAST_NOT_PLACED       kg.json lists a host/guest no voice was matched to
  UNPLACED_CAST         the record says placed: false, kg.json gives them a speaking role
  PLACED_NOT_CAST       a placed host/guest missing from kg.json's speakers
  LABEL_NOT_PLACED      the transcript names a speaker who is not a placed entry
  RAW_VS_ADFREE         raw and ad-free segments name different speakers
  RECORD_VS_DIAGNOSTICS the record and the roster's own diagnostics disagree

Episodes written before schema 1.2.0 carry no record and are SKIPPED unless --legacy-as-placed is
given, which treats every named roster entry as placed — for auditing an existing corpus, where it
inherits that roster's errors on purpose.

Read-only. Exits 1 when any examined episode is out of sync, so it can gate a validation run or a
deploy step.

    make speaker-sync-audit CORPUS_DIR=<corpus> [LEGACY=1]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from podcast_scraper.kg.speaker_coherence import (  # noqa: E402
    check_episode_in_sync,
    has_speaker_record,
)
from podcast_scraper.search.corpus_scope import (  # noqa: E402
    dedupe_metadata_paths_newest_run_per_episode,
    discover_metadata_files,
)

_SUFFIX = ".metadata.json"


def _load(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def episode_surfaces(meta_path: Path) -> Optional[Dict[str, Any]]:
    """Everything the sync check reads for one episode, or ``None`` when metadata is unreadable."""
    metadata = _load(meta_path)
    if not isinstance(metadata, dict):
        return None
    stem = meta_path.name[: -len(_SUFFIX)]
    rel = str((metadata.get("content") or {}).get("transcript_file_path") or "")
    segments = adfree = diagnostics = None
    if rel:
        base = str(meta_path.parent.parent / rel)
        base = base[: -len(".txt")] if base.endswith(".txt") else base
        segments = _load(Path(base + ".segments.json"))
        adfree = _load(Path(base + ".adfree.segments.json"))
        diagnostics = _load(Path(base + ".speakers.diagnostics.json"))
    return {
        "metadata": metadata,
        "kg": _load(meta_path.parent / f"{stem}.kg.json") or {},
        "gi": _load(meta_path.parent / f"{stem}.gi.json"),
        "segments": segments,
        "adfree_segments": adfree,
        "diagnostics": diagnostics if isinstance(diagnostics, dict) else None,
    }


def audit(corpus: Path, *, legacy_as_placed: bool = False) -> Tuple[List[dict], Dict[str, int]]:
    """``(findings, counts)``; one finding per out-of-sync episode."""
    paths = dedupe_metadata_paths_newest_run_per_episode(corpus, discover_metadata_files(corpus))
    counts = collections.Counter()
    findings: List[dict] = []
    for meta_path in sorted(paths):
        s = episode_surfaces(meta_path)
        if s is None:
            counts["unreadable"] += 1
            continue
        if not has_speaker_record(s["metadata"]) and not legacy_as_placed:
            counts["skipped_no_record"] += 1
            continue
        counts["examined"] += 1
        violations = check_episode_in_sync(
            s["metadata"],
            s["kg"],
            s["gi"],
            segments=s["segments"],
            adfree_segments=s["adfree_segments"],
            diagnostics=s["diagnostics"],
            legacy_as_placed=legacy_as_placed,
        )
        if not violations:
            counts["in_sync"] += 1
            continue
        feed = str((s["metadata"].get("feed") or {}).get("title") or "")
        findings.append(
            {
                "metadata": str(meta_path.relative_to(corpus)),
                "feed": feed,
                "episode": str((s["metadata"].get("episode") or {}).get("title") or ""),
                "violations": violations,
            }
        )
    counts["out_of_sync"] = len(findings)
    return findings, dict(counts)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus-dir", required=True, help="Corpus parent path")
    ap.add_argument(
        "--legacy-as-placed",
        action="store_true",
        help="Audit pre-1.2.0 artifacts, treating every named roster entry as placed",
    )
    ap.add_argument("--json", action="store_true", help="Emit findings as JSON")
    ap.add_argument("--quiet-ok", action="store_true", help="Print nothing when in sync")
    ap.add_argument("--limit", type=int, default=10, help="Examples to show per code")
    args = ap.parse_args(argv)

    corpus = Path(args.corpus_dir)
    if not corpus.is_dir():
        print(f"not a directory: {corpus}", file=sys.stderr)
        return 2
    findings, counts = audit(corpus, legacy_as_placed=args.legacy_as_placed)
    if args.json:
        print(json.dumps({"counts": counts, "findings": findings}, indent=2))
        return 1 if findings else 0
    if not findings:
        if not args.quiet_ok:
            print(
                f"speaker sync: OK — {counts.get('examined', 0)} episodes in sync "
                f"({counts.get('skipped_no_record', 0)} skipped: no speaker record)"
            )
        return 0

    by_code: collections.Counter = collections.Counter()
    episodes_by_code: collections.Counter = collections.Counter()
    examples: Dict[str, List[str]] = collections.defaultdict(list)
    for f in findings:
        codes = set()
        for v in f["violations"]:
            code = v.split(" ", 1)[0]
            by_code[code] += 1
            codes.add(code)
            if len(examples[code]) < args.limit:
                examples[code].append(f"{f['feed'][:28]} | {f['episode'][:40]} | {v}")
        for c in codes:
            episodes_by_code[c] += 1

    print("SPEAKER SYNC AUDIT (#2075)\n")
    for k in ("examined", "in_sync", "out_of_sync", "skipped_no_record", "unreadable"):
        print(f"  {k:20} {counts.get(k, 0)}")
    print("\n  violations by code (episodes / violations):")
    for code, n in by_code.most_common():
        print(f"    {code:22} {episodes_by_code[code]:5} / {n}")
    for code, rows in examples.items():
        print(f"\n  {code}:")
        for r in rows:
            print(f"    {r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
