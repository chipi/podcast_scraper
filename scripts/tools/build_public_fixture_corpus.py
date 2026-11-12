#!/usr/bin/env python3
"""Derive the public fixture corpus from the app validation corpus (E04 step 8).

``tests/fixtures/app-validation-corpus/public/`` is what a public-only install keeps of ``v3``:
the same feeds, episodes, transcripts, summaries, KG and translations, without what the private
extension writes (the split line, Common ``docs/wip/platform-plan/epics/E04-split-line.md``).
Deterministic: the same ``v3`` gives the same tree, so ``--check`` can gate it.

Dropped:

- GI and its bridge (``*.gi.json``, ``*.bridge.json``); the GI enrichers' per-episode outputs
  (``metadata/enrichments/``);
- speakers: ``*.speakers.diagnostics.json``, ``content.speakers`` and
  ``content.diarization_num_speakers`` in the metadata, ``speaker_label`` on every segment, and
  the ``Name: `` prefix on every transcript line (public writes a plain transcript, split line §8
  decision 1). Segment character ranges are remapped so each still slices its own text;
- the ad-free base (``*.adfree.*``; public alone writes none, decision 2);
- in the KG, people seated as host, guest or speaker and their edges (the basic pass writes no
  speaker or host roles, split line §5);
- corpus outputs of moving code: every corpus enricher but ``topic_cooccurrence_corpus`` (and its
  entries in ``run.jsonl`` / ``run_summary.json``), the person images, ``search/`` (topic
  clusters; the index is built on demand), ``.viewer/`` (server bookkeeping).

Kept as they are: summaries, translations (``*.translation.json``, ``<base>.<lang>.*``), the
pattern-cleaned ``*.cleaned.txt`` (pattern cleaning stays public), artwork, licences.

Not verified against a real public-only run: that is the deploy gate c ("a public-only install
runs end to end"), whose artifacts should match these rules; where they differ, the rules change.

Usage::

    python scripts/tools/build_public_fixture_corpus.py           # write public/ from v3
    python scripts/tools/build_public_fixture_corpus.py --check   # exit 1 if public/ is stale
"""

from __future__ import annotations

import argparse
import filecmp
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests" / "fixtures" / "app-validation-corpus" / "v3"
TARGET = ROOT / "tests" / "fixtures" / "app-validation-corpus" / "public"

PUBLIC_CORPUS_ENRICHERS = frozenset({"topic_cooccurrence_corpus"})
SPEAKER_ROLES = frozenset({"host", "guest", "speaker"})
_DROPPED_SUFFIXES = (".gi.json", ".bridge.json", ".speakers.diagnostics.json")
_ADFREE = re.compile(r"\.adfree\.")


def _dropped(rel: Path) -> bool:
    name = rel.name
    parts = rel.parts
    if parts[0] in ("search", ".viewer"):
        return True
    if parts[0] == "enrichments":
        if name in ("run.jsonl", "run_summary.json"):
            return False
        return Path(name).stem not in PUBLIC_CORPUS_ENRICHERS
    if "enrichments" in parts[:-1]:  # per-episode GI enricher outputs under metadata/
        return True
    return name.endswith(_DROPPED_SUFFIXES) or bool(_ADFREE.search(name))


def _dump(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _speaker_names(metadata: Dict[str, Any], segment_files: Iterable[Path]) -> Set[str]:
    names = {
        str(s.get("name") or "").strip()
        for s in (metadata.get("content") or {}).get("speakers") or []
        if isinstance(s, dict)
    }
    for path in segment_files:
        for row in _rows(json.loads(path.read_text(encoding="utf-8"))):
            names.add(str(row.get("speaker_label") or "").strip())
    return {n for n in names if n}


def _rows(data: Any) -> List[Dict[str, Any]]:
    return data if isinstance(data, list) else list((data or {}).get("segments") or [])


def _strip_prefixes(text: str, names: Set[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """``text`` without a ``Name: `` prefix on any line, plus ``(old_offset, removed)`` cuts."""
    out: List[str] = []
    cuts: List[Tuple[int, int]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        head, sep, _rest = line.partition(": ")
        if sep and head.strip() in names:
            removed = len(head) + len(sep)
            cuts.append((pos, removed))
            out.append(line[removed:])
        else:
            out.append(line)
        pos += len(line)
    return "".join(out), cuts


def _remap(offset: int, cuts: List[Tuple[int, int]]) -> int:
    shift = 0
    for start, removed in cuts:
        if offset >= start + removed:
            shift += removed
        elif offset > start:  # inside a removed prefix: the line's new start
            shift += offset - start
    return offset - shift


def _derive_episode_texts(src_tr: Path, dst_tr: Path, names: Set[str]) -> None:
    """Every transcript variant without speaker prefixes; its segments remapped and unlabelled."""
    for txt in sorted(src_tr.glob("*.txt")):
        if _ADFREE.search(txt.name):
            continue
        text, cuts = _strip_prefixes(txt.read_text(encoding="utf-8"), names)
        (dst_tr / txt.name).write_text(text, encoding="utf-8")
        seg = txt.with_name(txt.name[: -len(".txt")] + ".segments.json")
        if not seg.exists():
            continue
        data = json.loads(seg.read_text(encoding="utf-8"))
        rows = _rows(data)
        for row in rows:
            row.pop("speaker_label", None)
            if "char_start" in row:
                row["char_start"] = _remap(int(row["char_start"]), cuts)
                row["char_end"] = _remap(int(row["char_end"]), cuts)
                if text[row["char_start"] : row["char_end"]] != row["text"]:
                    raise SystemExit(f"remap broke a segment's range: {seg} {row.get('id')}")
        (dst_tr / seg.name).write_text(_dump(data), encoding="utf-8")


def _derive_metadata(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    content = data.get("content") or {}
    content.pop("speakers", None)
    content.pop("diarization_num_speakers", None)
    return data


def _derive_kg(path: Path) -> Dict[str, Any]:
    kg: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    seated = {
        n.get("id")
        for n in kg.get("nodes") or []
        if n.get("type") == "Person" and (n.get("properties") or {}).get("role") in SPEAKER_ROLES
    }
    kg["nodes"] = [n for n in kg.get("nodes") or [] if n.get("id") not in seated]
    kg["edges"] = [
        e
        for e in kg.get("edges") or []
        if not ({e.get("from"), e.get("to"), e.get("source"), e.get("target")} & seated)
        and e.get("type") not in ("HOSTS", "GUESTS_ON", "SPOKEN_BY", "ATTRIBUTED_TO")
    ]
    return kg


def _derive_runs(src: Path, dst: Path) -> None:
    runs = src / "enrichments" / "run.jsonl"
    if runs.exists():
        kept = []
        for line in runs.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("enricher_id") and event["enricher_id"] not in PUBLIC_CORPUS_ENRICHERS:
                continue
            if "enricher_set" in event:
                event["enricher_set"] = [
                    e for e in event["enricher_set"] if e in PUBLIC_CORPUS_ENRICHERS
                ]
            kept.append(json.dumps(event, sort_keys=True, ensure_ascii=False))
        (dst / "enrichments" / "run.jsonl").write_text("\n".join(kept) + "\n", encoding="utf-8")
    summary = src / "enrichments" / "run_summary.json"
    if summary.exists():
        data = json.loads(summary.read_text(encoding="utf-8"))
        per = data.get("per_enricher") or {}
        data["per_enricher"] = {k: v for k, v in per.items() if k in PUBLIC_CORPUS_ENRICHERS}
        (dst / "enrichments" / "run_summary.json").write_text(_dump(data), encoding="utf-8")


def build(src: Path, dst: Path) -> None:
    """Write the public corpus for ``src`` into the empty directory ``dst``."""
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        if path.is_dir() or _dropped(rel):
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.name.endswith(".metadata.json"):
            target.write_text(_dump(_derive_metadata(path)), encoding="utf-8")
        elif path.name.endswith(".kg.json"):
            target.write_text(_dump(_derive_kg(path)), encoding="utf-8")
        elif rel.parts[-2:-1] == ("transcripts",) and (
            path.suffix == ".txt" or path.name.endswith(".segments.json")
        ):
            continue  # written per episode below, so text and ranges stay consistent
        elif rel.parts[0] == "enrichments" and path.name in ("run.jsonl", "run_summary.json"):
            continue
        else:
            shutil.copy2(path, target)
    for meta in sorted(src.rglob("*.metadata.json")):
        src_tr = meta.parent.parent / "transcripts"
        if not src_tr.is_dir():
            continue
        dst_tr = dst / src_tr.relative_to(src)
        dst_tr.mkdir(parents=True, exist_ok=True)
        stem = meta.name[: -len(".metadata.json")]
        names = _speaker_names(
            json.loads(meta.read_text(encoding="utf-8")),
            sorted(src_tr.glob(f"{stem}*.segments.json")),
        )
        stem_dir = Path(tempfile.mkdtemp())
        for f in src_tr.glob(f"{stem}.*"):
            shutil.copy2(f, stem_dir / f.name)
        _derive_episode_texts(stem_dir, dst_tr, names)
        shutil.rmtree(stem_dir)
    _derive_runs(src, dst)


def _differences(a: Path, b: Path) -> List[str]:
    files_a = {p.relative_to(a) for p in a.rglob("*") if p.is_file()}
    files_b = {p.relative_to(b) for p in b.rglob("*") if p.is_file()}
    out = [f"only built: {p}" for p in sorted(files_a - files_b)]
    out += [f"only committed: {p}" for p in sorted(files_b - files_a)]
    out += [
        f"differs: {p}"
        for p in sorted(files_a & files_b)
        if not filecmp.cmp(a / p, b / p, shallow=False)
    ]
    return out


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if public/ is stale")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        built = Path(tmp) / "public"
        built.mkdir()
        build(SOURCE, built)
        if args.check:
            diffs = _differences(built, TARGET) if TARGET.exists() else ["public/ missing"]
            for d in diffs[:20]:
                print(d)
            print("STALE" if diffs else "UP TO DATE")
            return 1 if diffs else 0
        if TARGET.exists():
            shutil.rmtree(TARGET)
        shutil.copytree(built, TARGET)
    print(f"wrote {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
