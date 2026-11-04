#!/usr/bin/env python3
"""Rule A candidates for corpus migration 0043. Read-only.

m0016 removes a name from a voice the corpus's ad signatures call an ad, unless a source STATED the
name for the episode. One of the sources it counts is the metadata's own ``detected_guests``: in
artifacts before schema 1.2.0 (#2075) a projection of the roster by role, so the old roster's own
guest seat for a self-introduced ad reader counts as a statement and m0016 keeps the name (33
voices on snapshot-prod-20261103). Rule A takes m0016's excluded hits and keeps those whose name
is stated ONLY by ``content.detected_guests`` and appears nowhere in the episode's publisher text
(feed title and description, episode title and description).

    python scripts/measure/m0043_candidates.py --corpus <root> [--json out.jsonl]

Needs ``search/ad_signatures.json`` (``ad_signatures.write_for_corpus`` builds it). Prints one line
per excluded hit with its verdict (rule_a / kept: <why>) and a summary. The output holds names:
keep it out of the public repo.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from podcast_scraper.upgrade.migrations import m0016_ad_reader_names_removed as m16  # noqa: E402
from podcast_scraper.utils.run_relative import run_relative  # noqa: E402


def _strings(values: Any) -> List[str]:
    out = []
    for v in values or []:
        name = v if isinstance(v, str) else (v or {}).get("name") if isinstance(v, dict) else None
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
    return out


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def publisher_text(doc: Dict[str, Any]) -> str:
    """The episode's publisher-written text: feed and episode titles and descriptions."""
    feed, ep = doc.get("feed") or {}, doc.get("episode") or {}
    return _norm(
        " ".join(
            str(x or "")
            for x in (
                feed.get("title"),
                feed.get("description"),
                ep.get("title"),
                ep.get("description"),
            )
        )
    )


def diagnostics(meta: Path, content: Dict[str, Any]) -> Any:
    """The episode's speaker diagnostics, found through the run even when the metadata stores the
    transcript path written at ingest; ``None`` when there is none to read."""
    rel = run_relative(content.get("transcript_file_path"), meta.parent.parent)
    if not rel.endswith(".txt"):
        return None
    return m16._load(meta.parent.parent / rel.replace(".txt", ".speakers.diagnostics.json"))


def stated_by_source(meta: Path, content: Dict[str, Any], name: str) -> List[str]:
    """Which of m0016's stated-name sources list ``name`` (m16.same_person)."""
    diag = diagnostics(meta, content)
    tried = (diag or {}).get("tried") or {} if isinstance(diag, dict) else {}
    summary = (diag or {}).get("summary") or {} if isinstance(diag, dict) else {}
    sources = {
        "content.detected_hosts": _strings(content.get("detected_hosts")),
        "content.detected_guests": _strings(content.get("detected_guests")),
        "tried.known_hosts": _strings(tried.get("known_hosts")),
        "tried.detected_guests": _strings(tried.get("detected_guests")),
        "tried.metadata_named": _strings(tried.get("metadata_named")),
        "summary.unbound_names": _strings(summary.get("unbound_names")),
        "speakers.stated_entry": [
            str(e.get("name") or "")
            for e in content.get("speakers") or []
            if isinstance(e, dict)
            and (e.get("placed") is False or e.get("source") in m16._STATED_SOURCES)
        ],
    }
    return [k for k, names in sources.items() if any(m16.same_person(name, s) for s in names)]


def classify(root: Path, hit: tuple) -> Dict[str, Any]:
    rel, voice, name = hit
    meta = root / rel
    doc = json.loads(meta.read_text(encoding="utf-8"))
    content = doc.get("content") or {}
    sources = stated_by_source(meta, content, name)
    key = _norm(name)
    in_publisher = bool(key) and f" {key} " in f" {publisher_text(doc)} "
    others = [s for s in sources if s != "content.detected_guests"]
    if not isinstance(diagnostics(meta, content), dict):
        # Unread diagnostics would hide a host the feed states: never rule A on that.
        verdict = "kept: no speaker diagnostics to read"
    elif in_publisher:
        verdict = "kept: the publisher's text names them"
    elif others:
        verdict = "kept: stated by " + ",".join(others)
    elif sources == ["content.detected_guests"]:
        verdict = "rule_a"
    else:
        verdict = "kept: not stated by content.detected_guests"
    return {
        "meta": rel,
        "voice": voice,
        "name": name,
        "sources": sources,
        "in_publisher_text": in_publisher,
        "verdict": verdict,
        "feed": (doc.get("feed") or {}).get("title"),
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    loaded = m16._load_signatures(args.corpus)
    if loaded is None:
        print("no search/ad_signatures.json: build it first (ad_signatures.write_for_corpus)")
        return 2
    _recurrence, _language_only, excluded = m16.find_hits(args.corpus, loaded[0])
    rows = [classify(args.corpus, h) for h in excluded]
    for r in rows:
        print(f"{r['verdict'][:45]:45s} {r['name']:22s} {r['voice']:11s} {str(r['feed'])[:28]}")
    tally = Counter(r["verdict"].split(":")[0] for r in rows)
    print(
        f"excluded hits {len(rows)}: {dict(tally)}; rule_a episodes "
        f"{len({r['meta'] for r in rows if r['verdict'] == 'rule_a'})}"
    )
    if args.json:
        args.json.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
