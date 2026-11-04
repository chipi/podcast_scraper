#!/usr/bin/env python3
"""Freeze corpus migration 0043's list from the census. Read-only on the corpus.

Every voice ``m0043_census.py`` lists (stored name; today's roster unnames it and types it
commercial) is classified, in this order (record ``docs/wip/changes/0043_*.md`` section 3):

- **A**: m0016 skipped it because only the pre-1.2.0 ``detected_guests`` stated it
  (``m0043_candidates.classify`` says ``rule_a``);
- **C**: one of m0016's own recurrence hits on the corpus's signatures;
- a stated host of the episode (the feed's ``known_hosts``, or a roster entry with role host): kept
  named, unless its id is in :data:`HOST_NAME_ON_AN_AD` (**E**, operator 2026-11-04);
- an id in :data:`EXCLUDED`: kept named, for the reason given;
- **B**: every other voice (another show's or the publisher's promo, a supporter shout-out).

Each entry is 0034's shape ``(old name, old role, None, None, evidence)``; the evidence names the
class and the voice's first segment (the deciding segment a reviewer reads first).

    python scripts/measure/m0043_list.py --corpus <root> --census c_0.jsonl ... [--out <list.json>]

Needs ``search/ad_signatures.json`` (``ad_signatures.write_for_corpus``) for classes A and C.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from podcast_scraper.upgrade import reviewer_verdicts as rv  # noqa: E402
from podcast_scraper.upgrade.migrations import (  # noqa: E402
    m0016_ad_reader_names_removed as m16,
    m0034_talk_floor_names_removed as m34,
    m0043_non_participant_names_removed as m43,
)
from podcast_scraper.utils.run_relative import run_relative  # noqa: E402

#: Host names on voices that read an ad in another language (operator 2026-11-04, open call 3).
HOST_NAME_ON_AN_AD: Dict[rv.Key, str] = {
    ("743e06dc-a88d-11f1-a4dd-878b8bea9556", "SPEAKER_00"): "a German-language ad",
    ("47f783de-0450-4f3b-9a6c-251b63ddff6e", "SPEAKER_07"): "a German-language ad",
}
#: Census voices kept named, each with the operator's reason.
EXCLUDED: Dict[rv.Key, str] = {
    ("https://iono.fm/e/1726296", "SPEAKER_00"): (
        "the show's own host reading the show's premium promo (record section 3, class B)"
    ),
}


def _candidates() -> Any:
    spec = importlib.util.spec_from_file_location(
        "m0043_candidates", REPO / "scripts" / "measure" / "m0043_candidates.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _episode(root: Path, meta_rel: str) -> Tuple[Dict[str, Any], List[dict], Dict[str, Any]]:
    meta = root / meta_rel
    doc = m16._load(meta) or {}
    content = doc.get("content") or {}
    rel = run_relative(content.get("transcript_file_path"), meta.parent.parent)
    rows = m16._rows(m16._load(meta.parent.parent / rel.replace(".txt", ".segments.json")))
    diag = m16._load(meta.parent.parent / rel.replace(".txt", ".speakers.diagnostics.json"))
    return doc, rows, diag if isinstance(diag, dict) else {}


def is_stated_host(name: str, content: Dict[str, Any], diag: Dict[str, Any]) -> bool:
    """The feed states ``name`` as a host of the episode, or its roster seats it as one."""
    hosts = [str(h) for h in (diag.get("tried") or {}).get("known_hosts") or []]
    if any(m16.same_person(name, h) for h in hosts):
        return True
    return any(
        isinstance(e, dict)
        and e.get("role") == "host"
        and m16.same_person(name, str(e.get("name")))
        for e in content.get("speakers") or []
    )


def build(root: Path, census: List[Dict[str, Any]]) -> Tuple[Dict[rv.Key, Any], Dict[rv.Key, str]]:
    """``(list entries, {key: why not listed})`` for the census rows."""
    mc = _candidates()
    loaded = m16._load_signatures(root)
    if loaded is None:
        raise SystemExit("no search/ad_signatures.json: build it first")
    recurrence, _lang, excluded_hits = m16.find_hits(root, loaded[0])
    rule_a = {(h[0], h[1]) for h in excluded_hits if mc.classify(root, h)["verdict"] == "rule_a"}
    recurring = {(h[0], h[1]) for h in recurrence}
    entries: Dict[rv.Key, Any] = {}
    skipped: Dict[rv.Key, str] = {}
    for c in census:
        key = (str(c["episode_id"]), str(c["voice"]))
        doc, rows, diag = _episode(root, c["meta"])
        name, role = m34.served_by_voice(rows).get(key[1], (None, None))
        if name is None:
            skipped[key] = "no served name on the voice"
            continue
        first = next(i for i, r in enumerate(rows) if str(r.get("speaker")) == key[1])
        where = (c["meta"], key[1])
        if where in rule_a:
            why = "m0016 skipped it; stated only by the pre-1.2.0 detected_guests"
            cls = "A"
        elif where in recurring:
            why = "one of m0016's recurrence hits on the corpus's ad signatures"
            cls = "C"
        elif is_stated_host(name, doc.get("content") or {}, diag):
            if key not in HOST_NAME_ON_AN_AD:
                skipped[key] = "a stated host of the episode"
                continue
            why = f"the host's name on {HOST_NAME_ON_AN_AD[key]}"
            cls = "E"
        elif key in EXCLUDED:
            skipped[key] = EXCLUDED[key]
            continue
        else:
            why = "another show's or the publisher's promo, or a supporter shout-out"
            cls = "B"
        entries[key] = (
            name,
            role,
            None,
            None,
            f"class {cls}: {why}; today's roster types it commercial; segment {first}",
        )
    return entries, skipped


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--census", required=True, nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=m43.LIST_FILE)
    args = ap.parse_args(argv)
    census = [json.loads(ln) for p in args.census for ln in p.read_text().splitlines() if ln]
    entries, skipped = build(args.corpus, census)
    m34.write_list(args.out, entries, m43.MIGRATION_ID)
    m43.load_list(args.out)  # the migration's own checks refuse a bad entry here, not at apply
    by_class = Counter(str(e[4]).split(":")[0] for e in entries.values())
    print(
        f"census {len(census)}; listed {len(entries)} {dict(sorted(by_class.items()))}; "
        f"not listed {len(skipped)} -> {args.out}"
    )
    for key, why in sorted(skipped.items()):
        print(f"  not listed {key[0]} {key[1]}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
