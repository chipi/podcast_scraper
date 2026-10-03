"""Review one show after a deepen: its ``show.json`` checks plus every served episode's speakers.

WHY THIS IS COMMITTED. The Freakonomics deepen of 2026-10-03 was reviewed by hand from the show
sidecar and the episode artifacts, and that review found two real bugs in ten episodes (a KG reply
that was not JSON; "Pulitzer Prize-winning" published as a guest). Every show we deepen gets the
same review, so the review is a tool, not a remembered sequence of one-liners.

Read-only. Runs inside the api container, against the corpus it serves::

    ssh root@prod-podcast 'docker exec -i compose-api-1 python - "Gray Area"' \\
        < scripts/audit/show_review.py

The argument is a case-insensitive substring of the feed title (or the feed directory name).

What it prints, per show:

* the sidecar: hosts and how they were found, ``host_on_a_voice`` failures, artifact issues,
  ``recent_errors``, whether the host set changed between runs;
* per served episode (newest run wins, the API's own membership rule): every roster name with role,
  source and whether it is on a voice; names the deployed name gate refuses; a guest named only by a
  self-introduction while a guest from the episode metadata stays unplaced (the
  "Pulitzer Prize-winning" / Jennifer Egan shape); insights surfaceable of total; missing KG.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

CORPUS = Path("/app/output")


def _load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _feed_dirs(query: str) -> List[Path]:
    q = query.lower()
    hits = []
    for d in sorted((CORPUS / "feeds").iterdir()):
        title = str(_load(d / "show.json").get("feed", {}).get("title") or "")
        if q in title.lower() or q in d.name.lower():
            hits.append(d)
    return hits


def _sidecar(d: Path) -> None:
    s = _load(d / "show.json")
    feed = s.get("feed", {})
    print(f"\n=== {feed.get('title')}  ({d.name})  episodes={s.get('episodes')}")
    print(f"sidecar generated_at={s.get('generated_at')}")
    for h in s.get("hosts", []):
        print(
            f"host: {h.get('name')}  sources={h.get('sources')}  "
            f"used={h.get('episodes_used_by_pipeline')}  on_voice={h.get('episodes_on_a_voice')}"
        )
    if not s.get("hosts"):
        print("host: NONE")
    checks = s.get("checks", {})
    hov = checks.get("host_on_a_voice", {})
    print(
        f"host_on_a_voice: ok {hov.get('ok')}/{hov.get('episodes_with_record')}  "
        f"by_reason={hov.get('by_reason')}  without_record={hov.get('without_record')}"
    )
    for f in hov.get("failing", []):
        print(f"  FAIL {f.get('reason')}: {f.get('title')}")
    art = checks.get("artifacts", {})
    print(f"artifacts: issues on {art.get('episodes_with_issues')}  {art.get('by_issue')}")
    for e in art.get("episodes", []):
        print(f"  ISSUE {e.get('issues')}: {e.get('title')}")
    errs = s.get("recent_errors", [])
    kinds: Dict[str, int] = {}
    for e in errs:
        kinds[str(e.get("kind"))] = kinds.get(str(e.get("kind")), 0) + 1
    print(f"recent_errors: {kinds or 'none'}")
    print(f"host_set_changed_between_runs: {s.get('host_set_changed_between_runs')}")
    for det in s.get("host_detections", [])[:3]:
        if det.get("dropped_non_person"):
            print(f"  host detection dropped (non-person): {det.get('dropped_non_person')}")


def _episodes(d: Path) -> None:
    from podcast_scraper.search.corpus_scope import (
        dedupe_metadata_paths_newest_run_per_episode,
    )
    from podcast_scraper.speaker_detectors.hosts import is_publishable_speaker_name

    paths = [p for p in d.glob("run_*/metadata/*.metadata.json")]
    served = dedupe_metadata_paths_newest_run_per_episode(CORPUS, paths)
    print(f"served episodes: {len(served)}")
    for p in sorted(served, key=lambda x: str(_load(x).get("episode", {}).get("published") or "")):
        m = _load(p)
        title = str(m.get("episode", {}).get("title") or p.name)[:70]
        speakers = m.get("content", {}).get("speakers") or []
        flags: List[str] = []
        roster = []
        for sp in speakers:
            name = str(sp.get("name") or "")
            placed = bool(sp.get("placed"))
            roster.append(
                f"{sp.get('role')}:{name}[{sp.get('source')}{'' if placed else ',unplaced'}]"
            )
            if placed and name and not is_publishable_speaker_name(name):
                flags.append(f"GATE REFUSES '{name}'")
        placed_guests = [
            s for s in speakers if s.get("role") == "guest" and s.get("placed") and s.get("name")
        ]
        unplaced_meta = [
            s
            for s in speakers
            if s.get("role") == "guest"
            and not s.get("placed")
            and s.get("source") == "episode_metadata"
        ]
        if unplaced_meta and any(s.get("source") == "self_intro" for s in placed_guests):
            flags.append(
                "self-intro guest placed while metadata guest unplaced: "
                + ", ".join(str(s.get("name")) for s in unplaced_meta)
            )
        if not any(s.get("role") == "host" and s.get("placed") for s in speakers):
            flags.append("no host on a voice")
        gi = _load(p.with_name(p.name.replace(".metadata.json", ".gi.json")))
        ins = [n.get("properties", {}) for n in gi.get("nodes", []) if n.get("type") == "Insight"]
        # Absent means surfaceable; only an explicit False hides an insight (gi/pipeline.py).
        surf = sum(i.get("surfaceable") is not False for i in ins)
        attributed = sum(bool(i.get("speaker")) for i in ins)
        if ins and not attributed:
            flags.append(f"no insight carries a speaker ({len(ins)} insights)")
        if not p.with_name(p.name.replace(".metadata.json", ".kg.json")).exists():
            flags.append("no kg.json")
        print(f"- {title}")
        print(f"    roster: {'; '.join(roster) or 'EMPTY'}")
        hidden: Dict[str, int] = {}
        for i in ins:
            if i.get("surfaceable") is False:
                key = f"{i.get('speaker_voice_type') or '?'}:{i.get('speaker') or '-'}"
                hidden[key] = hidden.get(key, 0) + 1
        print(f"    insights: {surf}/{len(ins)} surfaceable, {attributed} with a speaker")
        if hidden:
            top = sorted(hidden.items(), key=lambda kv: -kv[1])[:4]
            print(f"    hidden by voice: {dict(top)}")
        for f in flags:
            print(f"    !! {f}")


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("usage: show_review.py <feed title substring>", file=sys.stderr)
        return 2
    dirs = _feed_dirs(argv[1])
    if not dirs:
        print(f"no feed matches {argv[1]!r}", file=sys.stderr)
        return 1
    for d in dirs:
        _sidecar(d)
        _episodes(d)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
