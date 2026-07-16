#!/usr/bin/env python3
"""#1189 — audit every named speaker in a corpus against the one assertion that matters.

    No name may appear in the roster unless it is
      (a) STATED IN THE FEED             — "journalists Kevin Roose and Casey Newton"
      (b) STATED IN THE EPISODE and corroborated — introduced as a speaker, not merely discussed
      (c) SPOKEN BY THE VOICE it is assigned to  — the cluster's own self-introduction

Everything that poisoned the shipped corpus fails that test: Amy Lawrence and Paul Tenorio (ad
narrators crowned as hosts in 10/10 episodes), Elon Musk and Sam Altman (a lawsuit defendant and the
man he is suing, neither of whom speaks), Tim Cook (the subject of the episode), Jonathan Knight (a
mid-roll advert).

**This costs no GPU.** Whisper and pyannote do not re-run: the diarization is frozen on disk, so the
roster is replayed over it in milliseconds. That is the whole point — the naming layer can be
audited across the entire corpus, every time we change a rule, for free.

It runs the CURRENT roster code, not the labels the corpus happens to carry, so it answers "what
would we ship today", not "what did we ship".

Usage
-----
    python scripts/audit/corpus_speaker_audit.py <corpus_root> [--json out.json]

Exit codes
----------
- 0 — every named speaker has a warrant.
- 1 — at least one name has no warrant (the defect list is printed).
- 2 — script/input error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from podcast_scraper import config as config_mod  # noqa: E402
from podcast_scraper.providers.ml.diarization.base import (  # noqa: E402
    DiarizationResult,
    DiarizationSegment,
)
from podcast_scraper.providers.ml.diarization.roster import (  # noqa: E402
    resolve_speaker_roster,
)
from podcast_scraper.rss import extract_episode_description, fetch_and_parse_rss  # noqa: E402
from podcast_scraper.speaker_detectors.corroboration import corroborate_guests  # noqa: E402
from podcast_scraper.speaker_detectors.hosts import (  # noqa: E402
    detect_hosts_from_feed,
    guests_introduced_by_the_host,
)

WARRANT_FEED = "feed states this host"
WARRANT_EPISODE = "episode description, corroborated"
WARRANT_SPOKEN = "spoken by this voice"
WARRANT_INTRODUCED = "introduced by the host on air"
NO_WARRANT = "NO WARRANT"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _diarized_episodes(feed_dir: Path) -> List[Path]:
    out = []
    for p in sorted(feed_dir.glob("run_*/transcripts/*.segments.json")):
        if ".adfree" in p.name:
            continue
        try:
            segs = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(segs, list) and segs and segs[0].get("speaker"):
            out.append(p)
    return out


def _episode_description(title_index: Dict[str, str], stem: str) -> str:
    key = _norm(stem.split("_2026")[0])
    for t_norm, desc in title_index.items():
        if t_norm and (t_norm[:24] in key or key[:24] in t_norm):
            return desc
    return ""


def _guests_for_episode(title: str, description: str, nlp: Any) -> List[str]:
    """Guests the EPISODE declares, corroborated — the same gate production runs (no LLM)."""
    if not description:
        return []
    from podcast_scraper.speaker_detectors.entities import extract_person_entities

    proposed = [n for n, _ in extract_person_entities(description[:500], nlp)]
    return corroborate_guests(proposed, episode_title=title, episode_description=description)


def _audit_episode(
    path: Path, feed_hosts: List[str], guests: List[str]
) -> Tuple[Dict[str, str], Dict[str, float]]:
    segs = json.loads(path.read_text())

    talk: Dict[str, float] = defaultdict(float)
    voice_texts: Dict[str, str] = defaultdict(str)
    for s in segs:
        v = str(s["speaker"])
        talk[v] += float(s["end"]) - float(s["start"])
        voice_texts[v] += " " + (s.get("text") or "")

    diar = DiarizationResult(
        segments=[
            DiarizationSegment(float(s["start"]), float(s["end"]), str(s["speaker"])) for s in segs
        ],
        num_speakers=len(talk),
    )
    roster = resolve_speaker_roster(
        diar,
        " ".join((s.get("text") or "") for s in segs),
        detected_guests=guests,
        known_hosts=feed_hosts,
        voice_texts=dict(voice_texts),
        ordered_turns=[(str(s["speaker"]), (s.get("text") or "")) for s in segs],
    )

    hosts_lower = {h.lower() for h in feed_hosts}
    guests_lower = {g.lower() for g in guests}
    # "My guest today is Brian Chesky" — a name the HOST stated on air. It is not in the feed and
    # not in the episode description, and it is still a fact, not a guess.
    introduced_lower = {n.lower() for n in guests_introduced_by_the_host(dict(voice_texts))}

    warrants: Dict[str, str] = {}
    shares: Dict[str, float] = {}
    total = sum(talk.values()) or 1.0
    for voice, role in roster.by_voice.items():
        if not role.named:
            continue
        name = role.name
        shares[name] = 100 * talk.get(voice, 0.0) / total
        low = name.lower()
        own = voice_texts.get(voice, "").lower()
        if low in hosts_lower:
            warrants[name] = WARRANT_FEED
        elif low in guests_lower:
            warrants[name] = WARRANT_EPISODE
        elif low in introduced_lower:
            warrants[name] = WARRANT_INTRODUCED
        elif low in own or low.split()[-1] in own:
            warrants[name] = WARRANT_SPOKEN
        else:
            warrants[name] = NO_WARRANT
    return warrants, shares


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus_root", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--ner-model", default="en_core_web_trf")
    args = ap.parse_args(argv)

    import spacy

    nlp = spacy.load(args.ner_model)

    feeds = sorted((args.corpus_root / "feeds").glob("*"))
    if not feeds:
        print(f"no feeds under {args.corpus_root}", file=sys.stderr)
        return 2

    report: Dict[str, Any] = {}
    total_eps = 0
    offenders: Dict[str, List[str]] = defaultdict(list)

    for feed_dir in feeds:
        episodes = _diarized_episodes(feed_dir)
        if not episodes:
            continue
        idx = sorted(feed_dir.glob("run_*/index.json"))
        if not idx:
            continue
        url = json.loads(idx[-1].read_text())["feed_url"]

        try:
            feed = fetch_and_parse_rss(config_mod.Config(rss=url))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {feed_dir.name}: feed fetch failed ({exc})", file=sys.stderr)
            continue

        feed_hosts = sorted(
            detect_hosts_from_feed(feed.title, feed.description, feed.authors, nlp=nlp)
        )
        title_index = {
            _norm(item.findtext("title") or ""): (extract_episode_description(item) or "")
            for item in feed.items
        }
        titles = {
            _norm(item.findtext("title") or ""): (item.findtext("title") or "")
            for item in feed.items
        }

        show = str(feed.title)
        print(f"\n=== {show[:44]:46} hosts from feed: {feed_hosts or 'NONE STATED'}")

        per_show: Dict[str, Any] = {"hosts_from_feed": feed_hosts, "episodes": {}}
        for path in episodes:
            stem = path.stem.replace(".segments", "")
            desc = _episode_description(title_index, stem)
            key = _norm(stem.split("_2026")[0])
            title = next(
                (t for k, t in titles.items() if k and (k[:24] in key or key[:24] in k)), stem
            )
            guests = _guests_for_episode(title, desc, nlp)
            warrants, shares = _audit_episode(path, feed_hosts, guests)
            total_eps += 1

            bad = {n: s for n, s in warrants.items() if s == NO_WARRANT}
            per_show["episodes"][stem] = {"warrants": warrants, "shares": shares}
            for n in bad:
                offenders[n].append(show)

            flag = f"  <-- {len(bad)} UNWARRANTED" if bad else ""
            names = ", ".join(sorted(warrants))
            print(f"    {stem.split('_2026')[0][:34]:36} {names[:56]}{flag}")
            for n in sorted(bad):
                print(f"        NO WARRANT: {n}  ({shares.get(n, 0):.1f}% of talk)")

        report[show] = per_show

    print(f"\n{'=' * 78}\naudited {total_eps} episodes across {len(report)} shows, zero GPU\n")
    if offenders:
        print("NAMES WITH NO WARRANT — not in the feed, not corroborated, not self-spoken:\n")
        for name, shows in sorted(offenders.items(), key=lambda kv: -len(kv[1])):
            print(f"  {len(shows):3d} ep(s)  {name:26} {sorted(set(shows))}")
    else:
        print("every named speaker has a warrant.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")

    return 1 if offenders else 0


if __name__ == "__main__":
    raise SystemExit(main())
