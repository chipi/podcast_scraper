#!/usr/bin/env python3
"""#1189 — for every UNNAMED voice, is its name recoverable, and from where?

The warrant audit measures precision: no name we cannot justify. This measures the other half —
COVERAGE — and, more usefully, the CEILING on coverage.

23% of the corpus's talk is still an anonymous SPEAKER_NN. Under-naming is the safe direction
(#876 — a wrong name is worse than no name), but it is not free: an unnamed voice means a quote with
no speaker, and an insight with nobody behind it. A STANCE needs a speaker.

So for each unnamed voice, ask where its name COULD have come from:

    OWN_INTRO      the voice says "I'm <Name>" in its own turns      -> we should have named it: BUG
    HOST_INTRO     another voice introduces it by name               -> we should have named it: BUG
    DESCRIPTION    the episode description names it                  -> the gate may be dropping it
    NOWHERE        nobody in the episode ever names this person      -> SPEAKER_NN is HONEST

The split between BUG and HONEST is the whole point: it tells us how much of the 23% is recoverable
before we spend anything on prompts, models, or GPU.

Zero GPU. The diarization is frozen; the roster is replayed over it.

Usage
-----
    python scripts/audit/attribution_ceiling.py <corpus_root>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from podcast_scraper import config as config_mod  # noqa: E402
from podcast_scraper.providers.ml.diarization.base import (  # noqa: E402
    DiarizationResult,
    DiarizationSegment,
)
from podcast_scraper.providers.ml.diarization.roster import (  # noqa: E402
    resolve_speaker_roster,
    VOICE_CAMEO,
    VOICE_COMMERCIAL,
)
from podcast_scraper.rss import extract_episode_description, fetch_and_parse_rss  # noqa: E402
from podcast_scraper.speaker_detectors.corroboration import corroborate_guests  # noqa: E402
from podcast_scraper.speaker_detectors.entities import extract_person_entities  # noqa: E402
from podcast_scraper.speaker_detectors.hosts import detect_hosts_from_feed  # noqa: E402

_NAME = r"[A-Z][\w'’\-]+(?:\s+[A-Z][\w'’\-]+)+"
_OWN_INTRO = re.compile(rf"\bI'?m\s+({_NAME})")

# The diagnostic MUST use the same rule the roster ships, or it reports a bug that does not exist.
# It did: a looser regex here counted every "welcome ..." and "talking to ..." as an introduction,
# and reported 5.2% of the corpus as recoverable. Real introductions are rare — roughly one per
# episode — and the number was an artifact of the measuring instrument.
from podcast_scraper.speaker_detectors.hosts import (  # noqa: E402
    _GUEST_INTRODUCED_BY_HOST as _HOST_INTRO,
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus_root", type=Path)
    ap.add_argument("--ner-model", default="en_core_web_trf")
    args = ap.parse_args(argv)

    import spacy

    nlp = spacy.load(args.ner_model)

    buckets: Dict[str, float] = defaultdict(float)
    per_show: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    examples: Dict[str, List[str]] = defaultdict(list)

    for feed_dir in sorted((args.corpus_root / "feeds").glob("*")):
        idx = sorted(feed_dir.glob("run_*/index.json"))
        eps = [
            p
            for p in sorted(feed_dir.glob("run_*/transcripts/*.segments.json"))
            if ".adfree" not in p.name
        ]
        if not idx or not eps:
            continue
        try:
            feed = fetch_and_parse_rss(
                config_mod.Config(rss=json.loads(idx[-1].read_text())["feed_url"])
            )
        except Exception:  # noqa: BLE001
            continue

        hosts = sorted(detect_hosts_from_feed(feed.title, feed.description, feed.authors, nlp=nlp))
        meta = {
            _norm(i.findtext("title") or ""): (
                (i.findtext("title") or ""),
                (extract_episode_description(i) or ""),
            )
            for i in feed.items
        }
        show = str(feed.title)[:30]

        for path in eps:
            segs = json.loads(path.read_text())
            if not (segs and segs[0].get("speaker")):
                continue
            vt: Dict[str, str] = defaultdict(str)
            talk: Dict[str, float] = defaultdict(float)
            for s in segs:
                v = str(s["speaker"])
                vt[v] += " " + (s.get("text") or "")
                talk[v] += float(s["end"]) - float(s["start"])

            stem = path.stem.replace(".segments", "")
            key = _norm(stem.split("_2026")[0])
            title, desc = next(
                ((t, d) for k, (t, d) in meta.items() if k and (k[:24] in key or key[:24] in k)),
                (stem, ""),
            )
            proposed = [n for n, _ in extract_person_entities(desc[:500], nlp)] if desc else []
            guests = corroborate_guests(proposed, episode_title=title, episode_description=desc)
            desc_names = (
                {n.lower() for n, _ in extract_person_entities(desc[:800], nlp)} if desc else set()
            )

            diar = DiarizationResult(
                segments=[
                    DiarizationSegment(float(s["start"]), float(s["end"]), str(s["speaker"]))
                    for s in segs
                ],
                num_speakers=len(talk),
            )
            roster = resolve_speaker_roster(
                diar,
                " ".join((s.get("text") or "") for s in segs),
                detected_guests=guests,
                known_hosts=hosts,
                voice_texts=dict(vt),
                ordered_turns=[(str(s2["speaker"]), (s2.get("text") or "")) for s2 in segs],
            )
            others = {v: t for v, t in vt.items()}

            for v, r in roster.by_voice.items():
                t = talk.get(v, 0.0)
                if r.named:
                    buckets["NAMED"] += t
                    per_show[show]["NAMED"] += t
                    continue
                if r.voice_type in (VOICE_COMMERCIAL, VOICE_CAMEO):
                    buckets["AD/CAMEO"] += t
                    per_show[show]["AD/CAMEO"] += t
                    continue

                own = _OWN_INTRO.search(vt.get(v, ""))
                introduced = any(_HOST_INTRO.search(txt) for vv, txt in others.items() if vv != v)
                if own:
                    bucket = "OWN_INTRO (BUG)"
                    if len(examples[bucket]) < 6:
                        examples[bucket].append(f'{show}: {v} says "I\'m {own.group(1)}"')
                elif desc_names:
                    bucket = "DESCRIPTION (gate?)"
                    if len(examples[bucket]) < 6:
                        examples[bucket].append(
                            f"{show}: {v}, description names {sorted(desc_names)[:2]}"
                        )
                elif introduced:
                    bucket = "HOST_INTRO (BUG)"
                    if len(examples[bucket]) < 6:
                        examples[bucket].append(f"{show}: {v} is introduced by another voice")
                else:
                    bucket = "NOWHERE (honest)"
                    if len(examples[bucket]) < 6:
                        examples[bucket].append(f"{show}: {v} — nobody names this person")
                buckets[bucket] += t
                per_show[show][bucket] += t

    total = sum(buckets.values()) or 1.0
    order = [
        "NAMED",
        "AD/CAMEO",
        "OWN_INTRO (BUG)",
        "HOST_INTRO (BUG)",
        "DESCRIPTION (gate?)",
        "NOWHERE (honest)",
    ]
    print("\nWHERE THE UNNAMED TALK COULD HAVE GOT ITS NAME (share of all corpus talk)\n")
    for k in order:
        v = buckets.get(k, 0.0)
        bar = "#" * int(60 * v / total)
        print(f"  {k:22} {100*v/total:6.2f}%  {bar}")

    recoverable = buckets.get("OWN_INTRO (BUG)", 0) + buckets.get("HOST_INTRO (BUG)", 0)
    gated = buckets.get("DESCRIPTION (gate?)", 0)
    honest = buckets.get("NOWHERE (honest)", 0)
    unnamed = recoverable + gated + honest
    print(f"\n  unnamed total          {100*unnamed/total:6.2f}%")
    print(f"    recoverable (BUG)    {100*recoverable/total:6.2f}%   <- fix this")
    print(f"    behind the gate      {100*gated/total:6.2f}%   <- may be recoverable")
    print(f"    genuinely nameless   {100*honest/total:6.2f}%   <- SPEAKER_NN is correct")
    print(f"\n  ceiling on coverage:   {100*(buckets['NAMED']+recoverable+gated)/total:6.2f}%")

    for k in order[2:]:
        if examples.get(k):
            print(f"\n  {k}:")
            for e in examples[k]:
                print(f"    - {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
