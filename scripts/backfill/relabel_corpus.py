#!/usr/bin/env python3
"""#1189 — re-resolve WHO SPOKE across a corpus, with no GPU at all.

The ASR was never wrong. The diarization clusters were never wrong. Only the NAMES were: ad
narrators crowned as hosts, a lawsuit defendant given a doctor's voice, one host spelled three
different ways.

So nothing needs re-deriving. The frozen diarization is replayed through the CURRENT roster —
metadata for who, conversation for which voice — and the corrected screenplay is written out.
Whisper does not run. pyannote does not run. A full corpus relabels in minutes.

That is what makes the feedback loop affordable: change a rule, relabel everything, audit everything
(`scripts/audit/corpus_speaker_audit.py`), read the defect list. No ten-at-a-time rationing.

Outputs, per episode, under ``--out/<feed>/``:
  - ``<stem>.segments.json``  the segments with corrected ``speaker_label``
  - ``<stem>.txt``            the screenplay transcript GI reads

Usage
-----
    python scripts/backfill/relabel_corpus.py <corpus_root> --out .test_outputs/relabel-v3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from podcast_scraper import config as config_mod  # noqa: E402
from podcast_scraper.providers.ml.diarization.base import (  # noqa: E402
    DiarizationResult,
    DiarizationSegment,
)
from podcast_scraper.providers.ml.diarization.formatting import (  # noqa: E402
    format_diarized_screenplay_with_offsets,
)
from podcast_scraper.providers.ml.diarization.roster import (  # noqa: E402
    resolve_speaker_roster,
)
from podcast_scraper.rss import extract_episode_description, fetch_and_parse_rss  # noqa: E402
from podcast_scraper.speaker_detectors.corroboration import corroborate_guests  # noqa: E402
from podcast_scraper.speaker_detectors.entities import extract_person_entities  # noqa: E402
from podcast_scraper.speaker_detectors.hosts import detect_hosts_from_feed  # noqa: E402


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _diarized(feed_dir: Path) -> List[Path]:
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


def _relabel(path: Path, feed_hosts: List[str], title: str, description: str, nlp: Any) -> Dict:
    segs = json.loads(path.read_text())

    voice_texts: Dict[str, str] = defaultdict(str)
    for s in segs:
        voice_texts[str(s["speaker"])] += " " + (s.get("text") or "")

    proposed = (
        [n for n, _ in extract_person_entities(description[:500], nlp)] if description else []
    )
    guests = corroborate_guests(proposed, episode_title=title, episode_description=description)

    diar = DiarizationResult(
        segments=[
            DiarizationSegment(float(s["start"]), float(s["end"]), str(s["speaker"])) for s in segs
        ],
        num_speakers=len({str(s["speaker"]) for s in segs}),
    )
    roster = resolve_speaker_roster(
        diar,
        " ".join((s.get("text") or "") for s in segs),
        detected_guests=guests,
        known_hosts=feed_hosts,
        voice_texts=dict(voice_texts),
        ordered_turns=[(str(s["speaker"]), (s.get("text") or "")) for s in segs],
    )

    for s in segs:
        s["speaker_label"] = roster.label_for(str(s["speaker"]))

    text, offset_segs = format_diarized_screenplay_with_offsets(segs)
    named = sorted({r.name for r in roster.by_voice.values() if r.named})
    return {"segments": offset_segs, "text": text, "named": named, "guests": guests}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus_root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ner-model", default="en_core_web_trf")
    ap.add_argument("--per-show", type=int, default=0, help="0 = every episode")
    args = ap.parse_args(argv)

    import spacy

    nlp = spacy.load(args.ner_model)

    total = 0
    for feed_dir in sorted((args.corpus_root / "feeds").glob("*")):
        eps = _diarized(feed_dir)
        idx = sorted(feed_dir.glob("run_*/index.json"))
        if not eps or not idx:
            continue
        url = json.loads(idx[-1].read_text())["feed_url"]
        try:
            feed = fetch_and_parse_rss(config_mod.Config(rss=url))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {feed_dir.name}: {exc}", file=sys.stderr)
            continue

        hosts = sorted(detect_hosts_from_feed(feed.title, feed.description, feed.authors, nlp=nlp))
        meta = {
            _norm(i.findtext("title") or ""): (
                (i.findtext("title") or ""),
                (extract_episode_description(i) or ""),
            )
            for i in feed.items
        }

        show_dir = args.out / feed_dir.name
        show_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {str(feed.title)[:40]:42} hosts: {hosts or 'NONE STATED'}")

        if args.per_show:
            eps = eps[: args.per_show]
        for path in eps:
            stem = path.stem.replace(".segments", "")
            key = _norm(stem.split("_2026")[0])
            title, desc = next(
                ((t, d) for k, (t, d) in meta.items() if k and (k[:24] in key or key[:24] in k)),
                (stem, ""),
            )
            r = _relabel(path, hosts, title, desc, nlp)
            (show_dir / f"{stem}.segments.json").write_text(json.dumps(r["segments"]))
            (show_dir / f"{stem}.txt").write_text(r["text"], encoding="utf-8")
            total += 1
            print(f"    {stem.split('_2026')[0][:34]:36} {', '.join(r['named'])[:50]}")

    print(f"\nrelabelled {total} episodes — zero GPU")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
