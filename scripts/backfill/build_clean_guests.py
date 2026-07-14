#!/usr/bin/env python3
"""#1188 — the corroborated guest list for a corpus's episodes, for the roster to name voices with.

``rediarize_corpus.py`` reuses whatever guest names the corpus already recorded. On the prod-v3
corpus those names came from the LLM speaker detector BEFORE the corroboration gate existed, so they
include Elon Musk (the man suing OpenAI) and Sam Altman (the man who runs it) as *speakers*. Feeding
them back in would re-poison the rebuilt roster with the very names the rebuild exists to remove.

So the guests are re-derived here, through the shipping path:

    feed metadata -> speaker detector -> corroborate_guests -> guests.json

``corroborate_guests`` is the gate: a name the LLM proposes must be introduced as a speaker by the
episode's own description, or it does not go near a voice cluster. The output feeds
``rediarize_corpus.py --guests-json``.

Usage
-----
    python scripts/backfill/build_clean_guests.py \
        --rss https://feeds.simplecast.com/l2i9YnTd \
        --media-dir <corpus>/feeds/<feed>/run_*/media \
        --out .test_outputs/rebuild-v3/guests.json [--provider ollama]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from podcast_scraper import config as config_mod  # noqa: E402
from podcast_scraper.rss import extract_episode_description, fetch_and_parse_rss  # noqa: E402
from podcast_scraper.speaker_detectors.corroboration import corroborate_guests  # noqa: E402
from podcast_scraper.speaker_detectors.factory import create_speaker_detector  # noqa: E402


def _norm(s: str) -> str:
    """Fold a title down to something that survives the media filename's mangling."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())[:28]


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rss", required=True)
    ap.add_argument("--media-dir", required=True, help="the corpus run's media/ directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--known-hosts", nargs="*", default=[])
    args = ap.parse_args(argv)

    media = sorted(Path(args.media_dir).glob("*.mp3"))
    if not media:
        print(f"no audio under {args.media_dir}", file=sys.stderr)
        return 2

    cfg = config_mod.Config(rss=args.rss, speaker_detector_provider=args.provider)
    detector = create_speaker_detector(cfg)
    detector.initialize()

    feed = fetch_and_parse_rss(cfg)
    known = set(args.known_hosts)

    out: Dict[str, Any] = {}
    for item in feed.items:
        title = (item.findtext("title") or "").strip()
        stem = next((m.stem for m in media if _norm(title)[:20] in _norm(m.stem)), None)
        if not stem:
            continue

        description = extract_episode_description(item) or ""
        proposed, hosts, ok, _ = detector.detect_speakers(
            episode_title=title,
            episode_description=description,
            known_hosts=known,
        )
        host_strings = {str(h) for h in (hosts or set())} | known
        guests = corroborate_guests(
            [p for p in (proposed or []) if p not in host_strings],
            episode_title=title,
            episode_description=description,
            known_hosts=host_strings,
        )
        rejected = [p for p in (proposed or []) if p not in host_strings and p not in guests]

        out[stem] = {"title": title, "guests": guests, "hosts": sorted(host_strings)}
        print(f"{stem[:36]:38} guests={guests}")
        if rejected:
            print(f"{'':38} REJECTED (never introduced as speakers): {rejected}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}  ({len(out)} episodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
