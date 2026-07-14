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
from podcast_scraper.speaker_detectors.boilerplate import recurring_shingles  # noqa: E402
from podcast_scraper.speaker_detectors.corroboration import corroborate_guests  # noqa: E402
from podcast_scraper.speaker_detectors.entities import extract_person_entities  # noqa: E402
from podcast_scraper.speaker_detectors.hosts import detect_hosts_from_feed  # noqa: E402
from podcast_scraper.speaker_detectors.resolution import (  # noqa: E402
    resolve_voices_from_conversation,
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _diarized(feed_dir: Path) -> List[Path]:
    """One file per EPISODE — the latest run.

    A feed can carry several `run_*` directories, and Hard Fork's ten episodes sit in eight of them
    (our own diarization sweeps). A naive glob returns 80 files, so the corpus gets relabelled eight
    times over, the cross-episode ad index is built from eight copies of everything, and every
    statistic ever computed from this walk was weighted 8x toward one show.
    """
    latest: Dict[str, tuple] = {}
    for p in sorted(feed_dir.glob("run_*/transcripts/*.segments.json")):
        if ".adfree" in p.name:
            continue
        try:
            segs = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not (isinstance(segs, list) and segs and segs[0].get("speaker")):
            continue
        m = re.match(r"(\d{4})", p.name)
        key = m.group(1) if m else p.name[:24]
        run = next((x for x in p.parts if x.startswith("run_")), "")
        if key not in latest or run > latest[key][0]:
            latest[key] = (run, p)
    return [p for _, p in sorted(latest.values())]


def _relabel(
    path: Path,
    feed_hosts: List[str],
    title: str,
    description: str,
    nlp: Any,
    llm: Any = None,
    recurring_text: Optional[set] = None,
) -> Dict:
    segs = json.loads(path.read_text())

    voice_texts: Dict[str, str] = defaultdict(str)
    for s in segs:
        voice_texts[str(s["speaker"])] += " " + (s.get("text") or "")

    proposed = (
        [n for n, _ in extract_person_entities(description[:800], nlp)] if description else []
    )
    guests = corroborate_guests(proposed, episode_title=title, episode_description=description)

    diar = DiarizationResult(
        segments=[
            DiarizationSegment(float(s["start"]), float(s["end"]), str(s["speaker"])) for s in segs
        ],
        num_speakers=len({str(s["speaker"]) for s in segs}),
    )
    ordered_turns = [(str(s["speaker"]), (s.get("text") or "")) for s in segs]

    # ADR-110 — ask who speaks AFTER we can hear them. The model may only MATCH a name the metadata
    # STATED; it can never author one. Without this the replay resolves a different roster than the
    # one production ships, and the corpus it writes is not the corpus we would build.
    llm_voice_names: Dict[str, str] = {}
    if llm is not None:
        candidates = list(dict.fromkeys([*proposed, *guests, *feed_hosts]))
        llm_voice_names = resolve_voices_from_conversation(
            candidates,
            dict(voice_texts),
            llm.complete_text,
            known_hosts=feed_hosts,
            ordered_turns=ordered_turns,
        )

    roster = resolve_speaker_roster(
        diar,
        " ".join((s.get("text") or "") for s in segs),
        detected_guests=guests,
        known_hosts=feed_hosts,
        voice_texts=dict(voice_texts),
        ordered_turns=ordered_turns,
        metadata_named=proposed,
        llm_voice_names=llm_voice_names,
        recurring_text=recurring_text,
    )

    # `voice_type` and `speaker_role` are what GI's gates read: an advertisement is never grounded,
    # and an unattributed voice is never surfaced. Writing only `speaker_label` leaves every one of
    # those gates INERT — the transcripts look right and the gates guard nothing. This mirrors
    # `_enriched_segments` in the production diarization pipeline, which is the contract GI expects.
    for s in segs:
        vid = str(s["speaker"])
        s["speaker_label"] = roster.label_for(vid)
        role = roster.by_voice.get(vid)
        if role is not None and not role.named:
            if role.voice_type != "person":
                s["voice_type"] = role.voice_type
            if role.role == "host":
                s["speaker_role"] = "host"

    text, offset_segs = format_diarized_screenplay_with_offsets(segs)

    # The formatter REBUILDS each segment dict (start/end/speaker_label/text/char_start/char_end),
    # so anything else on the input is dropped — including the `voice_type` GI's gates read. Without
    # it every gate is inert: the transcripts look right and nothing is guarded. Re-attach it here,
    # keyed on the label, since an unnamed voice's label IS its raw id.
    by_label = {roster.label_for(v): r for v, r in roster.by_voice.items() if not r.named}
    for seg in offset_segs:
        role = by_label.get(str(seg.get("speaker_label")))
        if role is None:
            continue
        if role.voice_type != "person":
            seg["voice_type"] = role.voice_type
        if role.role == "host":
            seg["speaker_role"] = "host"

    named = sorted({r.name for r in roster.by_voice.values() if r.named})
    return {"segments": offset_segs, "text": text, "named": named, "guests": guests}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus_root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ner-model", default="en_core_web_trf")
    ap.add_argument("--per-show", type=int, default=0, help="0 = every episode")
    ap.add_argument(
        "--llm",
        default="gemini",
        help="ADR-110 speaker resolution provider ('none' to replay the deterministic cues only)",
    )
    ap.add_argument("--llm-model", default="gemini-2.5-flash-lite")
    args = ap.parse_args(argv)

    import spacy

    nlp = spacy.load(args.ner_model)

    # The replay must resolve the SAME roster production would. Without the LLM resolution and the
    # feed's repeated-script index, it writes a corpus we do not actually build.
    llm = None
    if args.llm and args.llm != "none":
        from podcast_scraper.summarization.factory import create_summarization_provider

        llm_cfg = config_mod.Config(
            rss="https://example.com/f.xml", summary_provider=args.llm
        ).model_copy(update={f"{args.llm}_summary_model": args.llm_model})
        llm = create_summarization_provider(llm_cfg)
        llm.initialize()
        print(f"ADR-110 speaker resolution: {args.llm}/{args.llm_model}")

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

        # #1188 — the script THIS feed repeats. A voice that only reads it is a recording, and no
        # single episode can tell. Built from every diarized episode of the feed, not just the ones
        # we are about to write.
        recurring = recurring_shingles(
            [
                " ".join(s.get("text") or "" for s in json.loads(p.read_text()))
                for p in _diarized(feed_dir)
            ]
        )
        if recurring:
            print(f"    (#1188: {len(recurring)} repeated passages across this feed)")

        for path in eps:
            stem = path.stem.replace(".segments", "")
            key = _norm(stem.split("_2026")[0])
            title, desc = next(
                ((t, d) for k, (t, d) in meta.items() if k and (k[:24] in key or key[:24] in k)),
                (stem, ""),
            )
            r = _relabel(path, hosts, title, desc, nlp, llm=llm, recurring_text=recurring)
            (show_dir / f"{stem}.segments.json").write_text(json.dumps(r["segments"]))
            (show_dir / f"{stem}.txt").write_text(r["text"], encoding="utf-8")
            total += 1
            print(f"    {stem.split('_2026')[0][:34]:36} {', '.join(r['named'])[:50]}")

    print(f"\nrelabelled {total} episodes — zero GPU")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
