#!/usr/bin/env python3
"""Re-run every REFUSED punctuation-repair window (#2187) and record why, side by side.

The served record (``*.asr.json`` ``punctuation_repair``) keeps only word counts for a refused
window, not the text the re-decode produced, so a refusal cannot be judged from it. This script
re-runs exactly the production call -- the archived audio by guid, the same window cut, the
provider's ``transcribe_clip`` with the episode language's window prompt, the same verdict -- and
writes one JSON line per window: old text, new text, the verdict now, and every value a check
measured. It writes ONLY to ``--out``; the corpus is read, never changed.

Run inside the pipeline image (ffmpeg, rclone and the DGX client), e.g. via ``docker compose run``
of ``pipeline-llm`` with the corpus volume, as ``reprocess-prod.yml`` does::

    python scripts/measure/punctuation_refusals.py --corpus /app/output \
        --config config/profiles/prod_dgx_full.yaml --out /mnt/punct/refusals.jsonl

The output holds real episode text: keep it on the box, never commit it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Tuple


def _newest_asr_records(corpus: str) -> List[str]:
    newest: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for p in glob.glob(os.path.join(corpus, "feeds", "*", "run_*", "transcripts", "*.asr.json")):
        feed = p.split(os.sep + "feeds" + os.sep)[1].split(os.sep)[0]
        ts = p.split(os.sep + "run_")[1].split(os.sep)[0].split("_")[-1]
        stem = re.sub(r"_[0-9a-f-]{36}_\d{8}-\d{6}", "", os.path.basename(p))
        if (feed, stem) not in newest or ts > newest[(feed, stem)][0]:
            newest[(feed, stem)] = (ts, p)
    return [p for _, p in newest.values()]


def _clip_call(provider: Any) -> Optional[Any]:
    """The DGX provider's ``transcribe_clip``, looking through wrappers that do not forward it."""
    seen = 0
    while provider is not None and seen < 5:
        call = getattr(provider, "transcribe_clip", None)
        if callable(call):
            return call
        provider = next(
            (
                getattr(provider, a)
                for a in ("primary", "_primary", "inner", "_inner", "provider", "_provider")
                if getattr(provider, a, None) is not None
            ),
            None,
        )
        seen += 1
    return None


def _measures(
    old: List[Mapping[str, Any]], new_segs: List[Mapping[str, Any]], new_text: str, span
) -> Dict[str, Any]:
    from podcast_scraper.transcription import punctuation_repair as pr

    old_text = " ".join(str(s.get("text", "")) for s in old)
    old_bag, new_bag = pr._bag(old_text), pr._bag(new_text)
    old_n = sum(old_bag.values())
    covered = pr._merged([(float(s["start"]), float(s["end"])) for s in old])
    inside_n = outside_n = 0
    for seg in new_segs:
        mid = (float(seg["start"]) + float(seg["end"])) / 2
        n = len(str(seg.get("text", "")).split())
        if any(a <= mid <= b for a, b in covered):
            inside_n += n
        else:
            outside_n += n
    uncovered_s = (span[1] - span[0]) - sum(b - a for a, b in covered)
    return {
        "old_words": old_n,
        "retained": round(sum((old_bag & new_bag).values()) / old_n, 3) if old_n else None,
        "inside_words_ratio": round(inside_n / old_n, 3) if old_n else None,
        "outside_words": outside_n,
        "uncovered_s": round(uncovered_s, 1),
        "new_words_per_uncovered_s": round(outside_n / uncovered_s, 2) if uncovered_s > 0 else None,
        "repetitive": pr.repetitive_window(new_segs),
    }


def _language_of(corpus: str, asr_record: str) -> str:
    base = os.path.join(corpus, asr_record)[: -len(".asr.json")]
    run_dir = os.path.dirname(os.path.dirname(base))
    meta_path = os.path.join(run_dir, "metadata", os.path.basename(base) + ".metadata.json")
    try:
        feed = json.load(open(meta_path, encoding="utf-8")).get("feed") or {}
    except (OSError, ValueError):
        return "en"
    return str(feed.get("language") or "en").split("-")[0]


def rejudge(corpus: str, path: str) -> int:
    """Re-judge saved windows with the INSTALLED verdict -- no decode, so variants of the checks
    are compared on the very same re-transcriptions. One line per window: before -> now."""
    from podcast_scraper.transcription import punctuation, punctuation_repair as pr

    counts: Dict[Tuple[str, str], int] = {}
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        if "new_segments" not in row:
            continue
        a, b = float(row["window"][0]), float(row["window"][1])
        prompt = punctuation.window_prompt(_language_of(corpus, row["asr_record"])) or ""
        now = pr._verdict(row["old_segments"], row["new_segments"], row["new_text"], (a, b), prompt)
        key = (str(row["verdict_now"]), now or "repaired")
        counts[key] = counts.get(key, 0) + 1
        print(f"{key[0]:>20} -> {key[1]:<20} {row['episode']!s:.50} {a:.0f}-{b:.0f}s")
    for (before, now), n in sorted(counts.items()):
        print(f"TOTAL {before} -> {now}: {n}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--config", required=False)
    ap.add_argument("--out", required=False)
    ap.add_argument("--limit", type=int, default=0, help="stop after N windows (0 = all)")
    ap.add_argument("--rejudge", help="re-judge a saved --out file with the installed checks")
    args = ap.parse_args(argv)
    if args.rejudge:
        return rejudge(args.corpus, args.rejudge)
    if not (args.config and args.out):
        ap.error("--config and --out are required unless --rejudge")

    from podcast_scraper import config
    from podcast_scraper.transcription import punctuation, punctuation_repair as pr
    from podcast_scraper.transcription.gap_recovery import cut_clip
    from podcast_scraper.utils import audio_cache
    from podcast_scraper.workflow.orchestration import create_transcription_provider

    data = dict(config.load_config_file(args.config))
    data.setdefault("rss_url", "https://example.invalid/feed.xml")
    cfg = config.Config(**data)
    clip = _clip_call(create_transcription_provider(cfg))
    if clip is None:
        print("no provider with transcribe_clip in this profile", file=sys.stderr)
        return 2
    archive = audio_cache.resolve_backend(cfg, args.corpus)

    done = 0
    with open(args.out, "w", encoding="utf-8") as out, tempfile.TemporaryDirectory() as tmp:
        for asr_path in sorted(_newest_asr_records(args.corpus)):
            rec = json.load(open(asr_path, encoding="utf-8"))
            repairs = rec.get("punctuation_repair") or []
            refused = [r for r in repairs if r.get("status") == "refused"]
            if not refused:
                continue
            base = asr_path[: -len(".asr.json")]
            segs = json.load(open(base + ".segments.json", encoding="utf-8"))
            segs = segs if isinstance(segs, list) else segs.get("segments", [])
            run_dir = os.path.dirname(os.path.dirname(base))
            meta = json.load(
                open(
                    os.path.join(run_dir, "metadata", os.path.basename(base) + ".metadata.json"),
                    encoding="utf-8",
                )
            )
            ep, feed = meta.get("episode") or {}, meta.get("feed") or {}
            guid = ep.get("guid") or ep.get("episode_id")
            language = (feed.get("language") or "en").split("-")[0]
            prompt = punctuation.window_prompt(language)
            media = os.path.join(tmp, "episode.audio")
            got = audio_cache.fetch_into(archive, guid, media)
            for r in refused:
                row: Dict[str, Any] = {
                    "feed": feed.get("title"),
                    "episode": ep.get("title"),
                    "asr_record": os.path.relpath(asr_path, args.corpus),
                    "window": [r["start"], r["end"]],
                    "recorded_reason": r.get("reason"),
                }
                if not got:
                    out.write(json.dumps({**row, "error": "audio not in archive"}) + "\n")
                    continue
                a, b = float(r["start"]), float(r["end"])
                # Exactly production's selection: segments STARTING in the window. The recorded
                # end is the last one's end, so a segment starting there is the next window's.
                inside = [s for s in segs if pr._starts_in(s, a - 0.001, b - 0.001)]
                old_text = " ".join(str(s.get("text", "")).strip() for s in inside)
                clip_path = os.path.join(tmp, "window.wav")
                try:
                    cut_clip(media, a, b, clip_path)
                    new = dict(clip(clip_path, language=language, prompt=prompt))
                except Exception as exc:  # noqa: BLE001 - record and go on
                    out.write(json.dumps({**row, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                    continue
                new_segs = [
                    pr._shift(s, a)
                    for s in (new.get("segments") or [])
                    if isinstance(s, Mapping) and str(s.get("text", "")).strip()
                ]
                new_segs, _repeats = pr.drop_tail_repeats(new_segs)
                new_text = " ".join(str(s.get("text", "")).strip() for s in new_segs)
                verdict = pr._verdict(inside, new_segs, new_text, (a, b), prompt or "")
                out.write(
                    json.dumps(
                        {
                            **row,
                            "verdict_now": verdict or "repaired",
                            "measures": _measures(inside, new_segs, new_text, (a, b)),
                            "old_text": old_text,
                            "new_text": new_text,
                            # Kept so a changed check can be re-judged offline, with no new decode.
                            "old_segments": [
                                {"start": s["start"], "end": s["end"], "text": s.get("text", "")}
                                for s in inside
                            ],
                            "new_segments": [
                                {"start": s["start"], "end": s["end"], "text": s.get("text", "")}
                                for s in new_segs
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                out.flush()
                done += 1
                print(f"{done}: {row['episode']!s:.50} {a:.0f}-{b:.0f}s -> {verdict or 'repaired'}")
                if args.limit and done >= args.limit:
                    return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
