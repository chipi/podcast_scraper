#!/usr/bin/env python3
"""Generate RSS feeds that actually DESCRIBE the app-validation corpus, so the real pipeline
can build it.

Why this exists (2026-08-16)
----------------------------
``build_app_validation_corpus.py`` synthesizes the corpus from ``tests/fixtures/transcripts/<ver>/``
and reads RSS only for *feed-level* metadata. The RSS fixtures were therefore never a description of
the corpus, and nobody noticed until the real pipeline was pointed at them:

    corpus:            9 shows x 4 episodes  = 36
    RSS advertised:    p01-p05: 3 each, p06: 6, p07: 1, p08: 1, p09: 3 = 26

A pipeline run can only process what a feed advertises, so p07 and p08 would have produced ONE
episode each. Feeds that under-describe the corpus make a real pipeline run impossible.

These generated feeds close that gap: one feed per show, listing every episode that has BOTH a
transcript and an audio file, with **measured** enclosure sizes and durations.

Why NOT rewrite the existing tests/fixtures/rss/*.xml
----------------------------------------------------
Those are an e2e TEST surface with deliberate shapes — ``p01_episode_selection.xml`` is exactly
"three items, newest-first" because an episode-order test asserts on it, and the ``_fast`` feeds are
single 1-minute episodes for quick paths. Growing them to corpus size would break those tests for
reasons unrelated to what they check. Different purpose, different files.

Measured, not asserted
----------------------
The hand-authored feeds carry durations and enclosure lengths that do not match the audio: e.g.
``p02_e01`` claims ``<itunes:duration>00:11:00</itunes:duration>`` and ``length="3900000"`` while
the file is 00:06:32.96. Everything here is read off the actual mp3 (duration via ffmpeg, length via
stat), so the feed cannot drift from the media it points at.

Provenance of each field:
  * channel title/description/author/language — the show's existing RSS fixture, with
    SHOW_META_OVERRIDE applied for p06/p08/p09 (whose fixtures are themed for other shows).
  * item title      — the transcript's ``## `` header, the same source the corpus builder uses.
  * item description— the existing feed's description for that guid when there is one (they are
    hand-authored and good); otherwise built from the transcript's own Host:/Guest: headers. Never
    invented.
  * pubDate         — the authored ground-truth publish dates (#1148 schedule).
  * enclosure/duration — measured off tests/fixtures/audio/<ver>/.

Usage::

    python scripts/build_corpus_feeds.py
    python scripts/build_corpus_feeds.py --check   # CI: non-zero if stale
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET  # nosec B405 — Element/ParseError typing only
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Optional

from defusedxml.ElementTree import parse as safe_parse, ParseError as DefusedParseError

ROOT = Path(__file__).resolve().parents[1]
_ITUNES_NS = "{http://www.itunes.com/dtds/podcast-1.0.dtd}"
_EPISODE_RE = re.compile(r"^p(\d+)_e(\d+)$")

# (rss stem carrying the show's channel metadata, show id) — mirrors APP_SHOWS in
# build_app_validation_corpus.py, in corpus order.
SHOWS: list[tuple[str, str]] = [
    ("p01_mtb", "p01"),
    ("p02_software", "p02"),
    ("p03_scuba", "p03"),
    ("p04_photo", "p04"),
    ("p05_investing", "p05"),
    ("p06_edge_cases", "p06"),
    ("p07_sustainability", "p07"),
    ("p08_solar", "p08"),
    ("p09_biohacking", "p09"),
]

# p06/p08/p09 RSS fixtures are themed for other shows; restore the v3 identity (#1148).
SHOW_META_OVERRIDE: dict[str, dict[str, str]] = {
    "p06": {
        "title": "The Drift",
        "description": "Dialogue-heavy, meandering long-form interviews.",
    },
    "p08": {
        "title": "Public Hour",
        "description": "NPR-shape public-radio conversations.",
    },
    "p09": {
        "title": "Cross-Show",
        "description": "Recurring guests from other shows revisit their positions.",
    },
}


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _audio_duration_seconds(path: Path) -> Optional[float]:
    """Duration via ffmpeg's own report.

    None when ffmpeg is unavailable or the file is unreadable.
    """
    try:
        proc = subprocess.run(
            ["ffmpeg", "-i", str(path)], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)", proc.stderr or "")
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _hhmmss(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _transcript_title(path: Path, fallback: str) -> str:
    """The ``## <subtitle>`` header — the same source build_app_validation_corpus uses."""
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                return stripped[3:].strip() or fallback
    except OSError:
        pass
    return fallback


def _transcript_people(path: Path) -> tuple[Optional[str], Optional[str]]:
    """(host, guest) from the transcript's own ``Host:`` / ``Guest:`` header lines."""
    host = guest = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines()[:12]:
            stripped = line.strip()
            if stripped.startswith("Host:") and host is None:
                host = stripped[5:].strip() or None
            elif stripped.startswith("Guest:") and guest is None:
                guest = stripped[6:].strip() or None
            elif stripped.startswith("["):
                break  # dialogue started; headers are done
    except OSError:
        pass
    return host, guest


def _existing_feed(rss_dir: Path, stem: str) -> tuple[dict[str, str], dict[str, str]]:
    """(channel metadata, guid -> description) from a hand-authored fixture."""
    meta: dict[str, str] = {}
    descriptions: dict[str, str] = {}
    path = rss_dir / f"{stem}.xml"
    if not path.is_file():
        return meta, descriptions
    try:
        channel = safe_parse(path).getroot().find("channel")
    except (ET.ParseError, DefusedParseError):
        return meta, descriptions
    if channel is None:
        return meta, descriptions

    for key, tag in (("title", "title"), ("description", "description"), ("language", "language")):
        value = (channel.findtext(tag) or "").strip()
        if value:
            meta[key] = value
    author = channel.find(f"{_ITUNES_NS}author")
    if author is not None and (author.text or "").strip():
        meta["author"] = author.text.strip()

    for item in channel.findall("item"):
        guid = (item.findtext("guid") or "").strip()
        desc = (item.findtext("description") or "").strip()
        if guid and desc:
            descriptions[guid] = desc
    return meta, descriptions


def _publish_dates(gt_dir: Path) -> dict[str, str]:
    """episode id -> RFC-2822 pubDate, from the authored ground-truth sidecars (#1148)."""
    import json

    out: dict[str, str] = {}
    if not gt_dir.is_dir():
        return out
    for sidecar in sorted(gt_dir.glob("*.json")):
        try:
            doc = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        raw = doc.get("publish_date") or doc.get("published_at") or doc.get("pub_date")
        if not raw:
            continue
        text = str(raw).strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        out[sidecar.stem] = format_datetime(parsed)
    return out


def _episodes_for(show: str, transcripts_dir: Path, audio_dir: Path) -> list[str]:
    """Episode ids with BOTH a transcript and audio, in episode order.

    Strict ``pNN_eMM`` only — the ``_fast`` / ``_multi`` variants are separate smoke fixtures and
    are not corpus episodes.
    """
    found: list[tuple[int, str]] = []
    for transcript in transcripts_dir.glob(f"{show}_e*.txt"):
        episode_id = transcript.stem
        match = _EPISODE_RE.match(episode_id)
        if not match:
            continue
        if not (audio_dir / f"{episode_id}.mp3").is_file():
            print(f"  !! {episode_id}: transcript but NO audio — omitted", file=sys.stderr)
            continue
        found.append((int(match.group(2)), episode_id))
    return [episode_id for _, episode_id in sorted(found)]


def _render_feed(
    show: str,
    meta: dict[str, str],
    episodes: list[str],
    transcripts_dir: Path,
    audio_dir: Path,
    descriptions: dict[str, str],
    pub_dates: dict[str, str],
) -> str:
    title = meta.get("title", show)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<rss version="2.0"',
        '     xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">',
        "  <channel>",
        f"    <title>{_esc(title)}</title>",
        "    <link>http://localhost/</link>",
        f"    <description>{_esc(meta.get('description', ''))}</description>",
        f"    <language>{_esc(meta.get('language', 'en-us'))}</language>",
        "",
        f"    <itunes:author>{_esc(meta.get('author', ''))}</itunes:author>",
        "    <itunes:explicit>no</itunes:explicit>",
        f'    <itunes:image href="/images/{show}_cover.svg"/>',
    ]

    for index, episode_id in enumerate(episodes, start=1):
        transcript = transcripts_dir / f"{episode_id}.txt"
        audio = audio_dir / f"{episode_id}.mp3"
        ep_title = _transcript_title(transcript, f"{title} — Episode {index}")

        description = descriptions.get(episode_id)
        if not description:
            host, guest = _transcript_people(transcript)
            if host and guest:
                description = f"{host} talks with {guest}: {ep_title}."
            elif host:
                description = f"{host} on {ep_title}."
            else:
                description = f"{ep_title}."

        seconds = _audio_duration_seconds(audio)
        duration = _hhmmss(seconds) if seconds is not None else "00:00:00"
        if seconds is None:
            print(f"  !! {episode_id}: could not measure duration", file=sys.stderr)

        pub_date = pub_dates.get(episode_id, "Mon, 01 Sep 2025 07:00:00 +0000")
        lines += [
            "",
            "    <item>",
            f"      <title>{_esc(ep_title)}</title>",
            f'      <guid isPermaLink="false">{episode_id}</guid>',
            f"      <pubDate>{pub_date}</pubDate>",
            f"      <description>{_esc(description)}</description>",
            f'      <enclosure url="/audio/{episode_id}.mp3" '
            f'length="{audio.stat().st_size}" type="audio/mpeg"/>',
            f"      <itunes:episode>{index}</itunes:episode>",
            f"      <itunes:duration>{duration}</itunes:duration>",
            "      <itunes:episodeType>full</itunes:episodeType>",
            "    </item>",
        ]

    lines += ["  </channel>", "</rss>", ""]
    return "\n".join(lines)


_DURATION_RE = re.compile(r"<itunes:duration>(\d+):(\d{2}):(\d{2})</itunes:duration>")

#: A lossy MP3's reported length is not a fixed number — it depends on the decoder. ffmpeg 7.1 and
#: 9.0.1 disagree by up to a frame on the same file, which rounds to a 1-second difference in
#: ``HH:MM:SS``. Observed on p06/p08 (00:04:09 vs 00:04:08 and two others) purely by switching which
#: ffmpeg was on PATH.
_DURATION_TOLERANCE_S = 1


def _differs_beyond_decoder_rounding(committed: str, rendered: str) -> bool:
    """True when the committed feed is genuinely stale, ignoring ±1s duration jitter.

    ``--check`` exists to catch a real omission: someone edits a transcript or replaces an audio
    file and forgets to regenerate the feeds. It used to compare the two XML documents byte for
    byte, which also flags a difference that means nothing — a duration measured by a DIFFERENT
    ffmpeg build. The fixture is then only "up to date" relative to whichever ffmpeg last touched
    it, and the check fails for the next contributor with no actionable difference to fix. Worse,
    regenerating to satisfy it just moves the failure to everyone on the other version.

    So: durations may differ by at most a second; everything else must match exactly. A real change
    — a re-recorded episode, a new item, an edited title — moves things far more than one second,
    so the check keeps the teeth it was given.
    """
    if committed == rendered:
        return False
    a, b = _DURATION_RE.split(committed), _DURATION_RE.split(rendered)
    if len(a) != len(b):
        return True  # different number of items — a structural change, not rounding
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y:
            continue
        # The regex split interleaves literal text with (h, m, s) capture triples; a mismatching
        # literal chunk is always a real difference.
        if i % 4 == 0:
            return True
        return _duration_gap_exceeds_tolerance(a, b)
    return False


def _duration_gap_exceeds_tolerance(a: list[str], b: list[str]) -> bool:
    """Compare every (h, m, s) triple from the split, allowing ``_DURATION_TOLERANCE_S``."""
    for i in range(1, len(a), 4):
        secs_a = int(a[i]) * 3600 + int(a[i + 1]) * 60 + int(a[i + 2])
        secs_b = int(b[i]) * 3600 + int(b[i + 1]) * 60 + int(b[i + 2])
        if abs(secs_a - secs_b) > _DURATION_TOLERANCE_S:
            return True
    return False


def main(argv: Optional[list[str]] = None) -> int:
    version = (ROOT / "tests" / "fixtures" / "FIXTURES_VERSION").read_text(encoding="utf-8").strip()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rss-dir", type=Path, default=ROOT / "tests" / "fixtures" / "rss")
    ap.add_argument(
        "--transcripts-dir",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "transcripts" / version,
    )
    ap.add_argument(
        "--audio-dir", type=Path, default=ROOT / "tests" / "fixtures" / "audio" / version
    )
    ap.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=ROOT / "tests" / "fixtures" / "ground-truth" / version / "ground_truth",
    )
    ap.add_argument(
        "--check", action="store_true", help="verify without writing; non-zero if stale"
    )
    args = ap.parse_args(argv)

    for label, path in (
        ("--transcripts-dir", args.transcripts_dir),
        ("--audio-dir", args.audio_dir),
    ):
        if not path.is_dir():
            print(f"{label} does not exist: {path}", file=sys.stderr)
            return 2

    pub_dates = _publish_dates(args.ground_truth_dir)
    stale: list[str] = []
    total = 0

    for stem, show in SHOWS:
        meta, descriptions = _existing_feed(args.rss_dir, stem)
        meta.update(SHOW_META_OVERRIDE.get(show, {}))
        episodes = _episodes_for(show, args.transcripts_dir, args.audio_dir)
        if not episodes:
            print(f"  !! {show}: no episodes with both transcript and audio", file=sys.stderr)
            continue
        xml = _render_feed(
            show, meta, episodes, args.transcripts_dir, args.audio_dir, descriptions, pub_dates
        )
        dst = args.rss_dir / f"{show}_corpus.xml"
        total += len(episodes)
        if args.check:
            if not dst.is_file() or _differs_beyond_decoder_rounding(
                dst.read_text(encoding="utf-8"), xml
            ):
                stale.append(dst.name)
        else:
            dst.write_text(xml, encoding="utf-8")
            print(f"  {dst.name}: {len(episodes)} episodes ({', '.join(episodes)})")

    if args.check:
        if stale:
            print(f"stale or missing corpus feeds: {', '.join(stale)}", file=sys.stderr)
            print("run: python scripts/build_corpus_feeds.py", file=sys.stderr)
            return 1
        print(f"corpus feeds up to date ({len(SHOWS)} shows, {total} episodes)")
        return 0

    print(f"\nwrote {len(SHOWS)} corpus feeds, {total} episodes total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
