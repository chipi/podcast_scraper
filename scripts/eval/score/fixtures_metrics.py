"""Compute intrinsic metrics on a fixture transcript set.

Used to baseline v1 fixtures and compare v2 against them in the v2 fixtures
rebuild (issues #109, #111, #900). Outputs JSON suitable for diffing across
versions.

Phase 3 (now): cheap, transcript-text-derived metrics — sponsor pattern hits,
speaker recurrence, lexical diversity, rough entity-density proxy. No ML
pipeline invocation.

Phase 7 (later): augment with KG/GIL/CIL metrics computed from regenerated
goldens.

Usage:
    python scripts/eval/score/fixtures_metrics.py --version v1 \\
        --output tests/fixtures/baselines/v1-metrics.json

    # Auto-detect default from tests/fixtures/FIXTURES_VERSION:
    python scripts/eval/score/fixtures_metrics.py \\
        --output tests/fixtures/baselines/current-metrics.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from podcast_scraper.cleaning.commercial.patterns import SPONSOR_PATTERNS

SPEAKER_RE = re.compile(r"^([A-Z][A-Za-z '\-]{0,40}):\s+(.*)$")
TS_RE = re.compile(r"^\[\s*\d{1,2}:\d{2}(?::\d{2})?\s*\]$")
FILE_PREFIX_RE = re.compile(r"^(p\d{2})_", re.IGNORECASE)
PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")

# Header metadata tokens that match SPEAKER_RE but aren't dialogue.
HEADER_SPEAKER_TOKENS = frozenset({"Host", "Guest", "Title", "Topic", "Podcast", "Episode"})

COMMON_TITLECASE = frozenset(
    {
        "I",
        "The",
        "A",
        "An",
        "And",
        "But",
        "Or",
        "So",
        "If",
        "When",
        "While",
        "After",
        "Before",
        "Today",
        "Tomorrow",
        "Yesterday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "Welcome",
        "Hello",
        "Thanks",
        "Thank",
        "Yes",
        "No",
        "Okay",
        "Right",
        "Sure",
        "Well",
        "Let",
        "Most",
        "Some",
        "Many",
        "Few",
        "Both",
        "All",
        "One",
        "Two",
        "Three",
        "First",
        "Second",
        "Last",
        "What",
        "When",
        "Where",
        "Why",
        "How",
        "That",
        "This",
        "These",
        "Those",
        "Episode",
        "Podcast",
        "Host",
        "Guest",
    }
)


def parse_transcript(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    speakers: Counter[str] = Counter()
    speaker_text: dict[str, list[str]] = defaultdict(list)
    body_words = 0
    proper_nouns: Counter[str] = Counter()
    body_tokens: list[str] = []
    sponsor_hits = 0
    sponsor_pattern_hits: Counter[str] = Counter()

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or TS_RE.fullmatch(s):
            continue
        m = SPEAKER_RE.match(s)
        text = m.group(2) if m else s
        if m and m.group(1) not in HEADER_SPEAKER_TOKENS:
            speakers[m.group(1)] += 1
            speaker_text[m.group(1)].append(text)
        words = text.split()
        body_words += len(words)
        body_tokens.extend(w.lower().strip(".,!?;:\"'()[]") for w in words)
        for noun in PROPER_NOUN_RE.findall(text):
            if noun not in COMMON_TITLECASE:
                proper_nouns[noun] += 1
        for pat in SPONSOR_PATTERNS:
            for _ in pat.pattern.finditer(text):
                sponsor_hits += 1
                sponsor_pattern_hits[pat.pattern.pattern] += 1

    tokens_clean = [t for t in body_tokens if t]
    types = set(tokens_clean)
    ttr = len(types) / len(tokens_clean) if tokens_clean else 0.0

    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "speakers": dict(speakers),
        "unique_speakers": len(speakers),
        "body_words": body_words,
        "proper_nouns": dict(proper_nouns),
        "unique_proper_nouns": len(proper_nouns),
        "type_token_ratio": round(ttr, 4),
        "sponsor_pattern_hits": sponsor_hits,
        "sponsor_pattern_hits_by_pattern": dict(sponsor_pattern_hits),
    }


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    by_podcast: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        stem = Path(ep["path"]).stem
        m = FILE_PREFIX_RE.match(stem)
        pid = m.group(1).lower() if m else "unknown"
        by_podcast[pid].append(ep)

    word_counts = [ep["body_words"] for ep in episodes]
    ttrs = [ep["type_token_ratio"] for ep in episodes]
    sponsor_total = sum(ep["sponsor_pattern_hits"] for ep in episodes)

    # Cross-episode speaker recurrence within podcast (excluding the host)
    recurrence_per_podcast: dict[str, dict[str, Any]] = {}
    for pid, eps in by_podcast.items():
        speaker_episodes: dict[str, set[str]] = defaultdict(set)
        for ep in eps:
            stem = Path(ep["path"]).stem
            for sp in ep["speakers"]:
                speaker_episodes[sp].add(stem)
        recurring = {sp: sorted(eps_) for sp, eps_ in speaker_episodes.items() if len(eps_) > 1}
        recurrence_per_podcast[pid] = {
            "episodes": sorted({Path(ep["path"]).stem for ep in eps}),
            "recurring_speakers": recurring,
            "recurring_speaker_count": len(recurring),
        }

    # Cross-podcast proper-noun recurrence (rough proxy for cross-feed topic spans)
    noun_to_podcasts: dict[str, set[str]] = defaultdict(set)
    for pid, eps in by_podcast.items():
        for ep in eps:
            for n in ep["proper_nouns"]:
                noun_to_podcasts[n].add(pid)
    cross_podcast_nouns = {n: sorted(p) for n, p in noun_to_podcasts.items() if len(p) > 1}

    return {
        "episode_count": len(episodes),
        "podcast_count": len(by_podcast),
        "words_per_episode": {
            "mean": round(statistics.mean(word_counts), 1) if word_counts else 0,
            "median": int(statistics.median(word_counts)) if word_counts else 0,
            "min": min(word_counts) if word_counts else 0,
            "max": max(word_counts) if word_counts else 0,
        },
        "type_token_ratio": {
            "mean": round(statistics.mean(ttrs), 4) if ttrs else 0,
            "median": round(statistics.median(ttrs), 4) if ttrs else 0,
        },
        "sponsor_pattern_hits_total": sponsor_total,
        "sponsor_pattern_hits_per_episode_mean": (
            round(sponsor_total / len(episodes), 2) if episodes else 0
        ),
        "recurrence_per_podcast": recurrence_per_podcast,
        "cross_podcast_proper_noun_count": len(cross_podcast_nouns),
        "cross_podcast_proper_noun_samples": dict(list(cross_podcast_nouns.items())[:20]),
        "pipeline_metrics_status": "deferred to Phase 7 (post-pipeline rerun on v2)",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--version",
        default=None,
        help="Fixture version (defaults to value in tests/fixtures/FIXTURES_VERSION)",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON output path for metrics",
    )
    args = p.parse_args()

    version_file = PROJECT_ROOT / "tests" / "fixtures" / "FIXTURES_VERSION"
    version = args.version or version_file.read_text(encoding="utf-8").strip()
    transcripts_dir = PROJECT_ROOT / "tests" / "fixtures" / "transcripts" / version

    if not transcripts_dir.is_dir():
        print(f"Transcripts dir not found: {transcripts_dir}", file=sys.stderr)
        return 1

    txt_files = sorted(transcripts_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt transcripts in {transcripts_dir}", file=sys.stderr)
        return 1

    episodes = [parse_transcript(t) for t in txt_files]
    agg = aggregate(episodes)
    out = {
        "fixture_version": version,
        "transcripts_dir": str(transcripts_dir.relative_to(PROJECT_ROOT)),
        "episodes": episodes,
        "aggregate": agg,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.output} — {len(episodes)} episodes, {agg['podcast_count']} podcasts")
    print(f"  sponsor_pattern_hits_total: {agg['sponsor_pattern_hits_total']}")
    print("  recurring speakers across podcasts:")
    for pid, info in sorted(agg["recurrence_per_podcast"].items()):
        print(f"    {pid}: {info['recurring_speaker_count']} recurring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
