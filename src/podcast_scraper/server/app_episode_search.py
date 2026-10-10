"""Search within one episode for what a listener remembers (operator 2026-10-10).

The use case: a listener remembers a few words from an episode and wants that part again, to
highlight it or to find the insight about it. Two additions to the Brief's search, both reading
only the episode's timed transcript (its ``segments.json``) — no index, no model:

* :func:`exact_passages` — the passages that contain every remembered word, verbatim, with their
  real times. Measured on 94 real episodes (``docs/wip/BRIEF-SEARCH-EVAL-2026-10-10.md``), four
  remembered words verbatim found their passage in the search's top 10 only 73% of the time; a
  plain word search found it every time, usually as the only match.
* :func:`time_transcript_hits` — a real start time for the search's transcript results. The index
  stores none (``timestamp_start_ms`` is 0 on prod: its chunker times a passage only from segments
  carrying character offsets, which Whisper's do not), so "Play from" jumped to 0:00. The result's
  words are found in the timed transcript instead, at read time.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from podcast_scraper.server.app_content_source import transcript_corpus_relpath, transcript_relpath
from podcast_scraper.server.corpus_catalog import _load_metadata_doc, CatalogEpisodeRow
from podcast_scraper.server.segments_view import (
    segments_relpaths_for_transcript,
    to_contract_segments,
)
from podcast_scraper.utils.path_validation import safe_relpath_under_corpus_root

logger = logging.getLogger(__name__)

Segment = Tuple[int, int, str]  # (start_ms, end_ms, text)

_WORD = re.compile(r"\w+(?:'\w+)?")
#: Words a remembered query drops when it has others: "the worst week" is about worst + week.
_STOP = frozenset(
    "a an and are as at be but by for from had has have he her his i if in is it its me my no not "
    "of on or our she so that the their them they this to too us was we were what when which who "
    "will with you your".split()
)
#: How many consecutive segments one exact passage may span. A remembered phrase can straddle a
#: segment boundary (Whisper cuts mid-sentence), and three keeps a passage to a sentence or two.
_MAX_SPAN = 3
#: Consecutive words matched when locating a result's text in the transcript.
_ANCHOR = 6


def _words(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def _terms(query: str) -> List[str]:
    words = _words(query)
    content = [w for w in words if w not in _STOP]
    return content or words


def _hit(
    segments: Sequence[Segment], i: int, j: int, episode_id: str, match: str
) -> Dict[str, Any]:
    start_ms, end_ms = segments[i][0], segments[j][1]
    return {
        "doc_id": f"exact:{episode_id}:{start_ms}",
        "score": 1.0,
        "text": " ".join(segments[k][2].strip() for k in range(i, j + 1)),
        "source_tier": "segment",
        "metadata": {
            "doc_type": "transcript",
            "match": match,
            "episode_id": episode_id,
            "timestamp_start_ms": start_ms,
            "timestamp_end_ms": end_ms,
        },
    }


def exact_passages(
    segments: Sequence[Segment], query: str, *, episode_id: str, limit: int = 20
) -> List[Dict[str, Any]]:
    """Passages holding the remembered words verbatim (case-insensitive, whole words).

    First the words exactly as typed, in order (``phrase``) — found in the transcript's running
    words, so a phrase Whisper cut across two segments is found whole. Then passages with every
    content word in any order (``words``): the smallest run of up to ``_MAX_SPAN`` segments that
    holds them all. Each group in episode order; a passage is listed once.
    """
    typed = _words(query)
    terms = _terms(query)
    if not terms:
        return []
    words: List[str] = []
    word_seg: List[int] = []
    seg_words: List[List[str]] = []
    for k, (_, _, text) in enumerate(segments):
        ws = _words(text)
        seg_words.append(ws)
        words.extend(ws)
        word_seg.extend([k] * len(ws))

    found: List[Dict[str, Any]] = []
    taken: set[int] = set()
    n = len(typed)
    for p in range(len(words) - n + 1):
        if words[p : p + n] == typed and word_seg[p] not in taken:
            i, j = word_seg[p], word_seg[p + n - 1]
            found.append(_hit(segments, i, j, episode_id, "phrase"))
            taken.update(range(i, j + 1))

    wanted = set(terms)
    i = 0
    while i < len(segments):
        if i in taken or not wanted & set(seg_words[i]):
            i += 1
            continue
        for span in range(1, _MAX_SPAN + 1):
            j = i + span - 1
            if j >= len(segments) or j in taken:
                break
            if wanted <= {w for k in range(i, j + 1) for w in seg_words[k]}:
                found.append(_hit(segments, i, j, episode_id, "words"))
                taken.update(range(i, j + 1))
                i = j
                break
        i += 1
    return found[:limit]


def _locate(
    hit_words: Sequence[str], index: Dict[Tuple[str, ...], int], from_end: bool
) -> Optional[int]:
    """Transcript word position where the result's text is first (or last) found.

    The first run of ``_ANCHOR`` words that occurs in the transcript, from the start (or the end).
    Its own position, not projected back by the words before it: those can be a speaker label the
    timed segments do not carry ("Nora:"), and projecting over it lands in the segment before.
    """
    n = len(hit_words)
    if n < _ANCHOR:
        return None
    offsets = range(n - _ANCHOR, -1, -1) if from_end else range(0, n - _ANCHOR + 1)
    for off in offsets:
        pos = index.get(tuple(hit_words[off : off + _ANCHOR]))
        if pos is not None:
            return pos + _ANCHOR - 1 if from_end else pos
    return None


def time_transcript_hits(results: List[Any], segments: Sequence[Segment]) -> None:
    """Give each transcript result its real start and end time, found by its words.

    In place, on the response's hit models. A result whose words cannot be found keeps no time
    (both ``None``), so the app shows no "Play from" rather than one that jumps to 0:00.
    """
    word_seg: List[int] = []
    words: List[str] = []
    for k, (_, _, text) in enumerate(segments):
        ws = _words(text)
        words.extend(ws)
        word_seg.extend([k] * len(ws))
    index: Dict[Tuple[str, ...], int] = {}
    for p in range(len(words) - _ANCHOR + 1):
        index.setdefault(tuple(words[p : p + _ANCHOR]), p)
    for hit in results:
        md = hit.metadata if isinstance(hit.metadata, dict) else {}
        if md.get("doc_type") != "transcript" or md.get("match"):
            continue
        hit_words = _words(hit.text or "")
        first = _locate(hit_words, index, from_end=False)
        last = _locate(hit_words, index, from_end=True)
        timed = dict(md)
        if first is None or not words:
            timed["timestamp_start_ms"] = None
            timed["timestamp_end_ms"] = None
        else:
            first = min(max(first, 0), len(words) - 1)
            timed["timestamp_start_ms"] = segments[word_seg[first]][0]
            if last is not None:
                last = min(max(last, first), len(words) - 1)
                timed["timestamp_end_ms"] = segments[word_seg[last]][1]
        hit.metadata = timed


def episode_segments(root: Path, row: CatalogEpisodeRow) -> Optional[List[Segment]]:
    """The episode's canonical timed transcript as ``(start_ms, end_ms, text)``, or None.

    The same candidates, in the same order, as ``GET /segments`` serves by default: the raw
    canonical file first (its timestamps run on the original audio the player streams), the
    ad-free one only as a fallback.
    """
    doc = _load_metadata_doc(root / row.metadata_relative_path)
    content = doc.get("content") if isinstance(doc, dict) else None
    transcript_rel = transcript_relpath(content if isinstance(content, dict) else {})
    if transcript_rel is None:
        return None
    corpus_rel = transcript_corpus_relpath(row.metadata_relative_path, transcript_rel)
    for candidate in segments_relpaths_for_transcript(corpus_rel):
        safe = safe_relpath_under_corpus_root(root, candidate)
        if not safe or not (root / safe).is_file():
            continue
        try:
            raw = json.loads((root / safe).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Unreadable segments file %s: %s", root / safe, exc)
            return None
        return [(int(s.start * 1000), int(s.end * 1000), s.text) for s in to_contract_segments(raw)]
    return None


def time_hits_by_episode(root: Path, results: List[Any], rows: Iterable[CatalogEpisodeRow]) -> None:
    """:func:`time_transcript_hits` for results from many episodes (global search, Mine).

    One timed-transcript read per episode that has a transcript result, at most one per result.
    Episodes are matched by ``episode_id`` (and ``feed_id`` when the hit carries it).
    """
    by_scope: Dict[Tuple[str, str], CatalogEpisodeRow] = {}
    by_episode: Dict[str, CatalogEpisodeRow] = {}
    for row in rows:
        if row.episode_id:
            by_scope[(row.feed_id or "", row.episode_id)] = row
            by_episode.setdefault(row.episode_id, row)
    grouped: Dict[str, List[Any]] = {}
    for hit in results:
        md = hit.metadata if isinstance(hit.metadata, dict) else {}
        eid = md.get("episode_id")
        if md.get("doc_type") == "transcript" and not md.get("match") and isinstance(eid, str):
            grouped.setdefault(f"{md.get('feed_id') or ''}\x00{eid}", []).append(hit)
    for key, hits in grouped.items():
        feed, eid = key.split("\x00", 1)
        found = by_scope.get((feed, eid)) or by_episode.get(eid)
        segments = episode_segments(root, found) if found is not None else None
        if segments:
            time_transcript_hits(hits, segments)
        else:
            # No timed transcript to find it in: drop the false 0:00 rather than keep it.
            for hit in hits:
                hit.metadata = {
                    **hit.metadata,
                    "timestamp_start_ms": None,
                    "timestamp_end_ms": None,
                }
