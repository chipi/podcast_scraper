"""Search within one episode for what a listener remembers (operator 2026-10-10).

The use case: a listener remembers a few words from an episode and wants that part again, to
highlight it or to find the insight about it. Two additions to the Brief's search, both reading
only the episode's timed transcript (its ``segments.json``) — no index, no model:

* :func:`exact_passages` — the passages that contain every remembered word, verbatim, with their
  real times. Measured on 94 real episodes (``docs/wip/BRIEF-SEARCH-EVAL-2026-10-10.md``), four
  remembered words verbatim found their passage in the search's top 10 only 73% of the time; a
  plain word search found it every time, usually as the only match.
* :func:`time_transcript_hits` — a real start time for the search's transcript results. The index
  held none (``timestamp_start_ms`` 0 on prod: the indexer read times only from
  ``summary.timestamps``, which the pipeline does not write), so "Play from" jumped to 0:00. The
  indexer now times chunks from the segments and migration 0028 times the rows already indexed;
  this still finds the result's words in the timed transcript, which narrows a passage-long chunk
  to the moment they were said, and covers an index not yet migrated.
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


def _form(word: str) -> str:
    """The same key for a word's inflections: plural, -ed, -ing (operator 2026-10-10).

    "invested" / "investing" / "invests" -> "invest"; "policies" -> "policy"; "making" / "make"
    -> "mak". Deliberately not a stemmer: one only strips inflections, so "police" and "policy"
    stay apart (the over-merge ``graph_id_utils`` avoids for the same reason).
    """
    w = word.lower()
    if len(w) <= 3:
        return w
    if w.endswith(("ies", "ied")) and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) > 5:
        base = w[:-3]
    elif w.endswith("ed") and len(w) > 4:
        base = w[:-2]
    elif w.endswith(("ches", "shes", "sses", "xes", "zes")):
        base = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        base = w[:-1]
    else:
        base = w
    if len(base) > 3 and base[-1] == base[-2] and base[-1] not in "lsz":
        base = base[:-1]  # the doubled consonant goes: running and stopped meet run and stop
    if len(base) > 3 and base.endswith("e"):
        base = base[:-1]  # make -> mak, so making and make meet
    return base


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
    segments: Sequence[Segment], query: str, *, episode_id: str, limit: Optional[int] = 20
) -> List[Dict[str, Any]]:
    """Passages holding the remembered words verbatim (case-insensitive, whole words).

    First the words exactly as typed, in order (``phrase``) — found in the transcript's running
    words, so a phrase Whisper cut across two segments is found whole. Then passages with every
    content word in any order (``words``): the smallest run of up to ``_MAX_SPAN`` segments that
    holds them all. Then what memory changes (eval: a word in another form, or one word
    misremembered, was found by these exact tiers 1-5% of the time): every word in some form
    (``forms``: "invested" finds "investing"), and, for three or more words, all but one of them
    (``most``). Each group in episode order; a passage is listed once, and carries the transcript
    words it matched (``marks``) so the reader can see why. ``limit=None`` returns every passage
    (the Brief's word search, operator 2026-10-12: not limited).
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

    for p_hit in found:
        p_hit["metadata"]["marks"] = sorted(set(typed))

    def spans(keys: List[set], wanted: set, need: int, match: str, key: Any) -> None:
        i = 0
        while i < len(segments):
            if i in taken or not wanted & keys[i]:
                i += 1
                continue
            for span in range(1, _MAX_SPAN + 1):
                j = i + span - 1
                if j >= len(segments) or j in taken:
                    break
                if len(wanted & set().union(*keys[i : j + 1])) >= need:
                    hit = _hit(segments, i, j, episode_id, match)
                    hit["metadata"]["marks"] = sorted(
                        {w for k in range(i, j + 1) for w in seg_words[k] if key(w) in wanted}
                    )
                    found.append(hit)
                    taken.update(range(i, j + 1))
                    i = j
                    break
            i += 1

    exact = set(terms)
    spans([set(ws) for ws in seg_words], exact, len(exact), "words", lambda w: w)
    forms = {_form(t) for t in terms}
    seg_forms = [{_form(w) for w in ws} for ws in seg_words]
    spans(seg_forms, forms, len(forms), "forms", _form)
    if len(forms) >= 3:
        spans(seg_forms, forms, len(forms) - 1, "most", _form)
    return found[:limit]


def word_matches(results: List[Any], query: str) -> List[Any]:
    """The results that hold every remembered word, in some form; the rest are dropped.

    The Brief's search is a word search (operator 2026-10-12): a result found only by meaning, with
    none of the typed words in it, read as a wrong answer ("Intervention" returned a welcome line).
    The index ranks by words and meaning together, so its results are kept only when their own text
    has each content word in some form (:func:`_form`: "interventions" for "intervention"), and they
    carry the words they matched (``marks``) like the verbatim passages do. In place on each kept
    hit's metadata; the order is the index's.
    """
    forms = {_form(t) for t in _terms(query)}
    if not forms:
        return []
    kept: List[Any] = []
    for hit in results:
        ws = _words(hit.text or "")
        if not forms <= {_form(w) for w in ws}:
            continue
        md = dict(hit.metadata) if isinstance(hit.metadata, dict) else {}
        md["marks"] = sorted({w for w in ws if _form(w) in forms})
        hit.metadata = md
        kept.append(hit)
    return kept


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


def episode_segments_relpath(root: Path, row: CatalogEpisodeRow) -> Optional[str]:
    """The corpus-relative path of the file :func:`episode_segments` reads, or None."""
    doc = _load_metadata_doc(root / row.metadata_relative_path)
    content = doc.get("content") if isinstance(doc, dict) else None
    transcript_rel = transcript_relpath(content if isinstance(content, dict) else {})
    if transcript_rel is None:
        return None
    corpus_rel = transcript_corpus_relpath(row.metadata_relative_path, transcript_rel)
    for candidate in segments_relpaths_for_transcript(corpus_rel):
        safe = safe_relpath_under_corpus_root(root, candidate)
        if safe and (root / safe).is_file():
            return str(safe)
    return None


def episode_segments(root: Path, row: CatalogEpisodeRow) -> Optional[List[Segment]]:
    """The episode's canonical timed transcript as ``(start_ms, end_ms, text)``, or None.

    The same candidates, in the same order, as ``GET /segments`` serves by default: the raw
    canonical file first (its timestamps run on the original audio the player streams), the
    ad-free one only as a fallback.
    """
    safe = episode_segments_relpath(root, row)
    if safe is None:
        return None
    raw = _read_segments_json(root / safe)
    return None if raw is None else _segments_of(raw)


def _read_segments_json(path: Path) -> Any:
    """The parsed segments file, or None (logged) when it cannot be read."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable segments file %s: %s", path, exc)
        return None


def _segments_of(raw: Any) -> List[Segment]:
    return [(int(s.start * 1000), int(s.end * 1000), s.text) for s in to_contract_segments(raw)]


def _segment_dicts_of(raw: Any) -> List[Dict[str, Any]]:
    segs = raw.get("segments", raw) if isinstance(raw, dict) else raw
    return [s for s in segs if isinstance(s, dict)] if isinstance(segs, list) else []


def _segment_dicts(path: Path) -> List[Dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return _segment_dicts_of(raw)


def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for lo, hi in sorted(spans):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        elif hi > lo:
            out.append((lo, hi))
    return out


def _detected_ad_reads(raw: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """The ad READS inside the detector's ranges, not the ranges themselves.

    A detected range is the stretch the pipeline would cut, and it can sweep in content: in a
    fixture episode with a pre-roll at 12 s and a break at 42 s it ran 0-54 s, over two real
    points. A clip only needs to never play an ad, so here a segment is an ad when it lies in a
    range AND its own text reads as one, in any language the ad patterns cover
    (``gi.filters.AD_PATTERNS_BY_LANGUAGE``). Reads one line apart in the
    same range join up, so a sponsor read split across three segments is one span."""
    from podcast_scraper.gi.ad_regions import ad_time_intervals
    from podcast_scraper.gi.filters import AD_PATTERNS_BY_LANGUAGE

    ranges = ad_time_intervals(raw)
    if not ranges:
        return []
    patterns = [p for group in AD_PATTERNS_BY_LANGUAGE.values() for p in group]

    def range_of(seg: Dict[str, Any]) -> Optional[int]:
        mid = (float(seg.get("start") or 0.0) + float(seg.get("end") or 0.0)) / 2.0
        return next((i for i, (a, b) in enumerate(ranges) if a <= mid <= b), None)

    reads: List[Tuple[int, int]] = []  # (index in raw, which range)
    for i, seg in enumerate(raw):
        r = range_of(seg)
        if r is not None and any(p.search(str(seg.get("text") or "")) for p in patterns):
            reads.append((i, r))
    spans: List[Tuple[int, int]] = []
    prev: Optional[Tuple[int, int]] = None
    for i, r in reads:
        lo = int(float(raw[i].get("start") or 0.0) * 1000)
        hi = int(float(raw[i].get("end") or 0.0) * 1000)
        if prev is not None and prev[1] == r and i - prev[0] <= 2:
            spans[-1] = (spans[-1][0], hi)
        else:
            spans.append((lo, hi))
        prev = (i, r)
    return spans


def ad_spans_from_segments(
    raw: List[Dict[str, Any]], adfree: Optional[List[Dict[str, Any]]]
) -> List[Tuple[int, int]]:
    """Ad stretches of the raw timeline as ``(start_ms, end_ms)``, merged and sorted.

    From what is already on disk, in this order of evidence: with the ad-free segments the
    pipeline wrote, the raw segments it left out (the ad-free file keeps raw start times, so a
    missing start is a cut) — the very cut every insight was extracted after. Without that file
    (or with an empty one), the ad reads the same detector finds (:func:`_detected_ad_reads`).
    Either way, any segment diarization voiced as ``commercial``.
    """
    spans: List[Tuple[int, int]] = []
    if adfree:
        kept = {round(float(s.get("start") or 0.0), 2) for s in adfree}
        spans += [
            (int(float(s.get("start") or 0.0) * 1000), int(float(s.get("end") or 0.0) * 1000))
            for s in raw
            if round(float(s.get("start") or 0.0), 2) not in kept
        ]
    else:
        spans += _detected_ad_reads(raw)
    spans += [
        (int(float(s.get("start") or 0.0) * 1000), int(float(s.get("end") or 0.0) * 1000))
        for s in raw
        if s.get("voice_type") == "commercial"
    ]
    return _merge_spans(spans)


def episode_ad_spans(root: Path, row: CatalogEpisodeRow) -> List[Tuple[int, int]]:
    """:func:`ad_spans_from_segments` for an episode's raw canonical segments, cached until the
    file changes. ``[]`` when the episode has no raw segments (an ad-free file alone has no ads to
    find, and its timeline is the one the clips would then use)."""
    from podcast_scraper import perf_cache

    safe = episode_segments_relpath(root, row)
    if safe is None or safe.endswith(".adfree.segments.json"):
        return []
    path = root / safe
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []

    spans: List[Tuple[int, int]] = perf_cache.get_or_compute(
        "app_episode_ad_spans", str(path), mtime, lambda: _ad_spans_at(path, _segment_dicts(path))
    )
    return spans


def _ad_spans_at(path: Path, raw_dicts: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """:func:`ad_spans_from_segments` for the raw segments of ``path``, with its ad-free file."""
    adfree_path = path.with_name(path.name[: -len(".segments.json")] + ".adfree.segments.json")
    adfree = _segment_dicts(adfree_path) if adfree_path.is_file() else None
    return ad_spans_from_segments(raw_dicts, adfree)


def episode_turn_starts(root: Path, row: CatalogEpisodeRow) -> frozenset[int]:
    """Start (ms) of every segment whose speaker differs from the one before it: where one person
    stops and another starts. Unknown speakers are never a change. Cached until the file changes.

    A short listen is a clip of the person making the point, so it ends where they do — the next
    voice was a host's sponsor credit or native ad as often as a question (2026-10-12)."""
    from podcast_scraper import perf_cache

    safe = episode_segments_relpath(root, row)
    if safe is None:
        return frozenset()
    path = root / safe
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return frozenset()

    def build() -> frozenset[int]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return frozenset()
        return _turn_starts_of(raw)

    turns: frozenset[int] = perf_cache.get_or_compute(
        "app_episode_turn_starts", str(path), mtime, build
    )
    return turns


def _turn_starts_of(raw: Any) -> frozenset[int]:
    starts: set[int] = set()
    prev: Optional[str] = None
    for seg in to_contract_segments(raw):
        if seg.speaker and prev and seg.speaker != prev:
            starts.add(int(seg.start * 1000))
        if seg.speaker:
            prev = seg.speaker
    return frozenset(starts)


def episode_timeline(
    root: Path, row: CatalogEpisodeRow
) -> Tuple[Optional[List[Segment]], List[Tuple[int, int]], frozenset[int]]:
    """:func:`episode_segments`, :func:`episode_ad_spans` and :func:`episode_turn_starts` from ONE
    read of the metadata and the segments file, uncached: for scoring every episode's reel clips,
    where each of the three read both files again (2026-10-16: 3 parses of each per episode,
    ~30% of the scoring time)."""
    safe = episode_segments_relpath(root, row)
    if safe is None:
        return None, [], frozenset()
    path = root / safe
    raw = _read_segments_json(path)
    if raw is None:
        return None, [], frozenset()
    ads = (
        [] if safe.endswith(".adfree.segments.json") else _ad_spans_at(path, _segment_dicts_of(raw))
    )
    return _segments_of(raw), ads, _turn_starts_of(raw)


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
