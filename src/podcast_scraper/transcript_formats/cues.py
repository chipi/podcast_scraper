"""Parse WebVTT and SubRip captions into plain text and GI-compatible segment lists.

Each segment is ``{"start": float, "end": float, "text": str}`` (seconds), matching
Whisper-style sidecars used by ``_char_range_to_ms``. Plain text is the concatenation
of segment ``text`` values with no separator (exact length alignment for issue #545).

A WebVTT cue may also carry a VOICE SPAN — ``<v Speaker 3>`` — naming who is talking. Where it
does, the segment gains a ``"speaker"`` key, which is the same shape diarization produces, so a
publisher-supplied transcript reaches the roster exactly as an audio-derived one does.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# WebVTT cue timing: optional hours; comma or dot for fractional seconds.
# hours are ``\d+`` (not ``\d{1,2}``) so a >99h recording's later cues aren't
# silently dropped by a non-matching line (review 2026-07-17 low/cues).
_WEBVTT_CUE_LINE = re.compile(
    r"^(\d+:\d{2}(?::\d{2})?[.,]\d{3})\s*-->\s*(\d+:\d{2}(?::\d{2})?[.,]\d{3})"
)
# SRT typical line (hours optional in some files; require full h:m:s)
_SRT_CUE_LINE = re.compile(r"^(\d+:\d{2}:\d{2},\d{3})\s*-->\s*(\d+:\d{2}:\d{2},\d{3})")
_HTML_TAG = re.compile(r"<[^>]+>")
# WebVTT voice span: `<v Speaker 3>`, `<v.loud Mark>`, `<v Joe Wiesenthal>`. The name runs to the
# closing angle bracket; optional `.class` suffixes on the tag itself are not part of it.
# The class must EXCLUDE `.`, or `(?:\.[^\s>]+)*` is ambiguous with itself — `.a.b` can be read
# as one repetition or two, and the engine tries every split. CodeQL: exponential backtracking on
# `<v.` followed by many `.x`. Excluding the dot makes each `.segment` match exactly one way.
_VOICE_SPAN = re.compile(r"<v(?:\.[^\s>.]+)*\s+([^>]+)>")
# A speaker written as a CUE-TEXT prefix — `Speaker 3: …` — rather than as a voice tag. SubRip has
# no voice tag, so this was added for it (Odd Lots, whose feed lists the SRT FIRST, so a fixed
# WebVTT parser never saw its speakers).
#
# WebVTT NEEDS IT TOO, and not having it cost 50 episodes. Measured on prod 2026-09-26: six Odd Lots
# episodes published a roster of two hosts and NO guest while their titles named one, because their
# WebVTT writes each turn as text (`Speaker 1: …`) instead of `<v Speaker 1>`. `parse_webvtt` only
# read the tag, so the label stayed embedded in the prose — 280 literal `Speaker 1` strings in one
# 81 KB transcript with `speaker=None` on all 1,290 segments — where it corrupts BOTH attribution
# (which matches markers at line start) and chunking (which finds no sentence boundary). 51 corpus
# episodes carry that shape. Identical input gave opposite results from the two parsers.
#
# Deliberately only the generic `Speaker N` form: a free `<Name>:` prefix is indistinguishable
# from prose ("Note: …"), and In Moscow's Shadows writes its `MG:` on the first cue only.
_CUE_SPEAKER_PREFIX = re.compile(r"^\s*(Speaker\s+\d+)\s*:\s*", re.IGNORECASE)
#: Retained name — this pattern was SubRip-only until WebVTT started using it too.
_SRT_SPEAKER_PREFIX = _CUE_SPEAKER_PREFIX


def _separate_cues(segments: List[Dict[str, Any]]) -> None:
    """End a cue with a space when neither it nor the next cue carries one at the boundary.

    Plain text is the concatenation of cue texts (exact alignment, #545), and publishers cut cues
    between words without a trailing space — so "...Kansas City Fed President" + "Jeff Schmidt"
    became "PresidentJeff Schmidt" in the stored transcript. Measured on the production snapshot:
    94,927 of 233,953 cue boundaries (41%) across 249 publisher-transcript episodes were glued, and
    every sampled one sat between two words. The space goes INTO the preceding segment's text, so
    plain text is still exactly the segments joined and character offsets stay aligned.
    """
    for a, b in zip(segments, segments[1:]):
        ta, tb = a["text"], b["text"]
        if ta and tb and not ta[-1].isspace() and not tb[0].isspace():
            a["text"] = ta + " "


def _timestamp_to_seconds(ts: str) -> float:
    """Parse VTT/SRT timestamp fragment to seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(ts)


#: The escapes WebVTT cue text is allowed to carry, in the order they must be applied.
#:
#: WHY NOT ``html.unescape``. It decodes the HTML5 legacy set, which is wrong for a cue file in two
#: measured ways: ``"&copy 2026"`` (no semicolon — an ampersand followed by a word) becomes
#: ``"© 2026"``, silently rewriting text the publisher wrote literally; and ``&nbsp;`` becomes
#: ``\xa0``, which the ``[ \t]+`` collapse below does not treat as whitespace, so it survives into
#: the stored transcript, the search index and every quote span. The cue-text grammar defines
#: exactly these five, so decode exactly these five.
#:
#: ``&amp;`` MUST be last: decoding it first would turn ``&amp;lt;`` into ``&lt;`` and then into
#: ``<``, resurrecting markup the publisher escaped on purpose.
_CUE_ESCAPES = (
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&nbsp;", " "),
    ("&lrm;", ""),
    ("&rlm;", ""),
    ("&amp;", "&"),
)


def _normalize_cue_text(raw: str) -> str:
    """Strip simple HTML-like tags, decode cue escapes; newlines to space; collapse runs.

    Leading/trailing spaces within a cue are preserved so adjacent cues can form
    ``"Hello world"`` when the second cue begins with a space.

    ENTITIES ARE DECODED HERE, per segment, and that placement is load-bearing. ``plain`` is
    ``"".join(s["text"] for s in segments)`` and GI quote char offsets are derived from it (#545),
    so decoding on the joined string instead would shorten the text out from under every segment
    offset. Decoding here keeps segment text and plain text in one coordinate space by
    construction.

    Tags are stripped BEFORE escapes are decoded, so a literal ``&lt;b&gt;`` the publisher escaped
    survives as the text ``<b>`` instead of being stripped as markup. ``_VOICE_SPAN`` reads the RAW
    cue, so speaker labels are unaffected either way.
    """
    t = _HTML_TAG.sub("", raw)
    for entity, char in _CUE_ESCAPES:
        t = t.replace(entity, char)
    t = t.replace("\n", " ").replace("\r", " ")
    return re.sub(r"[ \t]+", " ", t)


def parse_webvtt(data: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Parse WebVTT body into ``(plain_text, segments)``.

    Returns empty segments if the file has no usable cues (caller should fall back
    to writing raw bytes).
    """
    lines = data.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[0].startswith("\ufeff"):
        lines[0] = lines[0].lstrip("\ufeff")

    i = 0
    while i < len(lines) and not lines[i].strip().upper().startswith("WEBVTT"):
        i += 1
    if i >= len(lines):
        return "", []
    i += 1
    while i < len(lines) and lines[i].strip():
        i += 1

    segments: List[Dict[str, Any]] = []
    while i < len(lines):
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break

        stripped = lines[i].strip()
        if stripped.startswith("NOTE"):
            i += 1
            while i < len(lines) and lines[i].strip():
                i += 1
            continue
        if stripped.startswith("STYLE") or stripped.startswith("REGION"):
            i += 1
            while i < len(lines) and lines[i].strip():
                i += 1
            continue

        if "-->" not in stripped:
            if i + 1 < len(lines) and "-->" in lines[i + 1]:
                i += 1
                if i >= len(lines):
                    break
                stripped = lines[i].strip()
            else:
                i += 1
                continue

        m = _WEBVTT_CUE_LINE.match(stripped)
        if not m:
            i += 1
            continue

        start_s = _timestamp_to_seconds(m.group(1))
        end_s = _timestamp_to_seconds(m.group(2))
        i += 1
        text_lines: List[str] = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i])
            i += 1
        raw_text = "\n".join(text_lines)
        # WHO IS TALKING, WHEN THE FILE SAYS SO. `_normalize_cue_text` strips `<v Speaker 3>` as an
        # HTML-like tag, so the speaker was being deleted before anything could read it: a
        # publisher transcript that names every turn arrived as one undifferentiated voice, and the
        # episode could never carry host/guest attribution however good naming became. Measured on
        # the production corpus, EVERY episode that used a publisher transcript ended with a single
        # voice — 128 of 128. Odd Lots' own WebVTT carries 781 voice spans and names both hosts in
        # the first minute.
        voice = _VOICE_SPAN.search(raw_text)
        # A text prefix is the OTHER way a WebVTT names its turns, and the tag wins when both are
        # present: explicit markup beats a text convention. Stripped from the cue body exactly as
        # `parse_srt` does, so the label does not survive into the prose it would otherwise corrupt.
        prefix = None if voice else _CUE_SPEAKER_PREFIX.match(raw_text)
        if prefix:
            raw_text = raw_text[prefix.end() :]
        norm = _normalize_cue_text(raw_text)
        if norm.strip():
            seg: Dict[str, Any] = {"start": start_s, "end": end_s, "text": norm}
            if voice:
                speaker = voice.group(1).strip()
                if speaker:
                    seg["speaker"] = speaker
            elif prefix:
                seg["speaker"] = " ".join(prefix.group(1).split())
            segments.append(seg)

    _separate_cues(segments)
    plain = "".join(s["text"] for s in segments)
    return plain, segments


def parse_srt(data: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Parse SubRip body into ``(plain_text, segments)``."""
    text = data.replace("\r\n", "\n").replace("\r", "\n")
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    blocks = re.split(r"\n\s*\n+", text.strip())
    segments: List[Dict[str, Any]] = []

    for block in blocks:
        block_lines = [ln for ln in block.split("\n") if ln is not None]
        if not block_lines:
            continue
        li = 0
        if re.match(r"^\d+\s*$", block_lines[0].strip()):
            li = 1
        if li >= len(block_lines):
            continue
        time_line = block_lines[li].strip()
        m = _SRT_CUE_LINE.match(time_line)
        if not m:
            continue
        start_s = _timestamp_to_seconds(m.group(1))
        end_s = _timestamp_to_seconds(m.group(2))
        raw_body = "\n".join(block_lines[li + 1 :])
        prefix = _SRT_SPEAKER_PREFIX.match(raw_body)
        if prefix:
            raw_body = raw_body[prefix.end() :]
        norm = _normalize_cue_text(raw_body)
        if not norm.strip():
            continue
        seg: Dict[str, Any] = {"start": start_s, "end": end_s, "text": norm}
        if prefix:
            seg["speaker"] = " ".join(prefix.group(1).split())
        segments.append(seg)

    _separate_cues(segments)
    plain = "".join(s["text"] for s in segments)
    return plain, segments
