"""Map on-disk Whisper segment artifacts to the player ``segments.json`` contract.

Contract (PRD-036 / RFC-098): ``{id, start, end, text, speaker?, speaker_kind?, speaker_index?}``
per segment, with ``start``/``end`` in seconds. Pure functions — unit-tested without HTTP or disk.
"""

from __future__ import annotations

import math
from typing import Any

from podcast_scraper.graph_id_utils import is_bare_speaker_label, is_raw_speaker_label
from podcast_scraper.providers.ml.diarization.roster import (
    UNNAMED_HOST_LABEL,
    VOICE_CAMEO,
    VOICE_COMMERCIAL,
    VOICE_TYPE_LABELS,
    VOICE_UNIDENTIFIED,
    VOICE_UNKNOWN,
    friendly_speaker_label,
)
from podcast_scraper.server.schemas import TranscriptSegment

#: speaker_kind values (the contract's vocabulary; clients localise from them, RFC-131 L7).
KIND_PERSON = "person"
KIND_HOST = "host"
KIND_BRIEF = "brief"
KIND_ADVERTISEMENT = "advertisement"
KIND_UNIDENTIFIED = "unidentified"
KIND_ROLE = "role"


def segments_relpaths_for_transcript(transcript_relpath: str) -> list[str]:
    """Candidate segment-file relpaths for a transcript file — **raw canonical preferred**.

    ``transcripts/ep1.txt`` -> ``[transcripts/ep1.segments.json,
    transcripts/ep1.adfree.segments.json]``. A trailing ``.adfree`` on the stem is stripped
    so both ``ep1.txt`` and ``ep1.adfree.txt`` resolve to the same candidates.

    The consumer Player streams the **original (unbridged) audio** — ads included — so its
    transcript-sync must use the **raw canonical** segments, whose timestamps run on the
    original timeline. The ad-free segments (ads removed) are minutes shorter and would drift
    the highlight/seek against the played audio; they are only a last-resort fallback here.
    """
    rel = (transcript_relpath or "").strip().replace("\\", "/")
    if not rel:
        return []
    base = rel[:-4] if rel.lower().endswith(".txt") else rel
    if base.lower().endswith(".adfree"):
        base = base[: -len(".adfree")]
    return [f"{base}.segments.json", f"{base}.adfree.segments.json"]


def _segment_speaker(raw: dict[str, Any]) -> tuple[str | None, str | None]:
    """``(kind, text)`` for a segment's speaker; ``text`` is None for an unidentified voice.

    A NAMED voice (a real ``speaker_label``, no ``voice_type``) shows its name, even though the
    segment also carries its host/guest ``speaker_role``. An UNNAMED voice gets a label: "Host"
    for an unnamed host, "Brief speaker" for a cameo, "Advertisement" for an ad read. Every other
    unnamed voice is UNIDENTIFIED, and :func:`to_contract_segments` numbers it.

    A raw diarization label (``SPEAKER_07``) or the detector's default (``unknown_guest_1``) is
    never shown (RFC-131, S3, operator decision 2026-10-30). The roster leaves a substantive
    unknown voice with its raw id by design (``friendly_speaker_label``); that stays true in the
    artifact, and here it becomes "Unidentified speaker N".

    This maps display only; the id-bearing ``speaker_label`` in the artifact is untouched.
    """
    label = raw.get("speaker_label")
    label = label.strip() if isinstance(label, str) else ""
    voice_type = raw.get("voice_type")
    if label and not is_bare_speaker_label(label) and not voice_type:
        return KIND_PERSON, label
    if voice_type in (VOICE_UNKNOWN, VOICE_UNIDENTIFIED):
        return KIND_UNIDENTIFIED, None
    friendly = friendly_speaker_label(raw.get("speaker_role"), voice_type)
    if friendly == UNNAMED_HOST_LABEL:
        return KIND_HOST, friendly
    if voice_type == VOICE_CAMEO and friendly:
        return KIND_BRIEF, friendly
    if voice_type == VOICE_COMMERCIAL and friendly:
        return KIND_ADVERTISEMENT, friendly
    for key in ("speaker_label", "speaker_id", "speaker"):
        val = raw.get(key)
        if not isinstance(val, str) or not val.strip():
            continue
        val = val.strip()
        if is_raw_speaker_label(val) or val.casefold() == _UNIDENTIFIED_TEXT.casefold():
            return KIND_UNIDENTIFIED, None
        # A role word ("Guest") or anything else non-raw keeps today's display.
        return (KIND_ROLE if is_bare_speaker_label(val) else KIND_PERSON), val
    return None, None


#: "Unidentified speaker", the label the roster defines for a voice nobody names.
_UNIDENTIFIED_TEXT = VOICE_TYPE_LABELS[VOICE_UNIDENTIFIED]


def _voice_key(raw: dict[str, Any]) -> str:
    """Which voice a segment is, for numbering: the diarized voice id, else its label."""
    for key in ("speaker", "speaker_id", "speaker_label"):
        val = raw.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def to_contract_segments(raw_segments: Any) -> list[TranscriptSegment]:
    """Map a raw Whisper segment list to contract segments; skip malformed entries.

    The output is SORTED BY START. The client's ``activeSegmentIndex`` is a binary search whose
    comment says "the contract guarantees this" — and nothing enforced it, so an out-of-order
    artifact did not fail, it returned a plausible WRONG segment: highlight-follow tracked the wrong
    paragraph and tap-to-seek jumped to the wrong moment, silently. A guarantee the contract asserts
    is the contract's job to keep. Sorting is stable, so equal starts keep their file order and the
    positional ``seg_NNNN`` ids stay in the order the transcript wrote them.

    Non-finite times are dropped alongside the other malformed entries. ``json.loads`` accepts the
    non-standard ``NaN`` / ``Infinity`` tokens, and one of them anywhere in the list makes the whole
    response body unserialisable (Starlette renders with ``allow_nan=False``) — which the player
    then reports as "Transcript pending", so a single bad number reads to the user as a missing
    transcript, forever.
    """
    out: list[TranscriptSegment] = []
    rows: list[tuple[float, float, str, str, str | None, str | None, str]] = []
    if not isinstance(raw_segments, list):
        return out
    for idx, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            continue
        try:
            start = float(raw["start"])
            end = float(raw["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end)):
            continue
        text = raw.get("text")
        if not isinstance(text, str):
            continue
        seg_id = raw.get("id")
        sid = f"seg_{seg_id:04d}" if isinstance(seg_id, int) else f"seg_{idx:04d}"
        kind, speaker = _segment_speaker(raw)
        rows.append((start, end, text, sid, kind, speaker, _voice_key(raw)))
    # Sorted before numbering, so "first appearance" is first in time.
    rows.sort(key=lambda r: r[0])
    numbers: dict[str, int] = {}
    for start, end, text, sid, kind, speaker, voice in rows:
        index = None
        if kind == KIND_UNIDENTIFIED:
            index = numbers.setdefault(voice, len(numbers) + 1)
            speaker = f"{_UNIDENTIFIED_TEXT} {index}"
        out.append(
            TranscriptSegment(
                id=sid,
                start=start,
                end=end,
                text=text,
                speaker=speaker,
                speaker_kind=kind,
                speaker_index=index,
            )
        )
    return out
