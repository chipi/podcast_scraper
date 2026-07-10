"""Map on-disk Whisper segment artifacts to the player ``segments.json`` contract.

Contract (PRD-036 / RFC-098): ``{id, start, end, text, speaker?}`` per segment, with
``start``/``end`` in seconds. Pure functions — unit-tested without HTTP or disk.
"""

from __future__ import annotations

from typing import Any

from podcast_scraper.providers.ml.diarization.roster import friendly_speaker_label
from podcast_scraper.server.schemas import TranscriptSegment


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


def _segment_speaker(raw: dict[str, Any]) -> str | None:
    """Best display speaker for a segment: a real name, else "Host" for an unnamed host, else a
    friendly type label for a cameo/commercial voice ("Brief speaker" / "Advertisement"), else the
    raw diarization tag.

    ``speaker_role`` / ``voice_type`` are set by the roster only for an *unnamed* voice; a named
    voice has a real ``speaker_label`` and neither, so it is returned as-is. This maps display only
    — the id-bearing ``speaker_label`` in the artifact is untouched (the GI still owns the ids)."""
    friendly = friendly_speaker_label(raw.get("speaker_role"), raw.get("voice_type"))
    if friendly:
        return friendly
    for key in ("speaker_label", "speaker_id", "speaker"):
        val = raw.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def to_contract_segments(raw_segments: Any) -> list[TranscriptSegment]:
    """Map a raw Whisper segment list to contract segments; skip malformed entries."""
    out: list[TranscriptSegment] = []
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
        text = raw.get("text")
        if not isinstance(text, str):
            continue
        seg_id = raw.get("id")
        sid = f"seg_{seg_id:04d}" if isinstance(seg_id, int) else f"seg_{idx:04d}"
        out.append(
            TranscriptSegment(
                id=sid,
                start=start,
                end=end,
                text=text,
                speaker=_segment_speaker(raw),
            )
        )
    return out
