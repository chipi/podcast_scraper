"""Map on-disk Whisper segment artifacts to the player ``segments.json`` contract.

Contract (PRD-036 / RFC-098): ``{id, start, end, text, speaker?}`` per segment, with
``start``/``end`` in seconds. Pure functions — unit-tested without HTTP or disk.
"""

from __future__ import annotations

import math
from typing import Any

from podcast_scraper.graph_id_utils import is_bare_speaker_label
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

    A NAMED voice (real ``speaker_label``, not a raw ``SPEAKER_NN``) shows its name — even though
    the segment now also carries its host/guest ``speaker_role``. Only an UNNAMED voice uses the
    friendly label ("Host" / "Brief speaker" / "Advertisement"). This maps display only — the
    id-bearing ``speaker_label`` in the artifact is untouched (the GI still owns the ids)."""
    # An UNNAMED voice carries a ``voice_type`` (cameo/commercial) or a raw ``SPEAKER_NN`` label and
    # renders a friendly label. A NAMED voice has a real label, no ``voice_type``, and (now) a
    # host/guest ``speaker_role`` — show the name, not "Host".
    label = raw.get("speaker_label")
    if (
        isinstance(label, str)
        and label.strip()
        and not is_bare_speaker_label(label)
        and not raw.get("voice_type")
    ):
        return label.strip()
    friendly = friendly_speaker_label(raw.get("speaker_role"), raw.get("voice_type"))
    if friendly:
        return friendly
    for key in ("speaker_label", "speaker_id", "speaker"):
        val = raw.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


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
        out.append(
            TranscriptSegment(
                id=sid,
                start=start,
                end=end,
                text=text,
                speaker=_segment_speaker(raw),
            )
        )
    out.sort(key=lambda s: s.start)
    return out
