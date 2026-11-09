"""How a transcript segment's speaker is shown: the speaker stage's voice-type vocabulary mapped to
the display kinds (moved verbatim from ``server.segments_view``, E04 S3).
"""

from __future__ import annotations

from typing import Any

from podcast_scraper.graph_id_utils import is_bare_speaker_label, is_raw_speaker_label
from podcast_scraper.providers.ml.diarization.roster import (
    friendly_speaker_label,
    UNNAMED_HOST_LABEL,
    VOICE_CAMEO,
    VOICE_COMMERCIAL,
    VOICE_TYPE_LABELS,
    VOICE_UNIDENTIFIED,
    VOICE_UNKNOWN,
)
from podcast_scraper.server.segments_view import (
    KIND_ADVERTISEMENT,
    KIND_BRIEF,
    KIND_HOST,
    KIND_PERSON,
    KIND_ROLE,
    KIND_UNIDENTIFIED,
)


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
