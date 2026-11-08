"""Speaker-stage sidecars written next to a transcript (moved from
``workflow.episode_processor``, E04 S2).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _write_anon_transcript(
    segments: List[Dict[str, Any]],
    rel_transcript_path: str,
    effective_output_dir: str,
) -> Optional[str]:
    """Write the PRE-NAMING render, ``<base>.anon.txt`` (D-40). Returns its relpath, or ``None``.

    WHY IT EXISTS. The order is diarize → write anonymous → name → re-render, and the naming
    re-render overwrites ``.txt``. That
    would destroy the only human-readable view of what the pipeline saw BEFORE it decided who was
    speaking — the first thing anyone debugging a naming failure wants.

    RENDERED FROM ``speaker``, NOT ``speaker_label``. The voice id is frozen at diarization and
    naming never touches it; only ``speaker_label`` is updated. So this is derivable at any point
    after diarization, and writing it here is cheapness and clarity rather than recoverability.

    ``None`` WHEN IT WOULD CARRY NOTHING NEW: undiarized segments (no ``speaker``), or a render
    byte-identical to the named one because naming resolved no voice. An identical copy is not a
    second artifact, and writing one for every unnamed episode would double the transcript corpus
    to say nothing.

    NEVER A RESOLUTION CANDIDATE. ``text_relpath_candidates`` does not list ``.anon.txt`` for
    either purpose, deliberately: a reader that resolved to it would show ``SPEAKER_01`` where a
    person's name belongs.
    """
    from podcast_scraper.providers.ml.diarization.formatting import (
        format_diarized_screenplay_with_offsets,
    )
    from podcast_scraper.workflow.transcript_resolution import anon_transcript_relpath

    anonymous = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        voice = seg.get("speaker")
        if not voice:
            return None
        anonymous.append({**seg, "speaker_label": str(voice)})
    if not anonymous:
        return None

    anon_text, _offsets = format_diarized_screenplay_with_offsets(anonymous)
    if not anon_text.strip():
        return None

    named_text, _named_offsets = format_diarized_screenplay_with_offsets(
        [s for s in segments if isinstance(s, dict)]
    )
    if anon_text == named_text:
        # Naming resolved nothing, so the two renders are the same file.
        return None

    rel = anon_transcript_relpath(rel_transcript_path)
    out_path = os.path.join(effective_output_dir, rel)
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(anon_text)
    except OSError as exc:
        # NOT fatal. This is a debugging view; losing it must not cost the episode, whose real
        # artifacts are already on disk by the time this runs.
        logger.warning("could not write the pre-naming transcript %s: %s", rel, exc)
        return None
    return rel
