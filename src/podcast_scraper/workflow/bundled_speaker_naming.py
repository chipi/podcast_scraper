"""Transcript naming inside the bundled calls (bundled, mega_bundled, extraction_bundled).

In the staged pipeline transcript naming is its own call in the diarization stage. In a bundled
mode the call that already reads the whole transcript carries one more field instead: the model
says who each speaker LABEL of the transcript really is (operator's option A, 2026-10-23; the
transcript is the named one, so this is a check of the ladder's labels, not the measured
voice-id prompt). This module is the metadata stage's side of it:

- :func:`labels_for_call` reads the speaker-diagnostics sidecar next to the transcript and returns
  the labels to ask about (none when transcript naming is off, the mode is staged, or the episode
  has no diarized voices, so the prompt is then byte-identical to today's);
- :func:`take_speakers` pops the ``speakers`` field out of a bundled summary's JSON, so the summary
  parser (and a degraded summary's stored raw text) never sees the model's speaker guesses;
- :func:`record` writes the answer into the sidecar under ``transcript_naming``, in the standalone
  entry's shape (``call`` says which bundled call). Shadow only: a bundled call never changes the
  published names yet (``publish`` is recorded with ``applied: false``).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)


def _diagnostics_path(output_dir: str, transcript_file_path: str) -> str:
    base, _ = os.path.splitext(os.path.join(output_dir, transcript_file_path))
    return base + ".speakers.diagnostics.json"


def _load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def labels_for_call(
    cfg: Any, pipeline_mode: str, output_dir: str, transcript_file_path: Optional[str]
) -> Optional[List[str]]:
    """The speaker labels a bundled call should judge, or None (then nothing is added)."""
    from ..providers.ml.diarization import transcript_naming as tn

    if tn.mode_of(cfg) == "off" or pipeline_mode not in tn.BUNDLED_MODES:
        return None
    if not transcript_file_path:
        return None
    diag = _load(_diagnostics_path(output_dir, transcript_file_path))
    if not diag:
        return None
    labels = sorted(tn.labels_from_diagnostics(diag))
    return labels or None


def take_speakers(result: Any) -> Optional[Dict[str, Any]]:
    """Pop ``speakers`` from a bundled summary result's JSON (``result["summary"]``), in place."""
    if not isinstance(result, dict) or not isinstance(result.get("summary"), str):
        return None
    try:
        data = json.loads(result["summary"])
    except ValueError:
        return None
    if not isinstance(data, dict) or "speakers" not in data:
        return None
    speakers = data.pop("speakers")
    result["summary"] = json.dumps(data, ensure_ascii=False)
    return speakers if isinstance(speakers, dict) else None


def record(
    cfg: Any,
    output_dir: str,
    transcript_file_path: Optional[str],
    call: str,
    speakers: Optional[Mapping[str, Any]],
) -> None:
    """Write a bundled call's answer into the sidecar. Never raises."""
    from ..languages import transcription_language
    from ..providers.ml.diarization import transcript_naming as tn

    if not transcript_file_path:
        return
    path = _diagnostics_path(output_dir, transcript_file_path)
    diag = _load(path)
    if diag is None:
        return
    try:
        mode = tn.mode_of(cfg)
        entry = tn.bundled_sidecar(
            mode,
            call,
            speakers,
            diag,
            floor=float(getattr(cfg, "speaker_name_min_talk_share", 0.0) or 0.0),
            # Operator renames come with the job, which the metadata stage does not hold; the
            # ladder side already shows them, the model side does not (a rename is rare).
            renames=None,
            language=transcription_language(cfg),
        )
        # A bundled call never changes the published names yet; say so, so the sidecar is not
        # read as a publish.
        entry["applied"] = False
        diag[tn.SIDECAR_KEY] = entry
        with open(path, "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2, allow_nan=False)
    except Exception:  # noqa: BLE001 - a shadow record must never cost the episode its summary
        logger.warning("transcript naming: could not record the %s answer", call, exc_info=True)
