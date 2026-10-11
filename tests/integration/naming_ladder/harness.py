"""The naming ladder as ONE input/output component (#2304, segment L).

A scenario states what the naming stage receives and what a CORRECT answer is:

- ``feed``: ``title``, ``description``, ``authors`` (the feed's own host statement is detected
  from these exactly as production does) and optional ``hosts`` (stated hosts, e.g. config);
- ``episode``: ``title``, ``description``;
- ``detected``: the names the speaker-detection stage handed over (``detected_speaker_names``), and
  ``metadata_named`` (every name the metadata stated);
- ``llm``: optional ``{voice: [name, role]}``, the verdicts the LLM resolution step returned. They
  are injected in place of the call (``_resolve_voices_via_llm``), so a scenario can pin how the
  ladder treats a model answer, right or wrong, without a model;
- ``turns``: ``[voice, seconds, text]`` in order. Durations stand in for the diarizer's times;
- ``expect``: per voice ``{"name": str|null, "role": "host"|"guest"|..., "name_not": [str]}``; a
  voice left out is not judged. ``name: null`` means "must stay unnamed" (a wrong name is worse
  than none); ``name_not`` lists names the voice must never carry, when the right answer is open.

It runs the real ``apply_diarization_to_result`` on the precomputed diarization, deterministic
(``speaker_resolution_llm=False``: the cue path four shipped profiles use), and reads the decision
off the speaker diagnostics: the same record the corpus stores.

Scenario texts are SYNTHETIC. They reproduce the shape of a real case (who greets whom, who says
which name, talk shares), never a real transcript.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from podcast_scraper import config
from podcast_scraper.providers.ml.diarization import pipeline as _pipeline
from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.pipeline import apply_diarization_to_result
from podcast_scraper.speaker_detectors.hosts import detect_hosts_from_feed

SCENARIOS = Path(__file__).resolve().parent / "scenarios"


def load_scenarios() -> List[Dict[str, Any]]:
    out = []
    for path in sorted(SCENARIOS.glob("*.json")):
        sc = json.loads(path.read_text(encoding="utf-8"))
        sc.setdefault("id", path.stem)
        out.append(sc)
    return out


def run(scenario: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """``{voice: {"name", "role", "named", "source", "voice_type"}}`` for one scenario."""
    feed = scenario.get("feed") or {}
    episode = scenario.get("episode") or {}
    detected = scenario.get("detected") or {}
    segments: List[Dict[str, Any]] = []
    turns: List[DiarizationSegment] = []
    t = 0.0
    for i, (voice, seconds, text) in enumerate(scenario["turns"]):
        start, end = t, t + float(seconds)
        segments.append({"id": i, "start": start, "end": end, "text": text})
        turns.append(DiarizationSegment(start=start, end=end, speaker=voice))
        t = end
    diarization = DiarizationResult(
        segments=turns, num_speakers=len({s.speaker for s in turns}), model_name="scenario"
    )
    feed_hosts = sorted(
        set(feed.get("hosts") or [])
        | detect_hosts_from_feed(
            feed.get("title"), feed.get("description"), feed.get("authors") or []
        )
    )
    llm = scenario.get("llm") or {}
    real_llm = _pipeline._resolve_voices_via_llm
    if llm:
        _pipeline._resolve_voices_via_llm = lambda *a, **k: (  # type: ignore[assignment]
            {v: nr[0] for v, nr in llm.items() if nr[0]},
            {v: nr[1] for v, nr in llm.items() if len(nr) > 1 and nr[1]},
        )
    try:
        result: Dict[str, Dict[str, Any]] = _run(
            scenario, segments, diarization, feed_hosts, detected, episode
        )
        return result
    finally:
        _pipeline._resolve_voices_via_llm = real_llm


def _run(
    scenario: Dict[str, Any],
    segments: List[Dict[str, Any]],
    diarization: DiarizationResult,
    feed_hosts: List[str],
    detected: Dict[str, Any],
    episode: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = config.Config(
            rss="https://example.com/f.xml",
            output_dir=tmp,
            speaker_resolution_llm=False,
        )
        out = apply_diarization_to_result(
            {"segments": segments, "text": " ".join(s["text"] for s in segments)},
            audio_path="/nonexistent/scenario.mp3",
            cfg=cfg,
            detected_speaker_names=detected.get("speaker_names"),
            metadata_named=detected.get("metadata_named"),
            precomputed_diarization=diarization,
            feed_hosts=feed_hosts or None,
            episode_title=episode.get("title"),
            episode_description=episode.get("description"),
            detection_ran=True,
        )
    voices = (out.get("speaker_diagnostics") or {}).get("voices") or []
    return {
        str(v.get("voice")): {
            "name": v.get("resolved_name") if v.get("named") else None,
            "role": v.get("role"),
            "named": v.get("named"),
            "source": v.get("source"),
            "voice_type": v.get("voice_type"),
        }
        for v in voices
        if isinstance(v, dict)
    }


def mismatches(scenario: Dict[str, Any], got: Dict[str, Dict[str, Any]]) -> List[str]:
    """Human-readable differences between the expected and the actual roster."""
    out: List[str] = []
    for voice, want in (scenario.get("expect") or {}).items():
        have = got.get(voice) or {}
        if "name" in want and want["name"] != have.get("name"):
            out.append(f"{voice}: name {have.get('name')!r}, expected {want['name']!r}")
        for banned in want.get("name_not") or []:
            if (have.get("name") or "").lower() == banned.lower():
                out.append(f"{voice}: named {banned!r}, which it must never be")
        if want.get("role") and want["role"] != have.get("role"):
            out.append(f"{voice}: role {have.get('role')!r}, expected {want['role']!r}")
    return out


def describe(got: Dict[str, Dict[str, Any]]) -> str:
    return json.dumps(got, indent=1, sort_keys=True, default=str)


__all__ = ["describe", "load_scenarios", "mismatches", "run", "Optional"]
