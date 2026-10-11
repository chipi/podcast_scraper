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
- ``stated_voice_names``: optional ``{voice: name}``, a publisher's own speaker labels;
- ``language``: optional transcript language (default English);
- ``turns``: ``[voice, seconds, text]`` in order. Durations stand in for the diarizer's times;
- ``expect``: per voice ``{"name": str|null, "role": "host"|"guest"|..., "name_not": [str]}``; a
  voice left out is not judged. ``name: null`` means "must stay unnamed" (a wrong name is worse
  than none); ``name_not`` lists names the voice must never carry, when the right answer is open;
  ``source`` pins where the name came from; ``steps`` lists trace steps that must have happened on
  that voice, each ``"rung"``, ``"rung:decision"`` or ``"rung:decision:value"`` (value = any field
  of the step, e.g. ``host_seat_step:seated:3_opener``). ``episode_steps`` does the same for the
  episode-level notes. This is how a scenario says WHICH rung must decide, not only the outcome.

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
from podcast_scraper.utils import correlation

SCENARIOS = Path(__file__).resolve().parent / "scenarios"
#: Key of the episode-level record in :func:`run`'s result (not a voice).
EPISODE = "__episode__"


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
    # Production's host detection records the feed title in the run context, and the roster reads it
    # from there (`pipeline._current_feed_title`): the show-name filters depend on it.
    previous_title = correlation.get_feed_title()
    correlation.set_feed_title(feed.get("title"))
    try:
        result: Dict[str, Dict[str, Any]] = _run(
            scenario, segments, diarization, feed_hosts, detected, episode
        )
        return result
    finally:
        _pipeline._resolve_voices_via_llm = real_llm
        correlation.set_feed_title(previous_title)


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
            **({"language": scenario["language"]} if scenario.get("language") else {}),
        )
        out = apply_diarization_to_result(
            {"segments": segments, "text": " ".join(s["text"] for s in segments)},
            audio_path="/nonexistent/scenario.mp3",
            cfg=cfg,
            detected_speaker_names=detected.get("speaker_names"),
            metadata_named=detected.get("metadata_named"),
            precomputed_diarization=diarization,
            stated_voice_names=scenario.get("stated_voice_names") or None,
            feed_hosts=feed_hosts or None,
            episode_title=episode.get("title"),
            episode_description=episode.get("description"),
            detection_ran=True,
        )
    diag = out.get("speaker_diagnostics") or {}
    trace = diag.get("decision_trace") or {}
    got: Dict[str, Dict[str, Any]] = {
        str(v.get("voice")): {
            "name": v.get("resolved_name") if v.get("named") else None,
            "role": v.get("role"),
            "named": v.get("named"),
            "source": v.get("source"),
            "voice_type": v.get("voice_type"),
            "steps": (trace.get("voices") or {}).get(str(v.get("voice"))) or [],
        }
        for v in diag.get("voices") or []
        if isinstance(v, dict)
    }
    got[EPISODE] = {"steps": list(trace.get("episode") or []), "trace": trace}
    return got


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
        if want.get("source") and want["source"] != have.get("source"):
            out.append(f"{voice}: source {have.get('source')!r}, expected {want['source']!r}")
        for step in want.get("steps") or []:
            if not _has_step(have.get("steps") or [], step):
                out.append(f"{voice}: no trace step {step!r}")
    for step in scenario.get("episode_steps") or []:
        if not _has_step((got.get(EPISODE) or {}).get("steps") or [], step):
            out.append(f"episode: no trace step {step!r}")
    return out


def _has_step(steps: List[Dict[str, Any]], spec: str) -> bool:
    rung, _, rest = spec.partition(":")
    decision, _, value = rest.partition(":")
    for st in steps:
        if st.get("rung") != rung:
            continue
        if decision and st.get("decision") != decision:
            continue
        if value and value not in {str(x) for x in st.values()}:
            continue
        return True
    return False


def rungs_seen(got: Dict[str, Dict[str, Any]]) -> set:
    """Every trace rung key that occurred anywhere in one scenario's run."""
    seen = set()
    for voice, rec in got.items():
        for st in rec.get("steps") or []:
            seen.add(st.get("rung"))
    trace = (got.get(EPISODE) or {}).get("trace") or {}
    for st in (trace.get("names") or {}).values() if isinstance(trace.get("names"), dict) else []:
        for x in st if isinstance(st, list) else []:
            seen.add(x.get("rung"))
    seen |= {f"inputs.{k}" for k in (trace.get("inputs") or {})}
    return {r for r in seen if r}


def describe(got: Dict[str, Dict[str, Any]]) -> str:
    view = {
        v: {k: r[k] for k in ("name", "role", "source", "voice_type") if k in r}
        for v, r in got.items()
        if v != EPISODE
    }
    return json.dumps(view, indent=1, sort_keys=True, default=str)


__all__ = ["describe", "load_scenarios", "mismatches", "run", "Optional"]
