"""E1d and S30s, the two rescues after the ladder (``ladder_rescues``): when each fires, and not.

The rules are ports of the measured ones; every case here is a condition of that rule. All
fixtures are synthetic (never-commit-real-episodes).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from podcast_scraper.providers.ml.diarization import ladder_rescues as lr
from podcast_scraper.providers.ml.diarization.naming_trace import NamingTrace
from podcast_scraper.providers.ml.diarization.roster import SpeakerRole, SpeakerRoster

pytestmark = pytest.mark.unit

LONG = " ".join(["indeed"] * 160)


def _roster(**voices: Tuple[Optional[str], str]) -> SpeakerRoster:
    return SpeakerRoster(
        by_voice={
            v: SpeakerRole(
                name=n or v, role=role, named=bool(n), source="llm_resolution" if n else "x"
            )
            for v, (n, role) in voices.items()
        },
        num_speakers=len(voices),
    )


def _refused(voice: str, name: str, role: str = "guest") -> Dict[str, Any]:
    return {"voice": voice, "name": name, "outcome": "third_person", "role": role}


# --- E1d -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "turns, cue",
    [
        ([("SPEAKER_01", "Hi, I'm Maria Lindqvist and I farm.")], "self_intro"),
        ([("SPEAKER_01", "Hello, I'm Dr. Lindqvist from the farm.")], "self_intro"),
        ([("SPEAKER_01", "My name is Maria and I farm.")], "self_intro"),
        ([("SPEAKER_01", "On the socials I'm at Maria Lindqvist.")], "self_intro"),  # a handle
        (
            [("SPEAKER_00", "Welcome Maria Lindqvist to the show."), ("SPEAKER_01", "Hi.")],
            "introduced",
        ),
        ([("SPEAKER_00", "Thanks, Maria, for coming."), ("SPEAKER_01", "Glad to.")], "introduced"),
    ],
)
def test_e1d_puts_an_introduced_refused_name_back(turns: List[Tuple[str, str]], cue: str) -> None:
    roster = _roster(SPEAKER_00=("Tobias Wren", "host"), SPEAKER_01=(None, "guest"))
    trace = NamingTrace()
    out = lr.rescue_introduced(roster, [_refused("SPEAKER_01", "Maria Lindqvist")], turns, trace)
    v = out.by_voice["SPEAKER_01"]
    assert (v.name, v.role, v.named, v.source) == ("Maria Lindqvist", "guest", True, lr.E1D_SOURCE)
    assert trace.voices["SPEAKER_01"][0]["cue"] == cue


@pytest.mark.parametrize(
    "why, roster, turns",
    [
        (
            # E1c's "it's X" matched a voice exclaiming another person's name: E1d drops it.
            "it's X is not an introduction",
            _roster(SPEAKER_00=("Tobias Wren", "host"), SPEAKER_01=(None, "guest")),
            [("SPEAKER_01", "Oh, it's Maria Lindqvist again!")],
        ),
        (
            "no cue at all",
            _roster(SPEAKER_00=("Tobias Wren", "host"), SPEAKER_01=(None, "guest")),
            [("SPEAKER_01", "Maria Lindqvist wrote a book.")],
        ),
        (
            "another voice already holds the name",
            _roster(SPEAKER_00=("Maria Lindqvist", "host"), SPEAKER_01=(None, "guest")),
            [("SPEAKER_01", "I'm Maria Lindqvist.")],
        ),
        (
            "the voice is already named",
            _roster(SPEAKER_00=("Tobias Wren", "host"), SPEAKER_01=("Ann Lee", "guest")),
            [("SPEAKER_01", "I'm Maria Lindqvist.")],
        ),
        (
            # v52: step 3, x046 — the host said "so I'm hoping that <guest>'s going to ...".
            "words between 'I'm' and the name are not an introduction",
            _roster(SPEAKER_00=("Tobias Wren", "guest"), SPEAKER_01=(None, "guest")),
            [("SPEAKER_01", "So I'm hoping that Maria's going to take on the hard part.")],
        ),
        (
            "a guest's name never goes onto the voice the ladder seated as host",
            _roster(SPEAKER_00=("Tobias Wren", "guest"), SPEAKER_01=(None, "host")),
            [("SPEAKER_01", "I'm Maria Lindqvist.")],
        ),
    ],
)
def test_e1d_does_not_fire(why: str, roster: SpeakerRoster, turns: List[Tuple[str, str]]) -> None:
    out = lr.rescue_introduced(roster, [_refused("SPEAKER_01", "Maria Lindqvist")], turns)
    assert out.by_voice["SPEAKER_01"] == roster.by_voice["SPEAKER_01"], why


def test_e1d_ignores_verdicts_that_were_not_third_person() -> None:
    roster = _roster(SPEAKER_00=("Tobias Wren", "host"), SPEAKER_01=(None, "guest"))
    verdict = {"voice": "SPEAKER_01", "name": "Maria Lindqvist", "outcome": "refused_spelling"}
    out = lr.rescue_introduced(roster, [verdict], [("SPEAKER_01", "I'm Maria Lindqvist.")])
    assert out is roster


# --- S30s ----------------------------------------------------------------------------------


def _split(share_words: int = 160) -> List[Tuple[str, str]]:
    return [("SPEAKER_00", LONG), ("SPEAKER_01", " ".join(["indeed"] * share_words))]


def _host_verdict() -> List[Dict[str, Any]]:
    return [{"voice": "SPEAKER_00", "name": "Tobias Wren", "outcome": "accepted"}]


def test_s30s_names_the_unnamed_half_of_a_split_host() -> None:
    roster = _roster(SPEAKER_00=("Tobias Wren", "host"), SPEAKER_01=(None, "unknown"))
    trace = NamingTrace()
    out = lr.merge_split_host(roster, _split(), _host_verdict(), ["Tobias Wren"], trace)
    v = out.by_voice["SPEAKER_01"]
    assert (v.name, v.role, v.source) == ("Tobias Wren", "host", lr.S30_SOURCE)
    assert trace.voices["SPEAKER_01"][0]["rung"] == lr.S30_SOURCE


@pytest.mark.parametrize(
    "why, roster, turns, verdicts, stated",
    [
        (
            # held-out x130: a stated guest nobody introduces; the strict rule refuses.
            "a stated name other than the host",
            _roster(SPEAKER_00=("Tobias Wren", "host"), SPEAKER_01=(None, "unknown")),
            _split(),
            _host_verdict(),
            ["Tobias Wren", "Maria Lindqvist"],
        ),
        (
            "the LLM was not asked",
            _roster(SPEAKER_00=("Tobias Wren", "host"), SPEAKER_01=(None, "unknown")),
            _split(),
            [],
            ["Tobias Wren"],
        ),
        (
            "the LLM named someone else",
            _roster(SPEAKER_00=("Tobias Wren", "host"), SPEAKER_01=(None, "unknown")),
            _split(),
            _host_verdict() + [{"voice": "SPEAKER_01", "name": "Ann Lee", "outcome": "refused"}],
            ["Tobias Wren"],
        ),
        (
            "the unnamed voice holds under 30% of the words",
            _roster(SPEAKER_00=("Tobias Wren", "host"), SPEAKER_01=(None, "unknown")),
            [("SPEAKER_00", " ".join(["indeed"] * 400)), ("SPEAKER_01", LONG)],
            _host_verdict(),
            ["Tobias Wren"],
        ),
        (
            "a voice under 150 words is not speaking",
            _roster(SPEAKER_00=("Tobias Wren", "host"), SPEAKER_01=(None, "unknown")),
            _split(share_words=100),
            _host_verdict(),
            ["Tobias Wren"],
        ),
        (
            "the named voice is the guest, not the host",
            _roster(SPEAKER_00=("Tobias Wren", "guest"), SPEAKER_01=(None, "unknown")),
            _split(),
            _host_verdict(),
            ["Tobias Wren"],
        ),
        (
            "three speaking voices",
            _roster(
                SPEAKER_00=("Tobias Wren", "host"),
                SPEAKER_01=(None, "unknown"),
                SPEAKER_02=(None, "unknown"),
            ),
            _split() + [("SPEAKER_02", LONG)],
            _host_verdict(),
            ["Tobias Wren"],
        ),
    ],
)
def test_s30s_does_not_fire(
    why: str,
    roster: SpeakerRoster,
    turns: List[Tuple[str, str]],
    verdicts: List[Dict[str, Any]],
    stated: List[str],
) -> None:
    assert lr.merge_split_host(roster, turns, verdicts, stated) is roster, why


def test_apply_runs_s30s_before_e1d_as_measured() -> None:
    # The unnamed voice is both a split-host candidate and an E1d candidate: S30s, run first,
    # names it the host, so E1d (which skips named voices) does not.
    roster = _roster(SPEAKER_00=("Tobias Wren", "host"), SPEAKER_01=(None, "unknown"))
    turns = [("SPEAKER_00", LONG), ("SPEAKER_01", "I'm Tobias, " + LONG)]
    report = {"verdicts": _host_verdict() + [_refused("SPEAKER_01", "Tobias Wren", "host")]}
    out = lr.apply(roster, report, turns, ["Tobias Wren"])
    assert out.by_voice["SPEAKER_01"].source == lr.S30_SOURCE


def test_off_by_default() -> None:
    from podcast_scraper import config

    assert config.Config.model_fields["ladder_rescue_rules"].default is False


# --- through the diarization stage ----------------------------------------------------------


def _stage_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, on: bool) -> List[Any]:
    from podcast_scraper.config import Config
    from podcast_scraper.providers.ml.diarization import pipeline as P
    from podcast_scraper.providers.ml.diarization.base import (
        DiarizationResult,
        DiarizationSegment,
    )

    def fake_llm(*a: Any, report: Optional[Dict[str, Any]] = None, **k: Any) -> Any:
        if report is not None:
            report["verdicts"] = [_refused("SPEAKER_01", "Maria Lindqvist")]
        return {}, {}

    calls: List[Any] = []

    def spy(
        roster: Any,
        llm_report: Any,
        turns: Any,
        stated: Any,
        trace: Any = None,
        embeddings: Any = None,
    ) -> Any:
        calls.append((llm_report, turns, stated))
        assert embeddings == {}  # the diarization here has no fingerprints
        return roster

    monkeypatch.setattr(P, "_resolve_voices_via_llm", fake_llm)
    monkeypatch.setattr(lr, "apply", spy)
    P._copresence_cache.clear()
    P.apply_diarization_to_result(
        {"text": "x", "segments": [{"start": 0, "end": 60, "text": "Thanks, Maria, for coming."}]},
        "",
        Config(output_dir=str(tmp_path), speaker_resolution_llm=False, ladder_rescue_rules=on),
        ["Maria Lindqvist"],
        precomputed_diarization=DiarizationResult(
            segments=[DiarizationSegment(0, 60, "SPEAKER_00")], num_speakers=1
        ),
        episode_title="Ep 1",
    )
    return calls


def test_the_stage_runs_the_rescues_only_when_the_flag_is_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    assert _stage_calls(monkeypatch, tmp_path, False) == []
    ((report, turns, stated),) = _stage_calls(monkeypatch, tmp_path, True)
    # What the measured arm saw: the resolver's verdicts, the ordered turns, the stated names.
    assert report["verdicts"][0]["outcome"] == "third_person"
    assert turns == [("SPEAKER_00", "Thanks, Maria, for coming.")]
    assert "Maria Lindqvist" in stated


def _fp(similar: bool) -> Dict[str, List[float]]:
    host = [1.0, 0.0, 0.0, 0.2]
    other = [0.98, 0.05, 0.0, 0.2] if similar else [0.1, 1.0, 0.3, 0.0]
    return {"SPEAKER_00": host, "SPEAKER_01": other}


def test_s30s_refuses_when_the_fingerprints_are_two_people() -> None:
    """A two-person episode with a guest whose name is never stated looks like a split host to the
    text rule; the fingerprints do not (DGX block, 2026-10-23: host and guest at cosine 0.17)."""
    roster = _roster(SPEAKER_00=("Tobias Wren", "host"), SPEAKER_01=(None, "unknown"))
    trace = NamingTrace()
    out = lr.merge_split_host(
        roster, _split(), _host_verdict(), ["Tobias Wren"], trace, embeddings=_fp(similar=False)
    )
    assert out is roster
    (step,) = trace.voices["SPEAKER_01"]
    assert (step["rung"], step["decision"]) == (lr.S30_SOURCE, "refused_fingerprints")
    assert step["similarity"] < lr.SPLIT_FINGERPRINT_MIN


def test_s30s_fires_when_the_fingerprints_are_one_person() -> None:
    roster = _roster(SPEAKER_00=("Tobias Wren", "host"), SPEAKER_01=(None, "unknown"))
    out = lr.merge_split_host(
        roster, _split(), _host_verdict(), ["Tobias Wren"], embeddings=_fp(similar=True)
    )
    assert out.by_voice["SPEAKER_01"].name == "Tobias Wren"


@pytest.mark.parametrize("embeddings", [None, {}, {"SPEAKER_00": [1.0, 0.0]}])
def test_s30s_keeps_the_text_rule_without_both_fingerprints(embeddings) -> None:
    roster = _roster(SPEAKER_00=("Tobias Wren", "host"), SPEAKER_01=(None, "unknown"))
    out = lr.merge_split_host(
        roster, _split(), _host_verdict(), ["Tobias Wren"], embeddings=embeddings
    )
    assert out.by_voice["SPEAKER_01"].name == "Tobias Wren"


def test_apply_passes_the_fingerprints_to_s30s() -> None:
    roster = _roster(SPEAKER_00=("Tobias Wren", "host"), SPEAKER_01=(None, "unknown"))
    report = {"verdicts": _host_verdict()}
    out = lr.apply(roster, report, _split(), ["Tobias Wren"], embeddings=_fp(similar=False))
    assert not out.by_voice["SPEAKER_01"].named
