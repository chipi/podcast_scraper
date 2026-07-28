"""Attribution metric (ADR-135): how much of the final naming/role came from the LLM vs the
deterministic cues, measured by diffing a pure-cue baseline roster against the shipped roster."""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization.pipeline import _resolution_attribution
from podcast_scraper.providers.ml.diarization.roster import SpeakerRole, SpeakerRoster

pytestmark = pytest.mark.unit


def _roster(voices: dict) -> SpeakerRoster:
    return SpeakerRoster(by_voice=voices, num_speakers=len(voices))


def test_llm_added_name_and_changed_role_are_attributed() -> None:
    # baseline (cues only): S0 host, S1 the guest is unnamed and mis-seated as a second host.
    baseline = _roster(
        {
            "S0": SpeakerRole(name="Elad Gil", role="host", named=True, source="known_hosts"),
            "S1": SpeakerRole(name="S1", role="host", named=False, source="raw"),
        }
    )
    # final (LLM): S1 named Stanley Tang and demoted to guest.
    final = _roster(
        {
            "S0": SpeakerRole(name="Elad Gil", role="host", named=True, source="known_hosts"),
            "S1": SpeakerRole(
                name="Stanley Tang", role="guest", named=True, source="llm_resolution"
            ),
        }
    )
    att = _resolution_attribution(baseline, final)
    assert att["deterministic"] == {"named": 1, "hosts": 2, "guests": 0}
    assert att["final"] == {"named": 2, "hosts": 1, "guests": 1}
    assert att["llm_delta"]["names_added"] == [{"voice": "S1", "name": "Stanley Tang"}]
    assert att["llm_delta"]["roles_changed"] == [{"voice": "S1", "from": "host", "to": "guest"}]


def test_no_llm_change_is_an_empty_delta() -> None:
    same = _roster(
        {"S0": SpeakerRole(name="Casey Newton", role="host", named=True, source="self_intro")}
    )
    att = _resolution_attribution(same, same)
    assert att["llm_delta"]["names_added"] == []
    assert att["llm_delta"]["roles_changed"] == []
    assert att["deterministic"] == att["final"]
