"""Attribution metric (ADR-135): how much of the final naming/role came from the LLM vs the
deterministic cues, measured by diffing a pure-cue baseline roster against the shipped roster."""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization.pipeline import (
    _reconcile_non_regression,
    _resolution_attribution,
)
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


def test_attribution_tracks_a_dropped_name_as_removed() -> None:
    # The exact prod-v2.4-100ep bug (John Kim): cues named 2, the LLM path un-named the guest.
    baseline = _roster(
        {
            "S0": SpeakerRole(
                name="Patrick O'Shaughnessy", role="host", named=True, source="known_hosts"
            ),
            "S1": SpeakerRole(name="John Kim", role="guest", named=True, source="self_intro"),
        }
    )
    final = _roster(
        {
            "S0": SpeakerRole(
                name="Patrick O'Shaughnessy", role="host", named=True, source="known_hosts"
            ),
            "S1": SpeakerRole(name="S1", role="guest", named=False, source="raw"),  # name dropped
        }
    )
    att = _resolution_attribution(baseline, final)
    assert att["llm_delta"]["names_removed"] == [{"voice": "S1", "name": "John Kim"}]
    assert att["deterministic"]["named"] == 2 and att["final"]["named"] == 1


def test_reconcile_restores_a_name_the_llm_path_dropped() -> None:
    baseline = _roster(
        {
            "S0": SpeakerRole(
                name="Patrick O'Shaughnessy", role="host", named=True, source="known_hosts"
            ),
            "S1": SpeakerRole(name="John Kim", role="guest", named=True, source="self_intro"),
        }
    )
    final = _roster(
        {
            "S0": SpeakerRole(
                name="Patrick O'Shaughnessy", role="host", named=True, source="known_hosts"
            ),
            "S1": SpeakerRole(name="S1", role="guest", named=False, source="raw"),
        }
    )
    fixed, restored = _reconcile_non_regression(baseline, final)
    assert restored == ["S1"]
    assert fixed.by_voice["S1"].named is True
    assert fixed.by_voice["S1"].name == "John Kim"
    # post-reconciliation the shipped roster no longer regresses vs the baseline
    att = _resolution_attribution(baseline, fixed)
    assert att["llm_delta"]["names_removed"] == []
    assert att["final"]["named"] == 2


def test_reconcile_is_a_noop_when_llm_only_adds() -> None:
    # LLM correctly ADDS a name + demotes an over-seated host — nothing to restore.
    baseline = _roster(
        {
            "S0": SpeakerRole(name="Elad Gil", role="host", named=True, source="known_hosts"),
            "S1": SpeakerRole(name="S1", role="host", named=False, source="raw"),
        }
    )
    final = _roster(
        {
            "S0": SpeakerRole(name="Elad Gil", role="host", named=True, source="known_hosts"),
            "S1": SpeakerRole(
                name="Stanley Tang", role="guest", named=True, source="llm_resolution"
            ),
        }
    )
    fixed, restored = _reconcile_non_regression(baseline, final)
    assert restored == []
    assert fixed is final  # unchanged object when there is nothing to restore
    assert fixed.by_voice["S1"].name == "Stanley Tang"


def test_reconcile_keeps_llm_role_correction_when_name_was_never_in_baseline() -> None:
    # A voice the baseline never NAMED (only raw) that the LLM leaves unnamed is NOT restored —
    # the guard protects names the cues established, it does not force-name anonymous voices.
    baseline = _roster({"S0": SpeakerRole(name="S0", role="unknown", named=False, source="raw")})
    final = _roster({"S0": SpeakerRole(name="S0", role="guest", named=False, source="raw")})
    fixed, restored = _reconcile_non_regression(baseline, final)
    assert restored == []
    assert fixed.by_voice["S0"].role == "guest"  # LLM role change preserved
