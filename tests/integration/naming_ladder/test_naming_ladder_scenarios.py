"""Every naming scenario must come out CORRECT, not merely unchanged (#2304, segment L).

See ``harness.py`` for the scenario format. A scenario the current code gets wrong carries
``"xfail": "<why>"``; it runs strict, so the day a fix makes it pass the marker must be removed.
"""

from __future__ import annotations

import pytest

from . import harness

pytestmark = pytest.mark.integration

_SCENARIOS = harness.load_scenarios()


def test_there_are_scenarios() -> None:
    assert len(_SCENARIOS) >= 10


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(
            sc,
            id=sc["id"],
            marks=([pytest.mark.xfail(reason=sc["xfail"], strict=True)] if sc.get("xfail") else []),
        )
        for sc in _SCENARIOS
    ],
)
def test_scenario(scenario) -> None:
    got = harness.run(scenario)
    wrong = harness.mismatches(scenario, got)
    assert not wrong, f"{scenario['id']}: {wrong}\nroster: {harness.describe(got)}"
