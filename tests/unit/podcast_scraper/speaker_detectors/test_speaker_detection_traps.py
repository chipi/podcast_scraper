"""A wrong name on a diarized voice is worse than no name at all.

The production failure this suite exists to prevent: Hard Fork's "OpenAI's Big Reset" names
Musk in its description as the man SUING OpenAI, and names Dr. Adam Rodman as the guest who
"returns". Speaker detection returned Musk and Sam Altman as speakers, their names were painted onto
the guests' diarized voice clusters, and every insight Dr. Rodman spoke was attributed to Elon Musk.
The stage reported success.

The existing unit tests all pass, because every one of them is a toy string with the interview cue
sitting next to the name. Real feed descriptions have several people in them and only one of them
speaks. These fixtures are that.

``must_not_include`` is the load-bearing half. A detector that finds every real guest is worthless
if it also invents three.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List

import yaml

FIXTURES = pathlib.Path("tests/fixtures/speaker_detection_traps.yaml")


def _cases() -> List[Dict[str, Any]]:
    if not FIXTURES.is_file():
        return []
    cases: List[Dict[str, Any]] = yaml.safe_load(FIXTURES.read_text())["cases"]
    return cases


CASES = _cases()
IDS = [c["id"] for c in CASES]


def test_the_fixtures_actually_contain_traps() -> None:
    """Guard the guard: a fixture file with no impostors in it would pass vacuously."""
    assert CASES, "no fixtures loaded"
    impostors = sum(len(c["must_not_include"]) for c in CASES)
    guests = sum(len(c["must_include"]) for c in CASES)
    assert impostors >= 10, f"only {impostors} impostors — the suite is too easy to pass"
    assert guests >= 4, f"only {guests} real guests — a detector that rejects everything would pass"

    # Every case must hand the gate at least one name that has to die, or the corroboration test
    # above passes without ever exercising the gate.
    for case in CASES:
        proposed = {n.lower() for n in case["llm_proposes"]}
        must_die = {n.lower() for n in case["must_not_include"]} | {
            h.lower() for h in case["known_hosts"]
        }
        assert proposed & must_die, (
            f"{case['id']}: llm_proposes contains nothing that should be rejected — "
            "the corroboration gate is not actually being tested by this case"
        )
