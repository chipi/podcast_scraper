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

import pytest
import yaml

from podcast_scraper.speaker_detectors.corroboration import corroborate_guests
from podcast_scraper.speaker_detectors.guests import _is_likely_actual_guest

FIXTURES = pathlib.Path("tests/fixtures/speaker_detection_traps.yaml")


def _cases() -> List[Dict[str, Any]]:
    if not FIXTURES.is_file():
        return []
    cases: List[Dict[str, Any]] = yaml.safe_load(FIXTURES.read_text())["cases"]
    return cases


CASES = _cases()
IDS = [c["id"] for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_impostors_are_rejected(case: Dict[str, Any]) -> None:
    """Nobody the episode is merely ABOUT may be accepted as a speaker.

    This is the half that matters. Every name here is a person (or a company) the episode discusses
    and who never says a word.
    """
    for impostor in case["must_not_include"]:
        assert not _is_likely_actual_guest(impostor, case["title"], case["description"]), (
            f"{case['id']}: '{impostor}' was accepted as a guest, but the episode only talks "
            "ABOUT them. Their name would be painted onto a real person's voice."
        )


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_real_guests_are_found(case: Dict[str, Any]) -> None:
    """The detector must not become so timid that it finds nobody.

    Rejecting everything would pass the impostor test and be useless. A guest introduced with a cue
    directly before their name has to be found.
    """
    for guest in case["must_include"]:
        assert _is_likely_actual_guest(
            guest, case["title"], case["description"]
        ), f"{case['id']}: real guest '{guest}' was rejected, but the episode introduces them."


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_an_llm_proposal_is_corroborated_before_it_reaches_a_voice(case: Dict[str, Any]) -> None:
    """THE PRODUCTION BUG. An LLM detector returns a name list and ``success=True``, and nothing
    checks either.

    ``llm_proposes`` in the first fixture is qwen3.5:35b's verbatim live output on the real Hard
    Fork metadata: it named Elon Musk and Sam Altman as *speakers* on an episode where one is being
    sued and the other is merely discussed. The prompt forbids exactly this. The model did it
    anyway and reported success — a prompt is not an enforcement mechanism.

    The gate re-checks the model's claim against the same text the model was shown.
    """
    kept = corroborate_guests(
        case["llm_proposes"],
        episode_title=case["title"],
        episode_description=case["description"],
        known_hosts=set(case["known_hosts"]),
    )
    kept_lower = {k.lower() for k in kept}

    for impostor in case["must_not_include"]:
        assert impostor.lower() not in kept_lower, (
            f"{case['id']}: the LLM proposed '{impostor}' and the gate let it through. "
            f"It would be painted onto a real person's diarized voice. Kept: {kept}"
        )
    for host in case["known_hosts"]:
        assert host.lower() not in kept_lower, f"{case['id']}: host '{host}' leaked into the guests"
    for guest in case["must_include"]:
        assert any(guest.lower() in k for k in kept_lower), (
            f"{case['id']}: the LLM proposed real guest '{guest}' and the gate dropped them. "
            f"Kept: {kept}"
        )


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
