"""A placeholder is not a person, so it must never be a resolution TARGET (#1685).

`resolve_candidates` accepts an id whose token set is a superset of the bare token.
`person:unresolved-dario-ep-42` tokenises to ``{unresolved, dario, ep, 42}`` — a superset of
`dario` — so before the exclusion it qualified as a "full name" and a placeholder became
something a real name could resolve INTO.

`is_bare_person_id` already excluded placeholders from being SCOPED, which made the pass
idempotent and made the gap easy to miss: the two questions ("should this be scoped?" and "may
this be a target?") look like one question and are not.

Both failure shapes were observed in production data. The 2026-08-26 prod audit printed
``resolvable dario -> unresolved-dario-substack-post-204178365`` — a bare name "resolving" to a
person the pipeline had already failed to identify.
"""

from __future__ import annotations

import pytest

from podcast_scraper.identity.bare_name_scope import (
    is_bare_person_id,
    is_scoped_person_id,
    plan_bare_name_ids,
    resolve_candidates,
)

pytestmark = [pytest.mark.unit]


def _plan(ids, ep, *, heal=True):
    """`plan_bare_name_ids` with the candidate pool equal to the roster.

    These tests hand an explicit list of ids rather than an artifact, so there is no
    node-backed/dangling distinction to draw — the roster IS the node-backed set. That
    distinction, and the reason `candidate_ids` is required rather than defaulted, is covered in
    `test_roster_matches_rewriter.py` (#1868).
    """
    return plan_bare_name_ids(ids, ep, heal=heal, candidate_ids=ids)


EP = "ep-42"
OTHER = "some-other-episode"


class TestAPlaceholderIsNeverACandidate:
    def test_it_is_excluded_even_though_its_tokens_match(self) -> None:
        assert (
            resolve_candidates("person:dario", ["person:dario", f"person:unresolved-dario-{EP}"])
            == []
        )

    def test_the_two_questions_are_distinct(self) -> None:
        """`is_bare_person_id` says "scope me"; `is_scoped_person_id` says "I am already one"."""
        assert is_bare_person_id("person:dario")
        assert not is_scoped_person_id("person:dario")
        assert not is_bare_person_id(f"person:unresolved-dario-{EP}")
        assert is_scoped_person_id(f"person:unresolved-dario-{EP}")
        # A real full name is neither — they are not complements.
        assert not is_bare_person_id("person:dario-amodei")
        assert not is_scoped_person_id("person:dario-amodei")


class TestAPlaceholderMustNotBlockACorrectHeal:
    """The costlier of the two: a real person is available and gets discarded."""

    def test_the_real_person_wins_when_a_placeholder_is_also_present(self) -> None:
        roster = ["person:dario", "person:dario-amodei", f"person:unresolved-dario-{EP}"]
        assert resolve_candidates("person:dario", roster) == ["person:dario-amodei"]
        assert _plan(roster, EP, heal=True) == {"person:dario": "person:dario-amodei"}

    def test_without_the_placeholder_the_answer_is_the_same(self) -> None:
        """The presence of a placeholder must not change the verdict at all."""
        clean = ["person:dario", "person:dario-amodei"]
        with_ph = clean + [f"person:unresolved-dario-{EP}"]
        assert _plan(clean, EP, heal=True) == _plan(with_ph, EP, heal=True)

    def test_genuine_ambiguity_still_refuses(self) -> None:
        """Excluding placeholders must not make the rule reckless — two REAL names still refuse."""
        roster = ["person:trump", "person:donald-trump", "person:eric-trump"]
        assert len(resolve_candidates("person:trump", roster)) == 2
        assert _plan(roster, EP, heal=True) == {"person:trump": f"person:unresolved-trump-{EP}"}


class TestNoCrossEpisodeContamination:
    """Importing another episode's scope is the exact harm episode-scoping exists to prevent."""

    def test_another_episodes_placeholder_is_not_a_target(self) -> None:
        roster = ["person:dario", f"person:unresolved-dario-{OTHER}"]
        assert resolve_candidates("person:dario", roster) == []
        assert _plan(roster, EP, heal=True) == {"person:dario": f"person:unresolved-dario-{EP}"}

    def test_the_result_carries_THIS_episode(self) -> None:
        roster = ["person:dario", f"person:unresolved-dario-{OTHER}"]
        new_id = _plan(roster, EP, heal=True)["person:dario"]
        assert new_id.endswith(EP)
        assert OTHER not in new_id


class TestHealFalseWasAlreadySafe:
    """Why this was not a hard blocker on the backfill: heal=False never consults candidates.

    Recorded so the sequencing decision keeps its evidence — the migration could always have run
    safely at heal=False, and the fix was scheduled first for cost, not safety.
    """

    @pytest.mark.parametrize(
        "roster",
        [
            ["person:dario", f"person:unresolved-dario-{EP}"],
            ["person:dario", "person:dario-amodei", f"person:unresolved-dario-{EP}"],
            ["person:dario", f"person:unresolved-dario-{OTHER}"],
            ["person:dario", "person:dario-amodei"],
        ],
    )
    def test_every_roster_scopes_to_this_episode(self, roster) -> None:
        assert _plan(roster, EP, heal=False) == {"person:dario": f"person:unresolved-dario-{EP}"}
