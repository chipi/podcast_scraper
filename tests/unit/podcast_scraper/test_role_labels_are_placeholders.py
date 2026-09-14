"""A role word is a position in a conversation, not a human being (#2059).

`person:host` on prod 2026-09-13: **54 episodes, 1,437 grounded insights, ranked #2 in "top
voices"**. It is not a person. Every episode whose host the pipeline failed to resolve emitted the
name "Host", slugification turned that into `person:host`, and they all merged into ONE global
node — a phantom human assembled from dozens of unrelated shows, whose top topics
(meaning-of-life, happiness science, AI) are simply several different people's material fused.

The repo had already solved this exact bug for `SPEAKER_03` (#1b, migration m0007): diarization
numbers are per-episode and unstable, so such a label must be episode-scoped, never global. Role
words are the same failure and were missed for a year because the guard was
`^speaker[\\s_-]*\\d+$` — it required DIGITS.

These tests exist because nothing in the pyramid asserted the property itself: "a placeholder must
never become a followable global person". Cases were tested; the rule was not — and one test in
`gi/test_pipeline.py` went further and asserted `speaker_id == "person:guest"`, pinning the defect
in place. That gap is #2058.
"""

from __future__ import annotations

import pytest

from podcast_scraper.graph_id_utils import entity_node_id, is_bare_speaker_label

pytestmark = pytest.mark.unit

_EP = "ep-2026-09-13-abc"
_OTHER_EP = "ep-2026-01-01-xyz"

ROLE_LABELS = [
    "Host",
    "host",
    "HOST",
    "Co-Host",
    "Guest",
    "Speaker",
    "Interviewer",
    "Moderator",
    "Panelist",
    "The Host",
    "Unknown Speaker",
    "Unidentified",
]
REAL_NAMES = [
    "Aaron Levie",
    "Theo Jaffee",
    "Russ Roberts",
    "Kaiser Kuo",
    "Hosteen Klah",
    "Guestavo Petro",
    "Speakman",
    "Anna Hostetler",
]


class TestTheRuleNobodyAsserted:
    """The property, not the cases: a placeholder must never become a global person."""

    @pytest.mark.parametrize("label", ROLE_LABELS)
    def test_a_role_word_is_a_placeholder(self, label: str) -> None:
        assert is_bare_speaker_label(label)

    @pytest.mark.parametrize("label", ROLE_LABELS)
    def test_a_role_word_never_gets_a_global_person_id(self, label: str) -> None:
        # `person:host` is precisely the id that must stop existing.
        node_id = entity_node_id("person", label, episode_id=_EP)
        assert node_id != f"person:{label.strip().lower().replace(' ', '-')}"
        assert _EP in node_id, "a placeholder must be episode-scoped"

    @pytest.mark.parametrize("label", ROLE_LABELS)
    def test_the_same_role_in_two_episodes_is_two_people(self, label: str) -> None:
        # This is the whole bug: 54 episodes' hosts were one node.
        assert entity_node_id("person", label, episode_id=_EP) != entity_node_id(
            "person", label, episode_id=_OTHER_EP
        )


class TestItDoesNotCreateAWithinEpisodePhantom:
    """Fixing a cross-episode merge by causing a within-episode one would be no fix at all."""

    def test_host_and_guest_of_one_episode_are_different_people(self) -> None:
        # Caught during implementation: the discriminator was `re.sub(r"\\D","",label) or "0"`,
        # so every role word stripped to no digits, fell back to "0", and collided.
        assert entity_node_id("person", "Host", episode_id=_EP) != entity_node_id(
            "person", "Guest", episode_id=_EP
        )

    def test_every_role_in_one_episode_is_distinct(self) -> None:
        ids = {
            entity_node_id("person", r, episode_id=_EP)
            for r in ["Host", "Guest", "Interviewer", "Moderator", "Panelist"]
        }
        assert len(ids) == 5

    def test_numbered_labels_still_work_and_do_not_collide_with_roles(self) -> None:
        ids = {
            entity_node_id("person", n, episode_id=_EP)
            for n in ["SPEAKER_00", "SPEAKER_03", "Host", "Guest"]
        }
        assert len(ids) == 4

    def test_a_placeholder_is_recognised_by_the_FILTER_not_just_the_prefix(self) -> None:
        """The property this file originally got WRONG.

        It asserted only `.startswith("person:speaker-")`, which is true and useless: the filter
        twelve modules actually consult required trailing DIGITS, so `person:speaker-ep1-host`
        passed the prefix check and FAILED the filter. That would have surfaced one followable
        "Host" person per episode — strictly worse than the single global phantom (#2059).
        """
        from podcast_scraper.enrichment.enrichers._loaders import (
            is_unresolved_speaker_placeholder,
        )

        node_id = entity_node_id("person", "Host", episode_id=_EP)
        assert is_unresolved_speaker_placeholder(node_id, "Host")

    @pytest.mark.parametrize("label", ROLE_LABELS)
    def test_every_role_scoped_id_is_filtered(self, label: str) -> None:
        from podcast_scraper.enrichment.enrichers._loaders import (
            is_unresolved_speaker_placeholder,
        )

        assert is_unresolved_speaker_placeholder(
            entity_node_id("person", label, episode_id=_EP), label
        )

    def test_the_legacy_global_role_id_is_filtered_without_a_re_derive(self) -> None:
        # `person:host` is on disk now — 54 episodes, 1,437 grounded insights. Filtering it by id
        # removes the phantom from every surface immediately; the artifacts keep it until
        # re-derived.
        from podcast_scraper.enrichment.enrichers._loaders import (
            is_unresolved_speaker_placeholder,
        )

        assert is_unresolved_speaker_placeholder("person:host", "Host")
        assert is_unresolved_speaker_placeholder("person:guest", "Guest")

    def test_a_real_person_is_not_filtered(self) -> None:
        from podcast_scraper.enrichment.enrichers._loaders import (
            is_unresolved_speaker_placeholder,
        )

        for name in REAL_NAMES:
            assert not is_unresolved_speaker_placeholder(entity_node_id("person", name), name), name

    def test_speaker_john_knight_is_a_person_not_a_placeholder(self) -> None:
        # The over-match the role-slug allowlist exists to prevent: this id starts with
        # `person:speaker-` and must still be a real human.
        from podcast_scraper.graph_id_utils import is_scoped_placeholder_person_id

        assert not is_scoped_placeholder_person_id("person:speaker-john-knight")


class TestRealPeopleAreUntouched:
    """Over-matching would erase real humans from the graph — worse than the phantom."""

    @pytest.mark.parametrize("name", REAL_NAMES)
    def test_a_real_name_is_not_a_placeholder(self, name: str) -> None:
        assert not is_bare_speaker_label(name)

    @pytest.mark.parametrize("name", REAL_NAMES)
    def test_a_real_name_keeps_its_global_followable_id(self, name: str) -> None:
        # Episode-scoping a real person would make them unfollowable across episodes.
        assert entity_node_id("person", name, episode_id=_EP) == entity_node_id("person", name)

    def test_names_that_merely_contain_a_role_word_survive(self) -> None:
        # "Hosteen Klah" and "Anna Hostetler" are real names; substring matching would eat them.
        for name in ("Hosteen Klah", "Anna Hostetler", "Guestavo Petro", "Speakman"):
            assert not is_bare_speaker_label(name), name

    @pytest.mark.parametrize("name", ["", "   ", None])
    def test_empty_input_is_not_a_placeholder(self, name) -> None:
        assert not is_bare_speaker_label(name)
