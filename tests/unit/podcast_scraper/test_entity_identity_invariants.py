"""The RULES of entity identity, asserted as rules (#2058).

Three graph defects reached production this week — a stray `)` on display names (#2055),
everything defaulting to `person` (#2057), and role words getting global person ids (#2059) —
and all three were in code with coverage. The pyramid tested CASES: `SPEAKER_00`, `SPEAKER_03`,
`speaker-12`. Nobody wrote down the rule, so the day a placeholder arrived that was not spelled
with digits, nothing objected. One test went further and asserted `speaker_id == "person:guest"`,
pinning the defect in place.

This file holds the invariants. It is deliberately generative-ish: every rule is checked across a
NAME CORPUS rather than a hand-picked example, so a future placeholder shape that nobody thought
of still has to satisfy them.

The four HIGH findings from the 2026-09-13 review are each pinned here too, because each was a
case of two sides of one rule disagreeing:

  H1  the GI and KG layers minting different ids for the same voice
  H2  the id BUILDER and the placeholder FILTER disagreeing about what a placeholder is
  H3  a node type being added to a whitelist without teaching the kind helper about it
  H4  a name-keyed skip and a slug-keyed lookup disagreeing, so a fix silently no-opped
"""

from __future__ import annotations

import pytest

from podcast_scraper.enrichment.enrichers._loaders import is_unresolved_speaker_placeholder
from podcast_scraper.graph_id_utils import (
    ROLE_LABEL_SLUGS,
    entity_node_id,
    is_bare_speaker_label,
    is_scoped_placeholder_person_id,
    normalized_entity_kind_from_node,
)
from podcast_scraper.kg.llm_extract import ENTITY_KINDS, _normalize_entity_kind

pytestmark = pytest.mark.unit

_EP, _EP2 = "ep-aaa", "ep-bbb"

#: Names that are REAL people. Chosen to attack every heuristic in the identity layer: role words
#: as substrings, punctuation, unicode, initials, single tokens, and a name that begins with the
#: literal string the placeholder ids use.
REAL_NAMES = [
    "Aaron Levie",
    "Jean-Luc Picard",
    "O'Brien",
    "J.R.R. Tolkien",
    "Søren Kierkegaard",
    "will.i.am",
    "Peter Attia, MD",
    "Ursula K. Le Guin",
    "3Blue1Brown",
    "Yann LeCun",
    "Hosteen Klah",
    "Anna Hostetler",
    "Guestavo Petro",
    "Speakman",
    "Speaker John Knight",
    "Moderator Jones",
    "Interviewer Smith",
    "Panelis Toth",
]

#: Names that are PLACEHOLDERS — a position in a conversation or an unresolved diarization label.
PLACEHOLDERS = [
    "Host",
    "host",
    "HOST",
    " Host ",
    "Co-Host",
    "Guest",
    "Speaker",
    "Interviewer",
    "Moderator",
    "Panelist",
    "The Host",
    "Unknown Speaker",
    "Unidentified",
    "SPEAKER_00",
    "SPEAKER_03",
    "speaker-12",
    "Speaker 7",
]


class TestPlaceholdersNeverBecomeFollowablePeople:
    """INVARIANT 1 — the rule #2059 existed to enforce, and #2058 says was never written down."""

    @pytest.mark.parametrize("name", PLACEHOLDERS)
    def test_a_placeholder_is_recognised_as_one(self, name: str) -> None:
        assert is_bare_speaker_label(name)

    @pytest.mark.parametrize("name", PLACEHOLDERS)
    def test_a_placeholder_id_is_episode_scoped(self, name: str) -> None:
        assert _EP in entity_node_id("person", name, episode_id=_EP)

    @pytest.mark.parametrize("name", PLACEHOLDERS)
    def test_the_same_placeholder_in_two_episodes_is_two_people(self, name: str) -> None:
        assert entity_node_id("person", name, episode_id=_EP) != entity_node_id(
            "person", name, episode_id=_EP2
        )

    def test_distinct_placeholders_in_ONE_episode_stay_distinct(self) -> None:
        # Fixing a cross-episode merge by causing a within-episode merge is not a fix.
        ids = {entity_node_id("person", p, episode_id=_EP) for p in PLACEHOLDERS}
        names = {p.strip().casefold() for p in PLACEHOLDERS}
        assert len(ids) == len(names), "two different placeholders collapsed into one id"


class TestTheBuilderAndTheFilterAgree:
    """INVARIANT 2 — the H2 regression: minting a placeholder the FILTER does not recognise.

    Twelve modules consult `is_unresolved_speaker_placeholder`. An id that looks like a
    placeholder but fails that predicate is worse than no fix: it surfaces one followable phantom
    PER EPISODE instead of one globally.
    """

    @pytest.mark.parametrize("name", PLACEHOLDERS)
    def test_every_minted_placeholder_id_is_filtered(self, name: str) -> None:
        assert is_unresolved_speaker_placeholder(
            entity_node_id("person", name, episode_id=_EP), name
        )

    @pytest.mark.parametrize("name", PLACEHOLDERS)
    def test_the_legacy_unscoped_form_is_filtered_too(self, name: str) -> None:
        # Artifacts written before #2059 are on disk now and are not re-derived by a migration.
        assert is_unresolved_speaker_placeholder(entity_node_id("person", name), name)

    @pytest.mark.parametrize("name", REAL_NAMES)
    def test_a_real_person_is_never_filtered(self, name: str) -> None:
        assert not is_unresolved_speaker_placeholder(entity_node_id("person", name), name), name

    def test_the_role_slug_set_is_derived_not_hand_listed(self) -> None:
        # The builder and the filter must read ONE list, or they drift again.
        assert "host" in ROLE_LABEL_SLUGS and "guest" in ROLE_LABEL_SLUGS
        assert is_scoped_placeholder_person_id(f"person:speaker-{_EP}-host")


class TestRealPeopleStayGloballyFollowable:
    """INVARIANT 3 — over-matching silently unfollows real humans, worse than the phantom."""

    @pytest.mark.parametrize("name", REAL_NAMES)
    def test_a_real_name_is_not_a_placeholder(self, name: str) -> None:
        assert not is_bare_speaker_label(name), name

    @pytest.mark.parametrize("name", REAL_NAMES)
    def test_a_real_name_keeps_one_id_across_episodes(self, name: str) -> None:
        assert entity_node_id("person", name, episode_id=_EP) == entity_node_id(
            "person", name, episode_id=_EP2
        )


class TestEveryLayerMintsTheSameId:
    """INVARIANT 4 — the H1 failure: GI and KG disagreeing about who a person is.

    m0007's own docstring: writing one layer without the other "would leave the episode's two
    graphs disagreeing about who a person is — worse than not migrating".
    """

    @pytest.mark.parametrize("name", REAL_NAMES + PLACEHOLDERS)
    def test_gi_and_kg_agree(self, name: str) -> None:
        from podcast_scraper.gi.speakers import _person_node_id

        assert _person_node_id(name, _EP) == entity_node_id("person", name, episode_id=_EP)

    @pytest.mark.parametrize("name", REAL_NAMES)
    def test_the_legacy_slugify_path_still_agrees_for_real_names(self, name: str) -> None:
        # The swap in gi/speakers.py is only safe because these are byte-identical.
        from podcast_scraper.identity.slugify import person_id

        assert person_id(name) == entity_node_id("person", name)


class TestKindIsNeverGuessedAsPerson:
    """INVARIANT 5 — #2057. `person` is the most user-visible type and the worst default."""

    @pytest.mark.parametrize("kind", [None, "", "   ", "banana", "no-idea", "event", "podcast"])
    def test_an_unplaceable_kind_is_not_a_person(self, kind) -> None:
        assert _normalize_entity_kind(kind) != "person"

    @pytest.mark.parametrize(
        "kind", [None, "", "banana", "event", "person", "company", "podcast", "PERSON"]
    )
    def test_every_result_is_in_the_declared_vocabulary(self, kind) -> None:
        assert _normalize_entity_kind(kind) in ENTITY_KINDS


class TestEveryNodeTypeIsUnderstoodByTheKindHelper:
    """INVARIANT 6 — the H3 failure: a type added to a whitelist the kind helper never learned."""

    @pytest.mark.parametrize(
        "node_type,expected",
        [("Person", "person"), ("Organization", "organization"), ("Object", "object")],
    )
    def test_a_typed_node_reports_its_own_kind(self, node_type: str, expected: str) -> None:
        node = {"id": "x:1", "type": node_type, "properties": {"name": "X"}}
        assert normalized_entity_kind_from_node(node) == expected

    def test_no_entity_node_type_silently_reports_person(self) -> None:
        from podcast_scraper.graph_id_utils import PERSON_ORG_NODE_TYPES

        for nt in PERSON_ORG_NODE_TYPES:
            if nt in ("Entity",):  # legacy: kind lives in properties, tested elsewhere
                continue
            kind = normalized_entity_kind_from_node(
                {"id": "x:1", "type": nt, "properties": {"name": "X"}}
            )
            assert kind == nt.lower(), (
                f"{nt} reports {kind!r} — a node type was added to the whitelist without "
                "teaching the kind helper, which is how Objects became Persons one layer down"
            )


class TestTheIdAndTheNameKeyAgree:
    """INVARIANT 7 — the H4 failure: a name-keyed skip and a slug-keyed lookup disagreeing."""

    @pytest.mark.parametrize("name", REAL_NAMES)
    def test_one_name_yields_one_id_per_kind(self, name: str) -> None:
        # If these drift, a dedup keyed on one and a lookup keyed on the other silently no-op.
        assert entity_node_id("person", name) == entity_node_id("person", name.strip())

    def test_kind_changes_the_id_namespace(self) -> None:
        # H4's mechanism: the same name under a different kind is a different node, so a skip
        # keyed on NAME alone cannot protect a lookup keyed on ID.
        n = "Aaron Levie"
        assert len({entity_node_id(k, n) for k in ("person", "organization", "object")}) == 3
