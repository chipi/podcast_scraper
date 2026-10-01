"""Every writer publishes the spelling the person id was minted from (#2130, m0010).

m0010 cleaned the corpus once; `upgrade verify` then found 52 names written AFTER it — KG nodes
(`Donald Trump Jr.`, `Lieutenant General John W. Brennan Jr.`) and roster entries (`Ali Ghodsi)`,
`Peter Attia, MD`) — because ids were canonicalised at minting and names were written raw.
"""

from __future__ import annotations

import pytest

from podcast_scraper.gi.pipeline import _attach_person_for_quote
from podcast_scraper.identity.slugify import person_id
from podcast_scraper.kg.pipeline import _typed_person_org_node
from podcast_scraper.workflow.metadata_generation import _unplaced_speakers

pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize(
    "raw, want",
    [
        ("Donald Trump Jr.", "Donald Trump Jr"),
        ("Aaron Levie)", "Aaron Levie"),
        ("Peter Attia, MD", "Peter Attia"),
    ],
)
def test_kg_person_node_name_matches_its_id(raw: str, want: str) -> None:
    node = _typed_person_org_node(name=raw, entity_kind="person", role="mentioned")
    assert node["properties"]["name"] == want
    assert node["properties"]["label"] == want
    assert node["id"] == person_id(raw)


def test_kg_organization_name_is_not_rewritten() -> None:
    """Credential and bracket rules are about PEOPLE; an org keeps its own spelling."""
    node = _typed_person_org_node(name="Acme, Inc.", entity_kind="organization", role="mentioned")
    assert node["properties"]["name"] == "Acme, Inc."


def test_unplaced_roster_entry_is_canonical() -> None:
    out = _unplaced_speakers(
        [],
        diagnostics={"tried": {"known_hosts": ["Aaron Levie)"]}, "summary": {}},
        detected_hosts=None,
        detected_guests=None,
        feed_title="Some Show",
    )
    assert [s.name for s in out] == ["Aaron Levie"]


def test_unplaced_show_name_filter_still_judges_the_stated_string() -> None:
    """The show-name check runs on the name AS STATED, exactly as before this change.

    `Peter Attia, MD` does not read as "The Peter Attia Drive", so it stays on the record — now
    spelled canonically. (That the bare `Peter Attia` DOES read as the show is a separate defect.)
    """
    out = _unplaced_speakers(
        [],
        diagnostics={"tried": {"known_hosts": ["Peter Attia, MD"]}, "summary": {}},
        detected_hosts=None,
        detected_guests=None,
        feed_title="The Peter Attia Drive",
    )
    assert [s.name for s in out] == ["Peter Attia"]


def test_gi_person_node_name_is_canonical() -> None:
    nodes: list = []
    edges: list = []
    _attach_person_for_quote(
        nodes,
        edges,
        "quote:1",
        "SPEAKER_00",
        "person:aaron-levie",
        set(),
        display_name="Aaron Levie)",
    )
    assert nodes[0]["properties"]["name"] == "Aaron Levie"
