"""Every writer publishes the spelling the person id was minted from (#2130, m0010).

m0010 cleaned the corpus once; `upgrade verify` then found 52 names written AFTER it — KG nodes
(`Donald Trump Jr.`, `Lieutenant General John W. Brennan Jr.`) and roster entries (`Ali Ghodsi)`,
`Peter Attia, MD`) — because ids were canonicalised at minting and names were written raw.
"""

from __future__ import annotations

import pytest

from podcast_scraper.identity.slugify import person_id
from podcast_scraper.kg.pipeline import _typed_person_org_node

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
