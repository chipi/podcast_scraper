"""The display seam (E04 split line, decision 3): what every surface reads stays public, and the
builder behind it can be replaced.

With nothing registered the built-in builder is the identity ledger's, so the surfaces show what
they showed before the seam. The ledger side leaves public in the core release and registers
itself from Common; ``plain_display`` is what public shows on its own: every episode's own name.

Invented names only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest

from podcast_scraper.identity import display as D

pytestmark = pytest.mark.unit

_LEDGER = {
    "schema_version": 1,
    "clusters_payload": {
        "clusters": [
            {
                "canonical_id": "person:ada-lovlace",
                "display_name": "Ada Lovelace",
                "members": [{"id": "person:ada-lovlace"}, {"id": "person:ada-lovelace-2"}],
            }
        ]
    },
}


@pytest.fixture(autouse=True)
def _built_in_builder() -> Iterator[None]:
    D.set_display_builder(None)
    yield
    D.set_display_builder(None)


def test_with_nothing_registered_the_ledger_builder_answers(tmp_path: Path) -> None:
    from podcast_scraper.identity.ledger_display import build_from_ledger

    assert D._current_builder() is build_from_ledger


def test_the_ledger_view_is_the_same_through_either_module() -> None:
    from podcast_scraper.identity import ledger_display as LD

    for view in (D.display_from_ledger(_LEDGER), LD.display_from_ledger(_LEDGER)):
        assert view.shown("person:ada-lovelace-2", "Ada Lovlace") == "Ada Lovelace"
        assert view.shown("person:someone-else", "Own Name") == "Own Name"
    assert len(D.display_from_ledger(None)) == 0


def test_a_registered_builder_answers_every_surface(tmp_path: Path) -> None:
    seen = []

    def builder(root: Path) -> D.PersonDisplay:
        seen.append(root)
        return D.PersonDisplay({"person:x": "Registered Spelling"})

    D.set_display_builder(builder)
    assert D.person_display(tmp_path).shown("person:x", "Own") == "Registered Spelling"
    assert seen == [tmp_path]


def test_the_plain_builder_shows_each_episodes_own_name(tmp_path: Path) -> None:
    D.set_display_builder(D.plain_display)
    view = D.person_display(tmp_path)
    assert len(view) == 0
    assert view.shown("person:ada-lovelace-2", "Ada Lovlace") == "Ada Lovlace"
    assert view.name_of("person:ada-lovelace-2") is None


def test_a_view_cached_under_one_builder_is_not_served_under_another(tmp_path: Path) -> None:
    D.set_display_builder(lambda root: D.PersonDisplay({"person:x": "First"}))
    assert D.person_display(tmp_path).name_of("person:x") == "First"
    D.set_display_builder(D.plain_display)
    assert D.person_display(tmp_path).name_of("person:x") is None


def test_none_restores_the_built_in_builder(tmp_path: Path) -> None:
    (tmp_path / "identity_ledger.json").write_text(json.dumps(_LEDGER), encoding="utf-8")
    D.set_display_builder(D.plain_display)
    D.set_display_builder(None)
    from podcast_scraper.identity.ledger_display import build_from_ledger

    assert D._current_builder() is build_from_ledger
