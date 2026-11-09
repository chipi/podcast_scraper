"""Which member of a joined person lends its bio or hosted photo to the others (moved from
``server.relational_view``, E04 S3; record s6f_person_web)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from podcast_scraper.identity.web_borrows import _group_borrows

pytestmark = pytest.mark.unit

LEDGER = {
    "clusters_payload": {
        "clusters": [
            {
                "canonical_id": "person:ada-lovelace",
                "display_name": "Ada Lovelace",
                "members": [
                    {"id": "person:ada-lovelace", "episode_count": 1},
                    {"id": "person:ada-lovelace-2", "episode_count": 5},
                    {"id": "person:a-lovelace", "episode_count": 9},
                ],
            },
            {"canonical_id": "org:acme", "display_name": "Acme", "members": [{"id": "org:x"}]},
        ]
    }
}


def _borrows(rows: list) -> tuple[dict[str, str], dict[str, str]]:
    with patch("podcast_scraper.identity.ledger.read_ledger", return_value=(LEDGER, "ok")):
        result: tuple[dict[str, str], dict[str, str]] = _group_borrows(Path("/c"), rows)
        return result


def test_the_busiest_member_with_the_groups_spelling_lends_its_bio_and_photo() -> None:
    rows = [
        {"person_id": "person:ada-lovelace", "name": "Ada Lovelace"},
        {
            "person_id": "person:ada-lovelace-2",
            "name": "Ada Lovelace",
            "bio": "Mathematician.",
            "image_hosted": True,
        },
        # Most episodes, but under another spelling: it may describe someone else.
        {
            "person_id": "person:a-lovelace",
            "name": "A. Lovelace",
            "bio": "Other.",
            "image_hosted": True,
        },
    ]
    bio_from, photo_from = _borrows(rows)
    assert bio_from == {"person:ada-lovelace": "person:ada-lovelace-2"}
    assert photo_from == {"person:ada-lovelace": "person:ada-lovelace-2"}


def test_nothing_is_borrowed_when_no_member_with_the_spelling_has_one() -> None:
    rows = [
        {"person_id": "person:ada-lovelace", "name": "Ada Lovelace", "bio": "  "},
        {
            "person_id": "person:a-lovelace",
            "name": "A. Lovelace",
            "bio": "Other.",
            "image_hosted": True,
        },
    ]
    assert _borrows(rows) == ({}, {})


def test_without_a_valid_ledger_nothing_is_borrowed() -> None:
    with patch("podcast_scraper.identity.ledger.read_ledger", return_value=(None, "missing")):
        assert _group_borrows(Path("/c"), [{"person_id": "person:x", "bio": "b"}]) == ({}, {})
