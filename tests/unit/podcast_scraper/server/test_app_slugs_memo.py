"""Episode slugs are memoised and ``slug_for_row`` stays patchable through ``episode_slug``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from podcast_scraper.server.slugs import episode_slug

pytestmark = [pytest.mark.unit]


def test_slugs_are_the_unmemoised_slugs_and_the_function_stays_patchable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from podcast_scraper.server import slugs

    args = ("Some Feed", "ep-1", "metadata/a.metadata.json")
    assert episode_slug(*args) == episode_slug.__wrapped__(*args)
    assert episode_slug("f", None, "p") == episode_slug.__wrapped__("f", None, "p")
    row = cast(
        Any,
        SimpleNamespace(feed_id="Some Feed", episode_id="ep-1", metadata_relative_path="m"),
    )
    assert slugs.slug_for_row(row) == episode_slug.__wrapped__("Some Feed", "ep-1", "m")
    monkeypatch.setattr(slugs, "episode_slug", lambda *a: "patched")
    assert slugs.slug_for_row(row) == "patched"
