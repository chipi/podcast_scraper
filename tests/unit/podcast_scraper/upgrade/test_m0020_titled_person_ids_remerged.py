"""m0020: m0017's rename, re-run for titled ids minted after it — on m0017's own fixture.

Prod shape (2026-10-03 verify): person:professor-hannah-fry / person:dr-moriba-jah on episodes
ingested after m0017 ran, because the GI speaker path minted ids from the titled display name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest
import test_m0017_speaker_names_canonicalised as m17  # noqa: E402  (same directory, not a package)

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0017_speaker_names_canonicalised import (
    SpeakerNamesCanonicalisedMigration,
)
from podcast_scraper.upgrade.migrations.m0020_titled_person_ids_remerged import (
    TitledPersonIdsRemergedMigration,
    undo,
)
from podcast_scraper.upgrade.registry import get_migrations

PLAIN_ID, TITLED, TITLED_ID = m17.PLAIN_ID, m17.TITLED, m17.TITLED_ID
_ids, _r, _snapshot = m17._ids, m17._r, m17._snapshot

pytestmark = [pytest.mark.unit]


@pytest.fixture
def corpus(tmp_path: Path) -> Dict[str, Path]:
    """m0017's own episode fixture: a titled id beside its plain twin, plus a lone titled id."""
    paths: Dict[str, Path] = m17._episode(tmp_path)
    paths["root"] = tmp_path
    return paths


def _run(root: Path, dry: bool = False):
    return TitledPersonIdsRemergedMigration().apply(MigrationContext(corpus_root=root, dry_run=dry))


def test_the_titled_id_merges_like_m0017_did(corpus: Dict[str, Path]) -> None:
    result = _run(corpus["root"])
    assert TITLED_ID in result.details["person_ids_changed"]
    for key in ("kg", "gi"):
        ids = _ids(_r(corpus[key]))
        assert TITLED_ID not in ids and ids.count(PLAIN_ID) == 1
    # The display name keeps its title, as in m0017.
    assert _r(corpus["meta"])["content"]["speakers"][1]["name"] == TITLED
    ok, msg = TitledPersonIdsRemergedMigration().verify(
        MigrationContext(corpus_root=corpus["root"])
    )
    assert ok, msg


def test_after_m0017_there_is_nothing_left_for_m0020(corpus: Dict[str, Path]) -> None:
    SpeakerNamesCanonicalisedMigration().apply(MigrationContext(corpus_root=corpus["root"]))
    before = _snapshot(corpus)
    result = _run(corpus["root"])
    assert result.details["episodes"] == []
    assert _snapshot(corpus) == before
    ok, _msg = TitledPersonIdsRemergedMigration().verify(
        MigrationContext(corpus_root=corpus["root"])
    )
    assert ok


def test_dry_run_writes_nothing_and_undo_restores(corpus: Dict[str, Path]) -> None:
    before = _snapshot(corpus)
    assert len(_run(corpus["root"], dry=True).details["episodes"]) == 1
    assert _snapshot(corpus) == before
    _run(corpus["root"])
    assert _snapshot(corpus) != before
    restored, refused = undo(corpus["root"])
    assert refused == [] and restored >= 1
    assert _snapshot(corpus) == before


def test_registered_after_0019() -> None:
    ids = [m.id for m in get_migrations()]
    assert ids.index("0020_titled_person_ids_remerged") == (
        ids.index("0019_descriptor_speaker_names_removed") + 1
    )
