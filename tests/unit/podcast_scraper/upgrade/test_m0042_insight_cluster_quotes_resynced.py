"""m0042: the insight clusters' copied quote fields resynced from the GI on disk."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import m0042_insight_cluster_quotes_resynced as m40
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]


def _gi(root: Path, run: str, quote_props: dict) -> None:
    meta = root / "feeds" / "f" / run / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "ep.metadata.json").write_text(json.dumps({"episode": {"episode_id": "ep"}}))
    gi = {
        "episode_id": "ep",
        "nodes": [
            {"id": "insight:1", "type": "Insight", "properties": {"text": "A point."}},
            {"id": "quote:1", "type": "Quote", "properties": quote_props},
        ],
        "edges": [{"type": "SUPPORTED_BY", "from": "insight:1", "to": "quote:1"}],
    }
    (meta / "ep.gi.json").write_text(json.dumps(gi))


def _clusters(root: Path, quote: dict) -> Path:
    path = root / "search" / "insight_clusters.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    member = {
        "insight_id": "insight:1",
        "text": "A point.",
        "insight_type": "claim",
        "episode_id": "ep",
        "similarity_to_centroid": 0.91,
        "supporting_quotes": [{"quote_id": "quote:1", **quote}],
    }
    other = dict(
        member, insight_id="insight:9", supporting_quotes=[{"quote_id": "quote:9", "text": "x"}]
    )
    payload = {
        "schema_version": "1",
        "clusters": [
            {
                "cluster_id": "ic:a",
                "canonical_insight": "A point.",
                "member_count": 2,
                "episode_ids": ["ep"],
                "members": [member, other],
            }
        ],
    }
    path.write_text(json.dumps(payload))
    return path


def _apply(root: Path, dry_run: bool = False):
    return m40.InsightClusterQuotesResyncedMigration().apply(
        MigrationContext(corpus_root=root, dry_run=dry_run)
    )


def _verify(root: Path):
    return m40.InsightClusterQuotesResyncedMigration().verify(MigrationContext(corpus_root=root))


def test_registered_last() -> None:
    assert [m.id for m in get_migrations()][-1] == m40.MIGRATION_ID


def test_a_quote_takes_its_gi_quotes_current_fields_and_nothing_else_moves(tmp_path: Path) -> None:
    _gi(tmp_path, "run_1", {"text": "Said so.", "char_start": 12, "char_end": 20})  # id removed
    path = _clusters(
        tmp_path,
        {"text": "Said so.", "speaker_id": "person:speaker-01", "char_start": 10, "char_end": 18},
    )
    before = json.loads(path.read_text())
    assert not _verify(tmp_path)[0]
    _apply(tmp_path, dry_run=True)
    assert json.loads(path.read_text()) == before
    res = _apply(tmp_path)
    assert res.details["counts"]["quotes_resynced"] == 1 and res.details["counts"]["unmatched"] == 1
    after = json.loads(path.read_text())
    q = after["clusters"][0]["members"][0]["supporting_quotes"][0]
    assert q == {
        "quote_id": "quote:1",
        "text": "Said so.",
        "speaker_id": None,
        "char_start": 12,
        "char_end": 20,
    }
    a_member, b_member = after["clusters"][0]["members"][0], before["clusters"][0]["members"][0]
    assert {k: v for k, v in a_member.items() if k != "supporting_quotes"} == {
        k: v for k, v in b_member.items() if k != "supporting_quotes"
    }
    assert (
        after["clusters"][0]["members"][1] == before["clusters"][0]["members"][1]
    )  # unmatched kept
    assert {k: v for k, v in after["clusters"][0].items() if k != "members"} == {
        k: v for k, v in before["clusters"][0].items() if k != "members"
    }
    assert _verify(tmp_path)[0]
    assert _apply(tmp_path).details["files_written"] == 0  # a second run writes nothing
    restored, refused = m40.undo(tmp_path)
    assert restored and not refused and json.loads(path.read_text()) == before


def test_the_newest_served_run_wins_when_an_episode_has_several(tmp_path: Path) -> None:
    _gi(tmp_path, "run_20261001-000000", {"text": "Old.", "char_start": 1, "char_end": 5})
    _gi(tmp_path, "run_20261002-000000", {"text": "New.", "char_start": 2, "char_end": 6})
    _clusters(tmp_path, {"text": "Old.", "char_start": 1, "char_end": 5})
    quotes, duplicated = m40.gi_quotes(tmp_path)
    assert duplicated == 1 and quotes[("ep", "insight:1", "quote:1")]["text"] == "New."


def test_no_clusters_file_means_nothing_to_do(tmp_path: Path) -> None:
    _gi(tmp_path, "run_1", {"text": "Said so."})
    assert _verify(tmp_path)[0] and _apply(tmp_path).details["files_written"] == 0
