"""What each storyline and theme run did, kept over time (RFC-128, #2349)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.search import clustering_history as h

pytestmark = pytest.mark.unit


def _c(cid: str, name: str, state: str | None = None) -> dict:
    out: dict = {"graph_compound_parent_id": cid, "canonical_label": name}
    if state:
        out["state"] = state
    return out


def test_changes_name_what_a_run_created_renamed_quieted_woke_absorbed_and_lost() -> None:
    before = [
        _c("thc:a", "risk", "active"),
        _c("thc:b", "trade", "quiet"),
        _c("thc:c", "rates", "active"),
        _c("thc:d", "oil", "active"),
        _c("thc:e", "small", "active"),
    ]
    after = [
        _c("thc:a", "risk management", "active"),
        _c("thc:b", "trade", "active"),
        _c("thc:c", "rates", "quiet"),
        _c("thc:new", "chips", "active"),
    ]
    got = h.changes(
        before,
        after,
        {"successors": {}, "gone": []},
        {"successors": {"thc:e": "thc:a"}, "gone": ["thc:d"]},
    )
    assert got == {
        "created": [{"id": "thc:new", "name": "chips"}],
        "renamed": [{"id": "thc:a", "from": "risk", "to": "risk management"}],
        "quiet": [{"id": "thc:c", "name": "rates"}],
        "woke": [{"id": "thc:b", "name": "trade"}],
        "absorbed": [
            {"id": "thc:e", "name": "small", "into": "thc:a", "into_name": "risk management"}
        ],
        "gone": [{"id": "thc:d", "name": "oil"}],
    }


def test_rows_read_oldest_first_and_a_torn_line_is_skipped(tmp_path: Path) -> None:
    h.append(tmp_path, {"ts": 2.0, "kind": "storyline", "n": 2})
    h.append(tmp_path, {"ts": 1.0, "kind": "storyline", "n": 1})
    h.append(tmp_path, {"ts": 3.0, "kind": "theme", "n": 3})
    with (tmp_path / h.HISTORY).open("a") as fh:
        fh.write('{"ts": 4')
    assert [r["n"] for r in h.read(tmp_path, "storyline")] == [1, 2]
    assert [r["n"] for r in h.read(tmp_path, "theme")] == [3]


def test_the_backfill_fills_only_the_past_and_is_replaced_not_appended(tmp_path: Path) -> None:
    h.append(tmp_path, {"ts": 100.0, "kind": "storyline", "n": "run"})
    past = [{"ts": t, "kind": "storyline", "n": f"m{t}"} for t in (10.0, 50.0, 120.0)]
    h.write_backfill(tmp_path, past)
    h.write_backfill(tmp_path, past)
    assert [r["n"] for r in h.read(tmp_path, "storyline")] == ["m10.0", "m50.0", "run"]


def test_seed_state_is_none_then_pending_then_taken(tmp_path: Path) -> None:
    enr = tmp_path / "enrichments"
    enr.mkdir()
    art = enr / "topic_theme_clusters.json"
    art.write_text(json.dumps({"data": {"registry": {}}}))
    assert h.seed_status(tmp_path) == {"state": "none", "id": None}
    (enr / "storyline_registry.seed.json").write_text(json.dumps({"id": "s1"}))
    assert h.seed_status(tmp_path) == {"state": "pending", "id": "s1"}
    art.write_text(json.dumps({"data": {"registry": {"seeded_from": "s1"}}}))
    assert h.seed_status(tmp_path) == {"state": "taken", "id": "s1"}


def test_appending_never_fails_a_run(tmp_path: Path) -> None:
    (tmp_path / "enrichments").write_text("a file where the directory should be")
    h.append(tmp_path, {"ts": 1.0, "kind": "storyline"})  # logged, not raised
