"""Cross-user work takes its account list from the users store, not from users/ directories.

Accounts are rows (ADR-169); a directory only exists once a file-backed store has written for the
person. An account with a row and no directory must still be counted, and a directory with no row
is not an account.
"""

from __future__ import annotations

from pathlib import Path

from podcast_scraper.server import (
    app_comms_store,
    app_engagement_series,
    app_graph_telemetry,
    app_user_state,
)
from podcast_scraper.server.app_user_store import get_or_create_user, list_user_ids


def _account(root: Path, subject: str) -> str:
    return get_or_create_user(
        root, provider="google", subject=subject, email=f"{subject}@x.com", name=subject
    ).user_id


def test_list_user_ids_is_sorted_and_needs_no_directories(tmp_path: Path) -> None:
    ids = [_account(tmp_path, s) for s in ("a", "b", "c")]
    assert not (tmp_path / "users").exists()
    assert list_user_ids(tmp_path) == sorted(ids)


def test_iter_user_ids_includes_an_account_with_no_directory(tmp_path: Path) -> None:
    uid = _account(tmp_path, "fresh")
    orphan = "u_" + "9" * 24
    (tmp_path / "users" / orphan).mkdir(parents=True)
    assert app_user_state.iter_user_ids(tmp_path) == [uid]


def test_engagement_series_includes_an_account_with_no_directory(tmp_path: Path) -> None:
    uid = _account(tmp_path, "fresh")
    assert app_engagement_series._iter_user_ids(tmp_path, None) == [uid]
    assert app_engagement_series._iter_user_ids(tmp_path, "u_x") == ["u_x"]


def test_graph_telemetry_reads_accounts_from_rows(tmp_path: Path) -> None:
    quiet = _account(tmp_path, "quiet")
    talker = _account(tmp_path, "talker")
    ghost = "u_" + "8" * 24  # a directory with events but no account
    app_graph_telemetry.record_events(tmp_path, talker, [{"action": "redraw", "ts": 1}])
    app_graph_telemetry.record_events(tmp_path, ghost, [{"action": "redraw", "ts": 2}])
    assert not (tmp_path / "users" / quiet).exists()
    assert [e["ts"] for e in app_graph_telemetry.read_all_events(tmp_path)] == [1]


def test_unsubscribe_scans_accounts_from_rows(tmp_path: Path) -> None:
    _account(tmp_path, "no-files-yet")
    assert app_comms_store.unsubscribe(tmp_path, "unknown-ref") is False  # no users/ dir at all
    target = _account(tmp_path, "subscriber")
    app_comms_store.set_comms(tmp_path, target, types={"digest": {"email": True}})
    ref = app_comms_store.get_comms(tmp_path, target)["unsubscribe_ref"]
    assert app_comms_store.unsubscribe(tmp_path, ref) is True
    assert app_comms_store.get_comms(tmp_path, target)["types"]["digest"]["email"] is False
