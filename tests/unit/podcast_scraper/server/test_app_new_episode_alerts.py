"""Unit tests for the new-episode alert sweep (wave-J)."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server import (
    app_comms_store,
    app_digest_sections,
    app_new_episode_alerts,
    app_notifications_store,
)

pytestmark = pytest.mark.unit

_UID = "u_0123456789abcdef01234567"
_ROOT = Path("/unused")  # new_in_follows_items is stubbed; the corpus root is never read.


def _stub_follows(monkeypatch: pytest.MonkeyPatch, items: list[dict]) -> None:
    monkeypatch.setattr(app_digest_sections, "new_in_follows_items", lambda *a, **k: list(items))


def test_sweep_emits_one_alert_per_new_episode(tmp_path: Path, monkeypatch) -> None:
    _stub_follows(
        monkeypatch,
        [
            {"episode_slug": "ep-a", "episode_title": "A", "deep_link": "/player/ep-a"},
            {"episode_slug": "ep-b", "episode_title": "B", "deep_link": "/player/ep-b"},
        ],
    )
    emitted = app_new_episode_alerts.sweep_for_user(_ROOT, tmp_path, _UID)
    assert len(emitted) == 2
    inbox = app_notifications_store.list_notifications(tmp_path, _UID)
    assert {n["title"] for n in inbox} == {"A", "B"}
    assert all(n["type"] == "new_episodes" for n in inbox)
    assert app_notifications_store.unread_count(tmp_path, _UID) == 2


def test_sweep_is_idempotent_across_runs(tmp_path: Path, monkeypatch) -> None:
    _stub_follows(
        monkeypatch, [{"episode_slug": "ep-a", "episode_title": "A", "deep_link": "/player/ep-a"}]
    )
    first = app_new_episode_alerts.sweep_for_user(_ROOT, tmp_path, _UID)
    second = app_new_episode_alerts.sweep_for_user(_ROOT, tmp_path, _UID)
    assert len(first) == 1 and second == []  # deduped by slug
    assert app_notifications_store.unread_count(tmp_path, _UID) == 1


def test_sweep_respects_in_app_consent(tmp_path: Path, monkeypatch) -> None:
    _stub_follows(
        monkeypatch, [{"episode_slug": "ep-a", "episode_title": "A", "deep_link": "/player/ep-a"}]
    )
    # Turn the in-app channel off for new_episodes → the sweep emits nothing.
    app_comms_store.set_comms(tmp_path, _UID, types={"new_episodes": {"in_app": False}})
    assert app_new_episode_alerts.sweep_for_user(_ROOT, tmp_path, _UID) == []
    assert app_notifications_store.unread_count(tmp_path, _UID) == 0


def test_sweep_no_follows_is_noop(tmp_path: Path, monkeypatch) -> None:
    _stub_follows(monkeypatch, [])
    assert app_new_episode_alerts.sweep_for_user(_ROOT, tmp_path, _UID) == []
    assert app_notifications_store.list_notifications(tmp_path, _UID) == []
