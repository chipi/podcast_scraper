"""`POST /api/app/playback-progress/{slug}` — listening milestones (#2266).

An OPEN is not a listen: this route records that a person actually reached 25/50/75/95% of an
episode, which is what completion rate and the beta's active-day metrics are built from. It shipped
with no test at all (found 2026-10-03 when codecov/patch failed on PR #2274), so these lock the
behaviours the docstrings promise: one canonical event per milestone, the enum enforced at the edge,
offline redelivery not double-counted, and a device clock that cannot write into the far past.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions, app_user_state
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_user_store import get_or_create_user

pytestmark = [pytest.mark.integration]

_SLUG = "show-abc123"


def _client(tmp_path: Path) -> tuple[TestClient, Path, str]:
    app = create_app(tmp_path, static_dir=False)
    data_dir = tmp_path / "appdata"
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(data_dir, provider="stub", subject="p1", email="p@x.com", name="P")
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client, data_dir, user.user_id


def _events(data_dir: Path, user_id: str) -> list[dict]:
    path = data_dir / "users" / user_id / "playback_events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_each_milestone_is_recorded_once_as_a_canonical_event(tmp_path: Path) -> None:
    client, data_dir, uid = _client(tmp_path)
    now = int(time.time())
    for m in (25, 50, 75, 95):
        resp = client.post(
            f"/api/app/playback-progress/{_SLUG}", json={"milestone": m, "client_ts": now}
        )
        assert resp.status_code == 204
    events = _events(data_dir, uid)
    assert [e["milestone"] for e in events] == [25, 50, 75, 95]
    assert all(e["event_type"] == "playback_progress" and e["slug"] == _SLUG for e in events)


def test_a_milestone_outside_the_four_is_refused_at_the_edge(tmp_path: Path) -> None:
    """A free integer would let a caller invent 33, which no completion chart would ever count."""
    client, data_dir, uid = _client(tmp_path)
    resp = client.post(f"/api/app/playback-progress/{_SLUG}", json={"milestone": 33})
    assert resp.status_code == 422
    assert _events(data_dir, uid) == []


def test_an_offline_redelivery_is_not_counted_twice(tmp_path: Path) -> None:
    """The queue replays an event that never got a RESPONSE; both attempts carry the same client_ts."""
    client, data_dir, uid = _client(tmp_path)
    body = {"milestone": 50, "client_ts": int(time.time()) - 60}
    for _ in range(2):
        assert client.post(f"/api/app/playback-progress/{_SLUG}", json=body).status_code == 204
    assert [e["milestone"] for e in _events(data_dir, uid)] == [50]


def test_two_milestones_at_one_clamped_timestamp_are_both_kept(tmp_path: Path) -> None:
    """The dedupe key includes the milestone: crossing 50% must not be discarded as a replay of 25%."""
    client, data_dir, uid = _client(tmp_path)
    ts = int(time.time()) - 30
    for m in (25, 50):
        client.post(f"/api/app/playback-progress/{_SLUG}", json={"milestone": m, "client_ts": ts})
    assert [e["milestone"] for e in _events(data_dir, uid)] == [25, 50]


def test_a_wrong_device_clock_cannot_write_into_the_far_past(tmp_path: Path) -> None:
    client, data_dir, uid = _client(tmp_path)
    client.post(f"/api/app/playback-progress/{_SLUG}", json={"milestone": 25, "client_ts": 1})
    (event,) = _events(data_dir, uid)
    assert not event["ts"].startswith("1970"), event["ts"]


def test_the_route_requires_a_session(tmp_path: Path) -> None:
    client, data_dir, uid = _client(tmp_path)
    client.cookies.clear()
    assert (
        client.post(f"/api/app/playback-progress/{_SLUG}", json={"milestone": 25}).status_code
        == 401
    )
    assert _events(data_dir, uid) == []


def test_the_store_ignores_an_unknown_milestone_from_any_caller(tmp_path: Path) -> None:
    """A second guard below the route, so a future caller cannot invent a value either."""
    app_user_state.append_playback_progress(
        tmp_path, "u_" + "0" * 24, _SLUG, None, 33, int(time.time())
    )
    assert not (tmp_path / "users").exists()
