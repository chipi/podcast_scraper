"""Integration tests for the daily post-episode recap digest (RFC-122 #2039).

Builds a tiny real corpus + a signed-in user with finished playback, and exercises the assembler
(0/1/many), the consent gates, the schema-valid envelope, and the typed one-click unsubscribe.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from jsonschema import Draft202012Validator

from podcast_scraper.server import (
    app_comms_store,
    app_digest_daily_recap,
    app_outbox_store,
    app_user_state,
)
from podcast_scraper.server.app_user_store import get_or_create_user
from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative
from podcast_scraper.server.slugs import slug_for_row

pytestmark = [pytest.mark.integration]

_NOW = 1_760_000_000  # a fixed epoch so "today" is deterministic
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEMA = json.loads(
    (_REPO_ROOT / "docs" / "api" / "delivery-envelope.schema.json").read_text(encoding="utf-8")
)


def _write_corpus(root: Path, *, stem: str = "0001-hello", ep_id: str = "ep1") -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {"feed_id": "myfeed", "title": "My Show", "url": "https://pod.example/f.xml"},
        "episode": {
            "episode_id": ep_id,
            "title": "Hello",
            "published_date": "2024-03-10T00:00:00",
            "description": "The publisher's own words about this episode.",
        },
        "summary": {"title": "Sum", "bullets": ["First point", "Second point"]},
        "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    gi = {
        "episode_id": ep_id,
        "nodes": [
            {
                "id": "insight:1",
                "type": "Insight",
                "properties": {"text": "Big claim.", "grounded": True},
            },
            {
                "id": "quote:1",
                "type": "Quote",
                "properties": {"text": "a memorable line", "speaker_name": "Jane Doe"},
            },
        ],
        "edges": [{"type": "SUPPORTED_BY", "from": "insight:1", "to": "quote:1"}],
    }
    (root / "metadata" / f"{stem}.gi.json").write_text(json.dumps(gi), encoding="utf-8")
    kg = {
        "episode_id": ep_id,
        "nodes": [{"id": "topic:ai", "type": "Topic", "properties": {"label": "AI"}}],
    }
    (root / "metadata" / f"{stem}.kg.json").write_text(json.dumps(kg), encoding="utf-8")


def _slug(root: Path) -> str:
    return slug_for_row(build_catalog_rows_cumulative(root)[0])


def _user(data_dir: Path) -> str:
    return get_or_create_user(
        data_dir, provider="google", subject="s1", email="u@gmail.com", name="U"
    ).user_id


def _finish(data_dir: Path, uid: str, slug: str, *, when: int = _NOW) -> None:
    app_user_state.set_playback(data_dir, uid, slug, 120.0, updated_at=when, finished=True)


def test_assemble_recaps_todays_finished_episode(tmp_path: Path) -> None:
    root, data_dir = tmp_path / "corpus", tmp_path / "app"
    _write_corpus(root)
    uid = _user(data_dir)
    slug = _slug(root)
    _finish(data_dir, uid, slug)

    payload = app_digest_daily_recap.assemble_daily_recap_payload(root, data_dir, uid, _NOW)
    assert payload is not None
    assert payload["count"] == 1
    ep = payload["episodes"][0]
    assert ep["slug"] == slug
    assert ep["title"] == "Hello"
    assert ep["key_points"] == ["First point", "Second point"]
    assert ep["signature_quote"] == {"text": "a memorable line", "speaker": "Jane Doe"}
    assert ep["deep_link"] == f"/episode/{slug}"
    assert {"id": "topic:ai", "label": "AI"} in ep["topics"]
    # The publisher's description, as every other email episode carries (operator 2026-10-05).
    assert ep["description"] == "The publisher's own words about this episode."


def test_assemble_returns_none_when_nothing_finished_today(tmp_path: Path) -> None:
    root, data_dir = tmp_path / "corpus", tmp_path / "app"
    _write_corpus(root)
    uid = _user(data_dir)
    slug = _slug(root)
    # Finished, but a day earlier → not today.
    _finish(data_dir, uid, slug, when=_NOW - 86_400)
    assert app_digest_daily_recap.assemble_daily_recap_payload(root, data_dir, uid, _NOW) is None
    # And a started-but-unfinished episode today does not count either.
    app_user_state.set_playback(data_dir, uid, slug, 5.0, updated_at=_NOW, finished=False)
    assert app_digest_daily_recap.assemble_daily_recap_payload(root, data_dir, uid, _NOW) is None


def test_enqueue_requires_opt_in_and_produces_a_schema_valid_envelope(tmp_path: Path) -> None:
    root, data_dir = tmp_path / "corpus", tmp_path / "app"
    _write_corpus(root)
    uid = _user(data_dir)
    slug = _slug(root)
    _finish(data_dir, uid, slug)

    # Opted OUT by default → no envelope even with a finished episode.
    assert app_digest_daily_recap.enqueue_for_user(root, data_dir, uid, _NOW) is None

    # Opt in → an envelope is enqueued.
    app_comms_store.set_comms(data_dir, uid, types={"daily_recap": {"email": True}})
    eid = app_digest_daily_recap.enqueue_for_user(root, data_dir, uid, _NOW)
    assert eid is not None and eid.startswith("drcp_")

    pending = app_outbox_store.list_pending(data_dir, channel="email", now=_NOW)
    env = next(e for e in pending if e["id"] == eid)
    assert env["type"] == "daily_recap"
    assert env["template"] == "daily-recap.v1"
    assert env["channel"] == "email"
    assert env["recipient"]["email"] == "u@gmail.com"
    # It satisfies the committed app<->infra delivery seam contract.
    Draft202012Validator(_SCHEMA).validate(env)


def test_enqueue_paused_returns_none(tmp_path: Path) -> None:
    root, data_dir = tmp_path / "corpus", tmp_path / "app"
    _write_corpus(root)
    uid = _user(data_dir)
    _finish(data_dir, uid, _slug(root))
    app_comms_store.set_comms(data_dir, uid, types={"daily_recap": {"email": True}})
    app_comms_store.set_comms(data_dir, uid, daily_recap_schedule={"paused": True})
    assert app_digest_daily_recap.enqueue_for_user(root, data_dir, uid, _NOW) is None


def test_typed_unsubscribe_silences_only_the_recap(tmp_path: Path) -> None:
    data_dir = tmp_path / "app"
    uid = _user(data_dir)
    comms = app_comms_store.set_comms(
        data_dir, uid, types={"digest": {"email": True}, "daily_recap": {"email": True}}
    )
    ref = comms["unsubscribe_ref"]
    assert app_comms_store.unsubscribe(data_dir, ref, "daily_recap") is True
    after = app_comms_store.get_comms(data_dir, uid)
    assert after["types"]["daily_recap"]["email"] is False  # the recap is off
    assert after["types"]["digest"]["email"] is True  # the weekly digest is untouched


def test_due_slot_matches_the_configured_hour(tmp_path: Path) -> None:
    data_dir = tmp_path / "app"
    uid = _user(data_dir)
    comms = app_comms_store.get_comms(data_dir, uid)  # default hour = 22, no tz → UTC
    # 2025-10-09 09:00 UTC is hour 9, not 22.
    nine_utc = 1_760_000_400
    assert app_digest_daily_recap._is_due_slot(comms, nine_utc) is False
    # Bump to the top of hour 22 UTC on the same day.
    twenty_two_utc = nine_utc + (22 - 9) * 3600
    assert app_digest_daily_recap._is_due_slot(comms, twenty_two_utc) is True


def _epoch_local(y: int, mo: int, d: int, h: int, tz: str) -> int:
    """The epoch for a wall-clock hour in a given IANA zone (so tests read in local time)."""
    return int(dt.datetime(y, mo, d, h, 0, tzinfo=ZoneInfo(tz)).timestamp())


def test_due_slot_fires_at_the_users_local_hour_across_dst() -> None:
    # #2041: the configured hour is LOCAL. The SAME config (9pm New York) fires in summer (EDT) and
    # winter (EST) — proving zoneinfo, not a frozen offset. 8pm local is never due.
    comms = {"timezone": "America/New_York", "daily_recap_schedule": {"hour": 21, "paused": False}}
    NY = "America/New_York"
    assert app_digest_daily_recap._is_due_slot(comms, _epoch_local(2026, 7, 15, 21, NY)) is True
    assert app_digest_daily_recap._is_due_slot(comms, _epoch_local(2026, 7, 15, 20, NY)) is False
    assert app_digest_daily_recap._is_due_slot(comms, _epoch_local(2026, 1, 15, 21, NY)) is True
    assert app_digest_daily_recap._is_due_slot(comms, _epoch_local(2026, 1, 15, 20, NY)) is False
    # No tz → UTC fallback (unchanged behavior); an invalid tz also falls back to UTC.
    for tz in ("", "Not/AZone"):
        c = {"timezone": tz, "daily_recap_schedule": {"hour": 21, "paused": False}}
        assert app_digest_daily_recap._is_due_slot(c, _epoch_local(2026, 7, 15, 21, "UTC")) is True
        assert app_digest_daily_recap._is_due_slot(c, _epoch_local(2026, 7, 15, 21, NY)) is False


def test_finished_today_buckets_by_the_users_local_day(tmp_path: Path) -> None:
    # An episode finished at 2026-07-16 05:00 Tokyo is the 16th LOCALLY but still the 15th in UTC.
    # With now = 2026-07-16 11:00 Tokyo, it counts as "today" only under the user's tz.
    data_dir = tmp_path / "app"
    uid = _user(data_dir)
    finish = _epoch_local(2026, 7, 16, 5, "Asia/Tokyo")  # = 2026-07-15 20:00 UTC
    now = _epoch_local(2026, 7, 16, 11, "Asia/Tokyo")  # = 2026-07-16 02:00 UTC
    app_user_state.set_playback(data_dir, uid, "ep-x", 100.0, updated_at=finish, finished=True)
    assert app_digest_daily_recap._finished_today(data_dir, uid, now, "Asia/Tokyo") == ["ep-x"]
    assert app_digest_daily_recap._finished_today(data_dir, uid, now, None) == []  # UTC: yesterday


# --- the traveller case (operator 2026-09-23) -------------------------------------------------
#
# Reported: "I listened to two episodes offline on a two-hour flight, finished a third after
# landing, and only got an email about ONE." Same day, same timezone -- so no local-day edge, and
# the 0/1/many rule means ONE email is correct. The question is what that email CONTAINED.
#
# Offline finishes replay with their own timestamp (#1913), so they land BACKDATED by the length of
# the flight. These pin that a backdated finish is still today's finish.


def _write_corpus_n(root: Path, n: int) -> list[str]:
    """n episodes in one corpus, so a digest can carry more than one."""
    for i in range(n):
        # Distinct episode_id per file: the catalog keys on it, so three files sharing "ep1"
        # collapse to one row and the fan-out this test needs never exists.
        _write_corpus(root, stem=f"000{i + 1}-ep{i + 1}", ep_id=f"ep{i + 1}")
    rows = build_catalog_rows_cumulative(root)
    return [slug_for_row(r) for r in rows]


def test_a_digest_carries_EVERY_episode_finished_today_including_backdated_ones(
    tmp_path: Path,
) -> None:
    """Two finished offline hours ago + one finished just now = one email naming all three.

    The offline pair are stamped when the listener actually finished them -- mid-flight -- which is
    hours before the replay. Same local day, so all three belong in today's recap.
    """
    root, data_dir = tmp_path / "corpus", tmp_path / "app"
    slugs = _write_corpus_n(root, 3)
    uid = _user(data_dir)

    two_hours = 2 * 60 * 60
    _finish(data_dir, uid, slugs[0], when=_NOW - two_hours)  # in the air
    _finish(data_dir, uid, slugs[1], when=_NOW - two_hours + 600)  # in the air
    _finish(data_dir, uid, slugs[2], when=_NOW)  # after landing

    payload = app_digest_daily_recap.assemble_daily_recap_payload(root, data_dir, uid, _NOW)
    assert payload is not None, "three episodes finished today must produce a digest"
    assert payload["count"] == 3, (
        f"the email named {payload['count']} episode(s); the listener finished 3 today. "
        "A backdated (offline-replayed) finish is still today's finish."
    )
    assert {ep["slug"] for ep in payload["episodes"]} == set(slugs)


def test_the_replay_ORDER_does_not_decide_what_the_digest_carries(tmp_path: Path) -> None:
    """Writing the online finish FIRST, then replaying the offline pair, must not lose them.

    This is the real sequence: the third episode finishes while connected, and only afterwards does
    the outbox drain the two from the flight. A digest assembled from whatever arrived first would
    carry one episode -- the reported symptom.
    """
    root, data_dir = tmp_path / "corpus", tmp_path / "app"
    slugs = _write_corpus_n(root, 3)
    uid = _user(data_dir)

    _finish(data_dir, uid, slugs[2], when=_NOW)  # lands first, chronologically last
    _finish(data_dir, uid, slugs[0], when=_NOW - 7200)  # replayed after
    _finish(data_dir, uid, slugs[1], when=_NOW - 6600)

    payload = app_digest_daily_recap.assemble_daily_recap_payload(root, data_dir, uid, _NOW)
    assert payload is not None
    assert payload["count"] == 3, f"arrival order dropped episodes: got {payload['count']} of 3"
    # Most-recently-FINISHED first, by the finish time -- not by when the write arrived.
    assert payload["episodes"][0]["slug"] == slugs[2]
