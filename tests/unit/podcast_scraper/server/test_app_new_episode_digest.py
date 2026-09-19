"""New-episode alerts on email + push (#2124 / #2125).

The type shipped with three checkboxes and only in-app implemented. These lock the three
mechanisms that replace a send-slot for an event-driven type — per-episode dedupe, a rate
floor, and quiet hours — because getting any of them wrong is either silence or a burst, and
both end with the user muting the channel.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.server import app_new_episode_digest as ned

pytestmark = pytest.mark.unit


class _U:
    def __init__(self, uid="u1", provider="google", email="a@b.c"):
        self.user_id = uid
        self.provider = provider
        self.email = email


def _items(*slugs):
    """The REAL shape app_digest_sections.new_in_follows_items emits.

    The first version of this helper invented ``slug`` and ``published_at_ts``. The module read
    those same invented keys, so every test passed against a shape production never produces.
    Keep this identical to the source.
    """
    return [
        {
            "episode_slug": s,
            "episode_title": f"T-{s}",
            "graph_refs": [{"id": f"g-{s}"}],
            "deep_link": f"/episode/{s}",
        }
        for s in slugs
    ]


def _seed(tmp_path: Path, user_id="u1", announced=None, last=None):
    """Mark this user as already past the first-run seed, so enqueues are live."""
    d = tmp_path / "users" / user_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "new_episode_alerts.json").write_text(
        json.dumps({"announced": announced or {}, "last_send_ts": last, "seeded": 1}),
        encoding="utf-8",
    )


def _wire(monkeypatch, *, items, comms=None, user=None, subs=None):
    from podcast_scraper.server import app_comms_store, app_digest_sections, app_push_store

    base = {
        "types": {"new_episodes": {"email": True, "push": False}},
        "new_episodes_schedule": {
            "floor_minutes": 240,
            "quiet_start": 22,
            "quiet_end": 8,
            "paused": False,
        },
        "timezone": "UTC",
        "unsubscribe_ref": "ref",
    }
    if comms:
        base = {**base, **comms}
    monkeypatch.setattr(ned, "get_user", lambda d, uid: user if user is not None else _U(uid))
    monkeypatch.setattr(app_comms_store, "get_comms", lambda d, uid: base)
    monkeypatch.setattr(app_digest_sections, "new_in_follows_items", lambda *a, **k: items)
    monkeypatch.setattr(app_push_store, "list_subscriptions", lambda d, uid: subs or [])
    sent: list[dict] = []
    monkeypatch.setattr(
        "podcast_scraper.server.app_outbox_store.enqueue", lambda d, e: sent.append(e)
    )
    return sent


# 10:00 UTC — outside the default 22..08 quiet window.
NOON = 1789740000


def test_enqueues_an_email_for_a_new_episode(tmp_path: Path, monkeypatch) -> None:
    _seed(tmp_path)
    sent = _wire(monkeypatch, items=_items("ep1"))
    ids = ned.enqueue_for_user(tmp_path / "root", tmp_path, "u1", NOON)
    assert len(ids) == 1
    assert sent[0]["type"] == "new_episodes"
    assert sent[0]["channel"] == "email"
    assert sent[0]["template"] == "new-episodes.v1"
    assert sent[0]["payload"]["count"] == 1


def test_an_episode_is_never_announced_twice(tmp_path: Path, monkeypatch) -> None:
    """THE defining property of an event-driven type. A per-period id cannot express it."""
    _seed(tmp_path)
    sent = _wire(monkeypatch, items=_items("ep1"))
    first = ned.enqueue_for_user(tmp_path / "root", tmp_path, "u1", NOON)
    assert len(first) == 1
    # Same episode still unheard, well past the rate floor -> must stay silent.
    later = ned.enqueue_for_user(tmp_path / "root", tmp_path, "u1", NOON + 10 * 86400)
    assert later == []
    assert len(sent) == 1


def test_a_genuinely_new_episode_still_gets_through(tmp_path: Path, monkeypatch) -> None:
    _seed(tmp_path)
    _wire(monkeypatch, items=_items("ep1"))
    ned.enqueue_for_user(tmp_path / "root", tmp_path, "u1", NOON)
    _wire(monkeypatch, items=_items("ep1", "ep2"))  # ep1 announced, ep2 is new
    ids = ned.enqueue_for_user(tmp_path / "root", tmp_path, "u1", NOON + 10 * 86400)
    assert len(ids) == 1
    led = ned.read_ledger(tmp_path, "u1")
    assert set(led["announced"]) == {"ep1", "ep2"}


def test_rate_floor_suppresses_a_burst(tmp_path: Path, monkeypatch) -> None:
    """Following twenty active shows must not produce a dozen notifications in a morning."""
    _seed(tmp_path)
    _wire(monkeypatch, items=_items("ep1"))
    ned.enqueue_for_user(tmp_path / "root", tmp_path, "u1", NOON)
    _wire(monkeypatch, items=_items("ep2"))
    assert ned.enqueue_for_user(tmp_path / "root", tmp_path, "u1", NOON + 60) == []  # 1 min later
    # ...and once the floor has elapsed, the held episode goes out.
    assert ned.enqueue_for_user(tmp_path / "root", tmp_path, "u1", NOON + 4 * 3600 + 60) != []


def test_quiet_hours_hold_rather_than_drop(tmp_path: Path, monkeypatch) -> None:
    """A held episode must roll into the next allowed send. Burning it would be a silent loss."""
    _seed(tmp_path)
    _wire(monkeypatch, items=_items("ep1"))
    quiet = NOON + 13 * 3600  # 23:00 UTC, inside 22..08
    assert ned.enqueue_for_user(tmp_path / "root", tmp_path, "u1", quiet) == []
    assert (
        ned.read_ledger(tmp_path, "u1")["announced"] == {}
    ), "quiet hours must not consume episodes"
    out = ned.enqueue_for_user(tmp_path / "root", tmp_path, "u1", quiet + 11 * 3600)  # 10:00
    assert len(out) == 1


def _at_utc_hour(hour: int) -> int:
    """A timestamp at exactly ``hour`` UTC. Derived, not hard-coded — an offset-arithmetic
    mistake here silently tests the wrong hour, which is how the first version of this passed
    on one assertion and failed on another for the same reason."""
    import datetime

    base = datetime.datetime(2026, 9, 19, hour, 0, tzinfo=datetime.timezone.utc)
    return int(base.timestamp())


def test_quiet_window_wraps_midnight() -> None:
    """22->08 spans midnight and is the default.

    The naive ``start <= hour < end`` gets this exactly backwards — it would treat the whole
    working day as quiet and the night as awake."""
    sched = {"quiet_start": 22, "quiet_end": 8}
    for hour in (22, 23, 0, 3, 7):
        assert ned.in_quiet_hours(sched, _at_utc_hour(hour), "UTC") is True, f"{hour}:00 quiet"
    for hour in (8, 10, 14, 21):
        assert ned.in_quiet_hours(sched, _at_utc_hour(hour), "UTC") is False, f"{hour}:00 awake"


def test_non_wrapping_quiet_window() -> None:
    """A window that does not cross midnight must still work (01..05)."""
    sched = {"quiet_start": 1, "quiet_end": 5}
    assert ned.in_quiet_hours(sched, _at_utc_hour(3), "UTC") is True
    assert ned.in_quiet_hours(sched, _at_utc_hour(23), "UTC") is False


def test_quiet_hours_are_evaluated_in_the_users_timezone() -> None:
    """23:00 in Amsterdam is 21:00 UTC — quiet there, awake by UTC. Getting this wrong sends
    push notifications in the middle of the night."""
    sched = {"quiet_start": 22, "quiet_end": 8}
    ts = _at_utc_hour(21)
    assert ned.in_quiet_hours(sched, ts, "Europe/Amsterdam") is True
    assert ned.in_quiet_hours(sched, ts, "UTC") is False


def test_equal_quiet_bounds_disable_the_window() -> None:
    assert ned.in_quiet_hours({"quiet_start": 9, "quiet_end": 9}, NOON, "UTC") is False


def test_push_enqueues_one_envelope_per_subscription(tmp_path: Path, monkeypatch) -> None:
    """A user may have an iPhone AND a browser; the id must include the endpoint or the second
    device is silently deduped away."""
    _seed(tmp_path)
    subs = [{"endpoint": "https://apns/1"}, {"endpoint": "https://fcm/2"}]
    sent = _wire(
        monkeypatch,
        items=_items("ep1"),
        comms={"types": {"new_episodes": {"email": False, "push": True}}},
        subs=subs,
    )
    ids = ned.enqueue_for_user(tmp_path / "root", tmp_path, "u1", NOON)
    assert len(ids) == 2
    assert len(set(ids)) == 2, "per-subscription ids must be distinct"
    assert all(e["channel"] == "push" for e in sent)


def test_push_with_no_subscriptions_is_silent_not_an_error(tmp_path: Path, monkeypatch) -> None:
    _seed(tmp_path)
    _wire(
        monkeypatch,
        items=_items("ep1"),
        comms={"types": {"new_episodes": {"email": False, "push": True}}},
        subs=[],
    )
    assert ned.enqueue_for_user(tmp_path / "root", tmp_path, "u1", NOON) == []


def test_both_channels_share_one_dedupe(tmp_path: Path, monkeypatch) -> None:
    """Email and push announce the same episode set in one pass — not two independent ledgers."""
    _seed(tmp_path)
    sent = _wire(
        monkeypatch,
        items=_items("ep1"),
        comms={"types": {"new_episodes": {"email": True, "push": True}}},
        subs=[{"endpoint": "https://apns/1"}],
    )
    ids = ned.enqueue_for_user(tmp_path / "root", tmp_path, "u1", NOON)
    assert len(ids) == 2
    assert {e["channel"] for e in sent} == {"email", "push"}


def test_paused_suppresses_everything(tmp_path: Path, monkeypatch) -> None:
    _seed(tmp_path)
    _wire(
        monkeypatch,
        items=_items("ep1"),
        comms={
            "new_episodes_schedule": {
                "floor_minutes": 240,
                "quiet_start": 22,
                "quiet_end": 8,
                "paused": True,
            }
        },
    )
    assert ned.enqueue_for_user(tmp_path / "root", tmp_path, "u1", NOON) == []


def test_unverified_email_gets_no_email(tmp_path: Path, monkeypatch) -> None:
    _seed(tmp_path)
    _wire(monkeypatch, items=_items("ep1"), user=_U(provider="smoke"))
    assert ned.enqueue_for_user(tmp_path / "root", tmp_path, "u1", NOON) == []


def test_consent_off_on_both_channels_is_silent(tmp_path: Path, monkeypatch) -> None:
    _seed(tmp_path)
    _wire(
        monkeypatch,
        items=_items("ep1"),
        comms={"types": {"new_episodes": {"email": False, "push": False}}},
    )
    assert ned.enqueue_for_user(tmp_path / "root", tmp_path, "u1", NOON) == []


def test_corrupt_ledger_reseeds_rather_than_blasting(tmp_path: Path, monkeypatch) -> None:
    """A corrupt ledger reads as empty, which means first-run — so it must SEED, not alert about
    everything the user has ever not listened to."""
    (tmp_path / "users" / "u1").mkdir(parents=True)
    (tmp_path / "users" / "u1" / "new_episode_alerts.json").write_text("{bad", encoding="utf-8")
    _wire(monkeypatch, items=_items("ep1"))
    assert ned.enqueue_for_user(tmp_path / "root", tmp_path, "u1", NOON) == []
    assert "ep1" in ned.read_ledger(tmp_path, "u1")["announced"]


def test_first_run_seeds_the_backlog_and_sends_nothing(tmp_path: Path, monkeypatch) -> None:
    """THE protection against a backlog blast. A user who has followed shows for months and then
    ticks the box must not be alerted about all of it — that is how notifications get disabled
    permanently. Alerts cover what appears AFTER opting in."""
    _wire(monkeypatch, items=_items("old1", "old2", "old3"))
    assert ned.enqueue_for_user(tmp_path / "root", tmp_path, "u1", NOON) == []
    led = ned.read_ledger(tmp_path, "u1")
    assert set(led["announced"]) == {"old1", "old2", "old3"}
    assert led["last_send_ts"] is None, "seeding must not start the rate-limit clock"


def test_an_episode_appearing_after_the_seed_does_alert(tmp_path: Path, monkeypatch) -> None:
    _wire(monkeypatch, items=_items("old1"))
    ned.enqueue_for_user(tmp_path / "root", tmp_path, "u1", NOON)  # seeds
    _wire(monkeypatch, items=_items("old1", "brand_new"))
    ids = ned.enqueue_for_user(tmp_path / "root", tmp_path, "u1", NOON + 3600)
    assert len(ids) == 1


def test_one_bad_user_does_not_abort_the_roster(tmp_path: Path, monkeypatch) -> None:
    from podcast_scraper.server import app_user_store

    monkeypatch.setattr(app_user_store, "list_users", lambda d: [_U("bad"), _U("good")])
    monkeypatch.setattr(ned, "list_users", lambda d: [_U("bad"), _U("good")])

    def _one(root, data, uid, now=None):
        if uid == "bad":
            raise RuntimeError("corrupt user")
        return ["ok"]

    monkeypatch.setattr(ned, "enqueue_for_user", _one)
    assert ned.enqueue_due_new_episodes(tmp_path / "root", tmp_path, NOON) == ["ok"]


def test_ledger_is_written_atomically(tmp_path: Path, monkeypatch) -> None:
    """A torn ledger would re-announce episodes — the one failure a user would notice."""
    _seed(tmp_path)
    _wire(monkeypatch, items=_items("ep1"))
    ned.enqueue_for_user(tmp_path / "root", tmp_path, "u1", NOON)
    led = json.loads(
        (tmp_path / "users" / "u1" / "new_episode_alerts.json").read_text(encoding="utf-8")
    )
    assert led["announced"]["ep1"] == NOON
    assert led["last_send_ts"] == NOON
    leftovers = list((tmp_path / "users" / "u1").glob(".new_episode_alerts.json.*"))
    assert leftovers == [], f"temp files left behind: {leftovers}"
