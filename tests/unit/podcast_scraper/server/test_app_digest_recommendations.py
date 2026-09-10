"""Unit tests for the monthly recommendations digest assembler (wave-H)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from podcast_scraper.server import (
    app_comms_store,
    app_digest_recommendations,
    app_digest_sections,
    app_outbox_store,
)
from podcast_scraper.server.app_user_store import get_or_create_user, get_user

pytestmark = pytest.mark.unit

_ROOT = Path("/unused")
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCHEMA = json.loads(
    (_REPO_ROOT / "docs" / "api" / "delivery-envelope.schema.json").read_text(encoding="utf-8")
)

_TRENDING = [
    {
        "episode_slug": "ep-ai",
        "graph_refs": [{"id": "topic:ai", "kind": "topic", "label": "AI"}],
        "deep_link": "/topic/ai?scope=mine",
    }
]
_INTERESTS = [
    {
        "episode_slug": "ep-chips",
        "episode_title": "Chips",
        "graph_refs": [{"id": "topic:chips", "kind": "topic", "label": "Chips"}],
        "deep_link": "/player/ep-chips",
    }
]


def _google_user(tmp_path: Path) -> str:
    u = get_or_create_user(tmp_path, provider="google", subject="s1", email="u@gmail.com", name="U")
    return u.user_id


def _stub_sections(monkeypatch, *, trending=_TRENDING, interests=_INTERESTS) -> None:
    monkeypatch.setattr(app_digest_sections, "trending_items", lambda *a, **k: list(trending))
    monkeypatch.setattr(
        app_digest_sections, "new_in_interests_items", lambda *a, **k: list(interests)
    )


def test_payload_carries_discovery_sections(tmp_path: Path, monkeypatch) -> None:
    _stub_sections(monkeypatch)
    payload = app_digest_recommendations.assemble_recommendations_payload(
        _ROOT, tmp_path, "u_x", now=10**9
    )
    assert payload is not None
    kinds = [s["kind"] for s in payload["sections"]]
    assert kinds == ["trending_in_your_corpus", "new_in_interests"]


def test_payload_none_when_no_discovery(tmp_path: Path, monkeypatch) -> None:
    _stub_sections(monkeypatch, trending=[], interests=[])
    assert (
        app_digest_recommendations.assemble_recommendations_payload(
            _ROOT, tmp_path, "u_x", now=10**9
        )
        is None
    )


def test_built_envelope_matches_contract_schema(tmp_path: Path, monkeypatch) -> None:
    _stub_sections(monkeypatch)
    uid = _google_user(tmp_path)
    comms = app_comms_store.set_comms(tmp_path, uid, types={"digest": {"email": True}})
    payload = app_digest_recommendations.assemble_recommendations_payload(
        _ROOT, tmp_path, uid, now=10**9
    )
    user = get_user(tmp_path, uid)
    assert user is not None and payload is not None
    env = app_digest_recommendations.build_recommendations_envelope(user, comms, payload, now=10**9)
    errors = sorted(Draft202012Validator(_SCHEMA).iter_errors(env), key=str)
    assert not errors, "\n".join(f"{list(e.path)}: {e.message}" for e in errors)
    assert env["template"] == "recommendations-digest.v1"
    assert env["consent_snapshot"]["cadence"] == "monthly"
    assert env["id"].startswith("rec_")


def test_enqueue_gated_on_digest_email_consent(tmp_path: Path, monkeypatch) -> None:
    _stub_sections(monkeypatch)
    uid = _google_user(tmp_path)
    # digest email off → nothing enqueued
    assert (
        app_digest_recommendations.enqueue_recommendations_for_user(_ROOT, tmp_path, uid, now=10**9)
        is None
    )
    app_comms_store.set_comms(tmp_path, uid, types={"digest": {"email": True}})
    eid = app_digest_recommendations.enqueue_recommendations_for_user(
        _ROOT, tmp_path, uid, now=10**9
    )
    assert eid is not None and eid.startswith("rec_")
    pending = app_outbox_store.list_pending(tmp_path, channel="email", now=10**9)
    assert eid in [e["id"] for e in pending]


def test_enqueue_skips_when_paused(tmp_path: Path, monkeypatch) -> None:
    _stub_sections(monkeypatch)
    uid = _google_user(tmp_path)
    app_comms_store.set_comms(
        tmp_path, uid, types={"digest": {"email": True}}, digest_schedule={"paused": True}
    )
    assert (
        app_digest_recommendations.enqueue_recommendations_for_user(_ROOT, tmp_path, uid, now=10**9)
        is None
    )


def test_enqueue_due_only_fires_on_the_monthly_slot(tmp_path: Path, monkeypatch) -> None:
    # The scheduler calls this hourly; it must enqueue only on the 1st at the user's hour.
    import datetime as dt

    _stub_sections(monkeypatch)
    uid = _google_user(tmp_path)
    app_comms_store.set_comms(tmp_path, uid, types={"digest": {"email": True}})  # hour defaults 13
    not_slot = int(dt.datetime(2026, 8, 2, 13, 0, tzinfo=dt.timezone.utc).timestamp())
    slot = int(dt.datetime(2026, 8, 1, 13, 0, tzinfo=dt.timezone.utc).timestamp())
    assert app_digest_recommendations.enqueue_due_recommendations(_ROOT, tmp_path, not_slot) == []
    fired = app_digest_recommendations.enqueue_due_recommendations(_ROOT, tmp_path, slot)
    assert len(fired) == 1 and fired[0].startswith("rec_")


def test_is_monthly_slot(tmp_path: Path) -> None:
    import datetime as dt

    comms = {"digest_schedule": {"hour": 13}}
    first_1300 = int(dt.datetime(2026, 8, 1, 13, 0, tzinfo=dt.timezone.utc).timestamp())
    second_1300 = int(dt.datetime(2026, 8, 2, 13, 0, tzinfo=dt.timezone.utc).timestamp())
    first_0900 = int(dt.datetime(2026, 8, 1, 9, 0, tzinfo=dt.timezone.utc).timestamp())
    assert app_digest_recommendations.is_monthly_slot(comms, first_1300) is True
    assert app_digest_recommendations.is_monthly_slot(comms, second_1300) is False  # not the 1st
    assert app_digest_recommendations.is_monthly_slot(comms, first_0900) is False  # wrong hour
