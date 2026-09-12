"""Integration tests for the daily post-episode recap digest (RFC-122 #2039).

Builds a tiny real corpus + a signed-in user with finished playback, and exercises the assembler
(0/1/many), the consent gates, the schema-valid envelope, and the typed one-click unsubscribe.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from podcast_scraper.server import (
    app_comms_store,
    app_digest_daily_recap,
    app_outbox_store,
    app_user_state,
)
from podcast_scraper.server.app_slugs import slug_for_row
from podcast_scraper.server.app_user_store import get_or_create_user
from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

pytestmark = [pytest.mark.integration]

_NOW = 1_760_000_000  # a fixed epoch so "today" is deterministic
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEMA = json.loads(
    (_REPO_ROOT / "docs" / "api" / "delivery-envelope.schema.json").read_text(encoding="utf-8")
)


def _write_corpus(root: Path, *, stem: str = "0001-hello") -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {"feed_id": "myfeed", "title": "My Show", "url": "https://pod.example/f.xml"},
        "episode": {"episode_id": "ep1", "title": "Hello", "published_date": "2024-03-10T00:00:00"},
        "summary": {"title": "Sum", "bullets": ["First point", "Second point"]},
        "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    gi = {
        "episode_id": "ep1",
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
        "episode_id": "ep1",
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
    assert ep["deep_link"] == f"/player/{slug}"
    assert {"id": "topic:ai", "label": "AI"} in ep["topics"]


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
    comms = app_comms_store.get_comms(data_dir, uid)  # default hour = 22 UTC
    # 2025-10-09 09:00 UTC is hour 9, not 22.
    nine_utc = 1_760_000_400
    assert app_digest_daily_recap._is_due_slot(comms, nine_utc) is False
    # Bump to the top of hour 22 UTC on the same day.
    twenty_two_utc = nine_utc + (22 - 9) * 3600
    assert app_digest_daily_recap._is_due_slot(comms, twenty_two_utc) is True
