"""End-to-end digest assembly over a real synthetic corpus (#1413).

Exercises the full chain — auto-picks (#1416, real GI resolution), new_in_follows, and
trending_in_your_corpus (#1413 2b) — through the actual `resolve_slug` / `insights_from_gi` /
`entities_from_kg` / temporal_velocity readers, not monkeypatched. Also validates the assembled
payload's envelope against the committed contract schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from podcast_scraper.server import app_comms_store, app_digest_personal, app_user_state
from podcast_scraper.server.app_slugs import episode_slug
from podcast_scraper.server.app_user_store import get_or_create_user

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEMA = json.loads(
    (_REPO_ROOT / "docs" / "api" / "delivery-envelope.schema.json").read_text(encoding="utf-8")
)


def _write_ep(
    root: Path,
    *,
    stem: str,
    feed_id: str,
    episode_id: str,
    topics: list[tuple[str, str]],
    gi_text: str | None = None,
    published: str = "2024-03-10T00:00:00",
) -> str:
    """Write a corpus episode (metadata + KG [+ GI]); return its slug."""
    meta_dir = root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    rel = f"metadata/{stem}.metadata.json"
    doc = {
        "feed": {"feed_id": feed_id, "title": f"Show {feed_id}", "url": "https://p/f.xml"},
        "episode": {
            "episode_id": episode_id,
            "title": f"Episode {episode_id}",
            "published_date": published,
            "duration_seconds": 1000,
        },
        "summary": {"title": "S", "bullets": ["a"]},
        "content": {
            "transcript_file_path": f"transcripts/{stem}.txt",
            "media_url": "https://cdn.example/a.mp3",
            "media_type": "audio/mpeg",
        },
    }
    (meta_dir / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "transcripts" / f"{stem}.txt").write_text("hello world", encoding="utf-8")
    nodes = [{"id": tid, "type": "Topic", "properties": {"label": la}} for tid, la in topics]
    (meta_dir / f"{stem}.kg.json").write_text(
        json.dumps({"episode_id": episode_id, "nodes": nodes}), encoding="utf-8"
    )
    if gi_text is not None:
        gi = {
            "episode_id": episode_id,
            "nodes": [
                {
                    "id": "i1",
                    "type": "Insight",
                    "properties": {"text": gi_text, "grounded": True, "salience": 1.0},
                },
                {
                    "id": "q1",
                    "type": "Quote",
                    "properties": {"text": "quote", "timestamp_start_ms": 60000},
                },
            ],
            "edges": [{"type": "SUPPORTED_BY", "from": "i1", "to": "q1"}],
        }
        (meta_dir / f"{stem}.gi.json").write_text(json.dumps(gi), encoding="utf-8")
    return episode_slug(feed_id, episode_id, rel)


def _mark_heard(data_dir: Path, uid: str, slug: str) -> None:
    app_user_state.set_playback(data_dir, uid, slug, 500.0, 1)  # 500/1000s ≥ 30% → heard


def _sections(payload: dict) -> dict[str, list]:
    return {s["kind"]: s["items"] for s in payload["sections"]}


def test_auto_pick_real_gi_chain(tmp_path: Path) -> None:
    root, data_dir = tmp_path / "corpus", tmp_path / "app"
    uid = get_or_create_user(
        data_dir, provider="google", subject="s", email="u@gmail.com", name="U"
    ).user_id
    slug = _write_ep(
        root,
        stem="0001",
        feed_id="fa",
        episode_id="e1",
        topics=[("topic:ai", "AI")],
        gi_text="a grounded point",
    )
    _mark_heard(data_dir, uid, slug)  # heard, not captured → auto-pick candidate

    payload = app_digest_personal.assemble_digest_payload(root, data_dir, uid, now=10**9)
    assert payload is not None
    revisit = _sections(payload)["revisit"]
    assert revisit[0]["source"] == "auto"
    assert revisit[0]["quote"] == "a grounded point"
    assert revisit[0]["graph_refs"] == [{"id": "topic:ai", "kind": "topic", "label": "AI"}]
    assert revisit[0]["deep_link"] == f"/player/{slug}?t=60"


def test_new_in_follows_section(tmp_path: Path) -> None:
    root, data_dir = tmp_path / "corpus", tmp_path / "app"
    uid = get_or_create_user(
        data_dir, provider="google", subject="s", email="u@gmail.com", name="U"
    ).user_id
    slug = _write_ep(root, stem="0001", feed_id="fa", episode_id="e1", topics=[("topic:ai", "AI")])
    app_user_state.add_subscription(
        data_dir, uid, {"feed_id": "fa"}
    )  # follow the show; episode unheard

    payload = app_digest_personal.assemble_digest_payload(root, data_dir, uid, now=10**9)
    assert payload is not None
    nif = _sections(payload)["new_in_follows"]
    assert nif[0]["episode_slug"] == slug
    assert nif[0]["graph_refs"] == [{"id": "topic:ai", "kind": "topic", "label": "AI"}]
    assert "source" not in nif[0]  # not a user/auto capture


def test_trending_section(tmp_path: Path) -> None:
    root, data_dir = tmp_path / "corpus", tmp_path / "app"
    uid = get_or_create_user(
        data_dir, provider="google", subject="s", email="u@gmail.com", name="U"
    ).user_id
    slug = _write_ep(root, stem="0001", feed_id="fa", episode_id="e1", topics=[("topic:ai", "AI")])
    _mark_heard(data_dir, uid, slug)  # topic:ai is in the user's corpus
    (root / "enrichments").mkdir(parents=True, exist_ok=True)
    (root / "enrichments" / "temporal_velocity.json").write_text(
        json.dumps(
            {"topics": [{"topic_id": "topic:ai", "velocity_last_over_6mo": 3.0, "total": 10}]}
        ),
        encoding="utf-8",
    )
    payload = app_digest_personal.assemble_digest_payload(root, data_dir, uid, now=10**9)
    assert payload is not None
    trend = _sections(payload)["trending_in_your_corpus"]
    assert trend[0]["graph_refs"] == [{"id": "topic:ai", "kind": "topic", "label": "AI"}]
    assert trend[0]["deep_link"] == "/topic/ai?scope=mine"


def test_full_envelope_matches_schema(tmp_path: Path) -> None:
    root, data_dir = tmp_path / "corpus", tmp_path / "app"
    user = get_or_create_user(
        data_dir, provider="google", subject="s", email="u@gmail.com", name="U"
    )
    slug = _write_ep(
        root, stem="0001", feed_id="fa", episode_id="e1", topics=[("topic:ai", "AI")], gi_text="pt"
    )
    _mark_heard(data_dir, user.user_id, slug)
    comms = app_comms_store.set_comms(data_dir, user.user_id, digest={"enabled": True})
    payload = app_digest_personal.assemble_digest_payload(root, data_dir, user.user_id, now=10**9)
    assert payload is not None
    env = app_digest_personal.build_email_envelope(user, comms, payload, now=10**9)
    errors = sorted(Draft202012Validator(_SCHEMA).iter_errors(env), key=str)
    assert not errors, "\n".join(f"{list(e.path)}: {e.message}" for e in errors)
