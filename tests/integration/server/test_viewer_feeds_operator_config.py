"""Integration tests for opt-in feeds and operator-config API (GitHub #626)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fastapi")

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "ep1.gi.json").write_text(
        json.dumps({"grounded_insights": {"version": "1.0"}}), encoding="utf-8"
    )
    return tmp_path


def test_feeds_roundtrip(corpus: Path) -> None:
    app = create_app(corpus, static_dir=False, enable_feeds_api=True)
    client = TestClient(app)
    h = client.get("/api/health")
    assert h.status_code == 200
    assert h.json().get("feeds_api") is True
    assert h.json().get("operator_config_api") is False

    g = client.get("/api/feeds", params={"path": str(corpus)})
    assert g.status_code == 200
    assert g.json()["urls"] == []
    assert g.json()["file_relpath"] == "rss_urls.list.txt"

    p = client.put(
        "/api/feeds",
        params={"path": str(corpus)},
        json={
            "urls": [
                "  https://a.example/feed  ",
                "https://b.example/feed",
                "https://a.example/feed",
            ]
        },
    )
    assert p.status_code == 200
    assert p.json()["urls"] == ["https://a.example/feed", "https://b.example/feed"]

    g2 = client.get("/api/feeds", params={"path": str(corpus)})
    assert g2.json()["urls"] == ["https://a.example/feed", "https://b.example/feed"]


def test_operator_config_roundtrip(corpus: Path) -> None:
    app = create_app(corpus, static_dir=False, enable_operator_config_api=True)
    client = TestClient(app)
    h = client.get("/api/health")
    assert h.json().get("operator_config_api") is True

    g = client.get("/api/operator-config", params={"path": str(corpus)})
    assert g.status_code == 200
    body = g.json()
    assert body["content"] == ""
    assert body["corpus_path"] == str(corpus.resolve())
    assert body["operator_config_path"] == str((corpus / "viewer_operator.yaml").resolve())

    yaml_text = "output_dir: ./out\nrss_urls:\n  - https://x.example/rss\n"
    p = client.put(
        "/api/operator-config", params={"path": str(corpus)}, json={"content": yaml_text}
    )
    assert p.status_code == 200
    assert p.json()["content"] == yaml_text

    g2 = client.get("/api/operator-config", params={"path": str(corpus)})
    assert g2.json()["content"] == yaml_text


def test_operator_config_put_rejects_api_key(corpus: Path) -> None:
    app = create_app(corpus, static_dir=False, enable_operator_config_api=True)
    client = TestClient(app)
    bad = "openai_api_key: sk-secret\n"
    r = client.put("/api/operator-config", params={"path": str(corpus)}, json={"content": bad})
    assert r.status_code == 400
    detail = r.json().get("detail")
    assert isinstance(detail, dict)
    assert detail.get("error") == "forbidden_operator_keys"


def test_operator_config_get_conflict_when_file_has_secrets(corpus: Path) -> None:
    cfg = corpus / "viewer_operator.yaml"
    cfg.write_text("gemini_api_key: x\n", encoding="utf-8")
    app = create_app(corpus, static_dir=False, enable_operator_config_api=True)
    client = TestClient(app)
    r = client.get("/api/operator-config", params={"path": str(corpus)})
    assert r.status_code == 409


def test_feeds_not_mounted_by_default(corpus: Path) -> None:
    app = create_app(corpus, static_dir=False)
    client = TestClient(app)
    assert client.get("/api/feeds", params={"path": str(corpus)}).status_code == 404


def test_feeds_rejects_path_outside_anchor(corpus: Path) -> None:
    outside = corpus.parent / "outside_corpus_feeds"
    outside.mkdir(exist_ok=True)
    app = create_app(corpus, static_dir=False, enable_feeds_api=True)
    client = TestClient(app)
    r = client.get("/api/feeds", params={"path": str(outside)})
    assert r.status_code == 400


def test_feeds_put_rejects_over_max_urls(corpus: Path) -> None:
    app = create_app(corpus, static_dir=False, enable_feeds_api=True)
    client = TestClient(app)
    urls = [f"https://e{i}.example/feed" for i in range(5001)]
    r = client.put("/api/feeds", params={"path": str(corpus)}, json={"urls": urls})
    assert r.status_code == 400


def test_operator_put_rejects_invalid_yaml(corpus: Path) -> None:
    app = create_app(corpus, static_dir=False, enable_operator_config_api=True)
    client = TestClient(app)
    r = client.put(
        "/api/operator-config",
        params={"path": str(corpus)},
        json={"content": "{\n  not_closed: true"},
    )
    assert r.status_code == 422


def test_operator_put_rejects_non_mapping_yaml_root(corpus: Path) -> None:
    app = create_app(corpus, static_dir=False, enable_operator_config_api=True)
    client = TestClient(app)
    r = client.put(
        "/api/operator-config",
        params={"path": str(corpus)},
        json={"content": "- item1\n- item2\n"},
    )
    assert r.status_code == 422


def test_both_apis_enabled_exposed_in_health(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "ep1.gi.json").write_text(
        json.dumps({"grounded_insights": {"version": "1.0"}}), encoding="utf-8"
    )
    app = create_app(
        tmp_path,
        static_dir=False,
        enable_feeds_api=True,
        enable_operator_config_api=True,
    )
    client = TestClient(app)
    h = client.get("/api/health").json()
    assert h.get("feeds_api") is True
    assert h.get("operator_config_api") is True


def test_operator_config_explicit_file_path(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "metadata").mkdir()
    (corpus / "metadata" / "ep1.gi.json").write_text(
        json.dumps({"grounded_insights": {"version": "1.0"}}),
        encoding="utf-8",
    )
    cfg = tmp_path / "custom_operator.yaml"
    cfg.write_text("batch_size: 2\n", encoding="utf-8")
    app = create_app(
        corpus,
        static_dir=False,
        enable_operator_config_api=True,
        operator_config_file=str(cfg),
    )
    client = TestClient(app)
    r = client.get("/api/operator-config", params={"path": str(corpus)})
    assert r.status_code == 200
    body = r.json()
    assert body["operator_config_path"] == str(cfg.resolve())
    assert body["content"] == "batch_size: 2\n"
