"""Integration tests for opt-in feeds and operator-config API (GitHub #626)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fastapi")

from podcast_scraper.server.app import create_app
from podcast_scraper.server.operator_yaml_profile import split_operator_yaml_profile

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


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
    assert g.json()["feeds"] == []
    assert g.json()["file_relpath"] == "feeds.spec.yaml"

    p = client.put(
        "/api/feeds",
        params={"path": str(corpus)},
        json={
            "feeds": [
                "  https://a.example/feed  ",
                "https://b.example/feed",
                "https://a.example/feed",
            ]
        },
    )
    assert p.status_code == 200
    assert p.json()["feeds"] == ["https://a.example/feed", "https://b.example/feed"]

    g2 = client.get("/api/feeds", params={"path": str(corpus)})
    assert g2.json()["feeds"] == ["https://a.example/feed", "https://b.example/feed"]


def test_operator_config_roundtrip(corpus: Path) -> None:
    app = create_app(corpus, static_dir=False, enable_operator_config_api=True)
    client = TestClient(app)
    h = client.get("/api/health")
    assert h.json().get("operator_config_api") is True

    g = client.get("/api/operator-config", params={"path": str(corpus)})
    assert g.status_code == 200
    body = g.json()
    seeded = body["content"].strip()
    # Seeded from packaged overrides example when present (no ``profile:`` in file).
    if seeded:
        assert "max_episodes" in seeded
        assert split_operator_yaml_profile(seeded)[0] == ""
    # else: no packaged example in this environment — empty GET is fine.
    assert body["corpus_path"] == str(corpus.resolve())
    assert body["operator_config_path"] == str((corpus / "viewer_operator.yaml").resolve())
    assert isinstance(body.get("available_profiles"), list)
    assert isinstance(body["available_profiles"], list)
    assert len(body["available_profiles"]) >= 1

    yaml_text = "output_dir: ./out\nmax_episodes: 3\n"
    p = client.put(
        "/api/operator-config", params={"path": str(corpus)}, json={"content": yaml_text}
    )
    assert p.status_code == 200
    assert p.json()["content"] == yaml_text

    g2 = client.get("/api/operator-config", params={"path": str(corpus)})
    assert g2.json()["content"] == yaml_text
    assert "local" in g2.json()["available_profiles"]


def test_operator_config_get_auto_creates_missing_subdir(corpus: Path) -> None:
    """First-run UX (#693): GET /api/operator-config against a fresh subdir of
    the configured corpus root must auto-create the directory and seed
    ``viewer_operator.yaml`` from the packaged example.

    Regression guard for the failure surfaced by ``operator-first-run.spec.ts``
    on stack-test post-merge: ``resolve_corpus_path_param`` previously rejected
    missing directories with 400 "Not a directory", so the day-1 operator path
    (pick a fresh subdir → viewer makes it work) was unreachable. The route
    now passes ``must_be_dir=False`` and ``mkdir(parents=True, exist_ok=True)``
    before seeding.
    """
    fresh_subdir = corpus / "firstrun-empty"
    assert not fresh_subdir.exists()  # precondition: subdir doesn't exist on disk

    app = create_app(corpus, static_dir=False, enable_operator_config_api=True)
    client = TestClient(app)

    g = client.get("/api/operator-config", params={"path": str(fresh_subdir)})
    assert (
        g.status_code == 200
    ), f"Expected 200 (auto-create + seed), got {g.status_code}: {g.text[:200]}"
    assert fresh_subdir.is_dir(), "GET should have auto-created the corpus subdir"
    body = g.json()
    assert body["corpus_path"] == str(fresh_subdir.resolve())
    # If a packaged example is present in this env, content is non-empty.
    seeded = body["content"].strip()
    if seeded:
        assert "max_episodes" in seeded


def test_operator_config_get_via_symlinked_anchor(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Regression: corpus root reached through a symlink must still validate
    correctly when the requested path is the literal (unresolved) form.

    Bug surfaced by stack-test on main: ``/app/output`` inside the docker
    container is a bind mount whose canonical form differs from the literal.
    Anchor was ``.resolve()``d at app startup but the user-provided path was
    only ``normpath()``d — the ``startswith`` anchor check failed because the
    two used different forms (canonical vs literal). With ``must_be_dir=True``
    the bug was masked (``os.path.isdir`` follows symlinks at FS layer);
    flipping to ``must_be_dir=False`` for #693's first-run UX exposed it.

    Repro: create real dir, symlink to it, point the api at the SYMLINK,
    then request a fresh subdir via the symlink path.
    """
    real = tmp_path_factory.mktemp("real_corpus_root")
    sym = tmp_path_factory.mktemp("sym_parent") / "corpus_via_link"
    sym.symlink_to(real)
    # API is configured with the SYMLINK path; .resolve() inside create_app
    # canonicalises it to ``real``.
    fresh_via_sym = sym / "firstrun-empty"  # literal symlink path; subdir doesn't exist
    assert not fresh_via_sym.exists()

    app = create_app(sym, static_dir=False, enable_operator_config_api=True)
    client = TestClient(app)

    g = client.get("/api/operator-config", params={"path": str(fresh_via_sym)})
    assert (
        g.status_code == 200
    ), f"Expected 200 for path under symlinked anchor; got {g.status_code}: {g.text[:200]}"
    # The actual subdir on disk lives under the resolved anchor.
    assert (real / "firstrun-empty").is_dir()


def test_operator_config_path_outside_anchor_still_rejected(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Auto-create path doesn't loosen the anchor check — paths escaping the
    configured corpus root still 400. Security regression guard for #693 fix.
    """
    corpus_dir = tmp_path_factory.mktemp("corpus_inside")
    outside = tmp_path_factory.mktemp("elsewhere_separate") / "not-allowed"
    app = create_app(corpus_dir, static_dir=False, enable_operator_config_api=True)
    client = TestClient(app)
    g = client.get("/api/operator-config", params={"path": str(outside)})
    assert (
        g.status_code == 400
    ), f"Path outside anchor must be rejected with 400, got {g.status_code}"
    assert not outside.exists(), "auto-create must NOT happen for paths outside anchor"


def test_operator_put_profile_only_merges_packaged_example(corpus: Path) -> None:
    """PUT with only ``profile:`` (empty overrides) picks up packaged example keys."""
    app = create_app(corpus, static_dir=False, enable_operator_config_api=True)
    client = TestClient(app)
    r = client.put(
        "/api/operator-config",
        params={"path": str(corpus)},
        json={"content": "profile: cloud_balanced\n"},
    )
    assert r.status_code == 200
    body = r.json()["content"]
    assert "profile: cloud_balanced" in body
    assert "max_episodes" in body
    g = client.get("/api/operator-config", params={"path": str(corpus)})
    assert g.status_code == 200
    assert "max_episodes" in g.json()["content"]


def test_operator_config_put_rejects_feed_keys(corpus: Path) -> None:
    app = create_app(corpus, static_dir=False, enable_operator_config_api=True)
    client = TestClient(app)
    bad = "rss_urls:\n  - https://x.example/rss\n"
    r = client.put("/api/operator-config", params={"path": str(corpus)}, json={"content": bad})
    assert r.status_code == 400
    detail = r.json().get("detail")
    assert isinstance(detail, dict)
    assert detail.get("error") == "forbidden_operator_feed_keys"


def test_operator_config_put_rejects_api_key(corpus: Path) -> None:
    app = create_app(corpus, static_dir=False, enable_operator_config_api=True)
    client = TestClient(app)
    bad = "openai_api_key: sk-secret\n"
    r = client.put("/api/operator-config", params={"path": str(corpus)}, json={"content": bad})
    assert r.status_code == 400
    detail = r.json().get("detail")
    assert isinstance(detail, dict)
    assert detail.get("error") == "forbidden_operator_keys"


def test_operator_config_get_seeds_whitespace_only_file(corpus: Path) -> None:
    cfg = corpus / "viewer_operator.yaml"
    cfg.write_text("  \n  \n", encoding="utf-8")
    app = create_app(corpus, static_dir=False, enable_operator_config_api=True)
    client = TestClient(app)
    r = client.get("/api/operator-config", params={"path": str(corpus)})
    assert r.status_code == 200
    body = r.json()
    seeded = body["content"].strip()
    if seeded:
        assert "max_episodes" in seeded
        assert split_operator_yaml_profile(seeded)[0] == ""


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


def test_feeds_put_accepts_object_entries_roundtrip(corpus: Path) -> None:
    app = create_app(corpus, static_dir=False, enable_feeds_api=True)
    client = TestClient(app)
    payload = {
        "feeds": [
            {"url": "https://obj.example/a", "timeout": 30, "rss_conditional_get": True},
            "https://plain.example/b",
        ]
    }
    p = client.put("/api/feeds", params={"path": str(corpus)}, json=payload)
    assert p.status_code == 200
    feeds = p.json()["feeds"]
    assert any(
        isinstance(x, dict) and x.get("url") == "https://obj.example/a" and x.get("timeout") == 30
        for x in feeds
    )
    g = client.get("/api/feeds", params={"path": str(corpus)})
    assert g.status_code == 200
    assert len(g.json()["feeds"]) >= 1


def test_feeds_get_500_on_invalid_spec_on_disk(corpus: Path) -> None:
    """Unknown top-level keys raise ``ValueError`` in ``load_feeds_spec_file`` → HTTP 500."""
    (corpus / "feeds.spec.yaml").write_text(
        "not_allowed_root: true\nfeeds: []\n",
        encoding="utf-8",
    )
    app = create_app(corpus, static_dir=False, enable_feeds_api=True)
    client = TestClient(app)
    r = client.get("/api/feeds", params={"path": str(corpus)})
    assert r.status_code == 500


def test_feeds_put_rejects_non_string_non_object_entries(corpus: Path) -> None:
    app = create_app(corpus, static_dir=False, enable_feeds_api=True)
    client = TestClient(app)
    r = client.put("/api/feeds", params={"path": str(corpus)}, json={"feeds": [1, 2, 3]})
    assert r.status_code in (400, 422)


def test_feeds_put_rejects_over_max_urls(corpus: Path) -> None:
    app = create_app(corpus, static_dir=False, enable_feeds_api=True)
    client = TestClient(app)
    feeds = [f"https://e{i}.example/feed" for i in range(5001)]
    r = client.put("/api/feeds", params={"path": str(corpus)}, json={"feeds": feeds})
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
