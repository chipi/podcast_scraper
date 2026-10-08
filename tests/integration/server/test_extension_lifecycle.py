"""Extensions own their routes, lifecycle hooks, scheduled jobs, CLI commands and the listener
engagement trending blends in (ADR-158 decision 4).

With none installed the platform serves its own routes only, starts nothing extra, rejects a job
kind nothing runs, offers no extension subcommand and ranks trending on content alone.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from podcast_scraper import extensions
from podcast_scraper.extensions import CliCommand, Extension, use_extensions
from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


def _installed(name: str) -> Extension:
    module = {
        "player": "podcast_scraper.server.app_player_extension",
        "mcp": "podcast_scraper.server.app_mcp_extension",
        "intelligence": "podcast_scraper.enrichment.intelligence_extension",
    }[name]
    ext = extensions._from_module(module)
    assert ext is not None
    return ext


def _paths(app: Any) -> set[str]:
    paths: set[str] = set()
    for route in app.routes:
        if getattr(route, "path", ""):
            paths.add(route.path)
        contexts = getattr(route, "effective_route_contexts", None)
        if callable(contexts):
            for ctx in contexts():
                child = getattr(getattr(ctx, "route", ctx), "path", None)
                if child:
                    paths.add(child)
    return paths


PLAYER_PATHS = {"/api/app/discover", "/api/app/episodes/{slug}", "/api/app/your-week"}
INTELLIGENCE_PATHS = {
    "/api/corpus/topic-clusters",
    "/api/corpus/storylines",
    "/api/corpus/trending",
}


def test_the_platform_alone_serves_no_player_or_intelligence_route(tmp_path: Path) -> None:
    with use_extensions([]):
        paths = _paths(create_app(tmp_path, static_dir=False))
    assert not PLAYER_PATHS & paths
    assert not INTELLIGENCE_PATHS & paths
    # Its own consumer routes stay: sign-in, profile, preferences.
    assert {"/api/app/me", "/api/app/preferences"} <= paths


def test_the_player_and_intelligence_extensions_bring_their_routes(tmp_path: Path) -> None:
    with use_extensions([_installed("player")]):
        assert PLAYER_PATHS <= _paths(create_app(tmp_path, static_dir=False))
    with use_extensions([_installed("intelligence")]):
        assert INTELLIGENCE_PATHS <= _paths(create_app(tmp_path, static_dir=False))


def test_intelligence_routes_reach_the_public_operator_surface_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PODCAST_SERVE_OPERATOR_PUBLIC", "1")
    with use_extensions([_installed("intelligence")]):
        app = create_app(tmp_path, static_dir=False)
    assert INTELLIGENCE_PATHS <= _paths(app)
    assert TestClient(app).get("/api/corpus/trending").status_code == 401


def test_hooks_run_at_build_and_start_and_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "appdata"))
    events: list[str] = []

    def _configured(app: Any) -> None:
        events.append(f"configured data_dir={app.state.app_data_dir is not None}")

    def _started(app: Any):
        events.append("started")
        return lambda: events.append("stopped")

    def _broken(app: Any) -> None:
        raise RuntimeError("boom")

    ext = Extension(
        name="fake", app_configured=(_broken, _configured), server_started=(_broken, _started)
    )
    with use_extensions([ext]):
        app = create_app(tmp_path, static_dir=False)
        assert events == ["configured data_dir=True"]
        with TestClient(app):
            assert events[-1] == "started"
    assert events[-1] == "stopped"


def test_the_mcp_extension_owns_the_internal_verify_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INTERNAL_MCP_TOKEN", "s3cret")
    with use_extensions([]):
        assert not hasattr(create_app(tmp_path, static_dir=False).state, "internal_mcp_token")
    with use_extensions([_installed("mcp")]):
        assert create_app(tmp_path, static_dir=False).state.internal_mcp_token == "s3cret"


def test_a_job_kind_exists_only_while_an_extension_runs_it() -> None:
    from podcast_scraper.server.scheduler import ScheduledJobConfig

    with use_extensions([]):
        with pytest.raises(ValueError, match="kind"):
            ScheduledJobConfig(name="d", cron="0 * * * *", kind="digest")
        assert ScheduledJobConfig(name="p", cron="0 * * * *").kind == "pipeline"
    with use_extensions([_installed("player")]):
        assert ScheduledJobConfig(name="d", cron="0 * * * *", kind="digest").kind == "digest"


def test_a_scheduled_fire_of_an_extension_kind_goes_to_its_runner(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from podcast_scraper.server.scheduler import make_app_spawn_callback

    fired: list[tuple[str, Path, Any]] = []
    app = SimpleNamespace(state=SimpleNamespace())
    ext = Extension(
        name="fake", job_kinds={"digest": lambda n, root, a: fired.append((n, root, a))}
    )
    with use_extensions([ext]):
        make_app_spawn_callback(app)("hourly", tmp_path, tmp_path / "op.yaml", "digest")
    assert fired == [("hourly", tmp_path, app)]

    with use_extensions([]), caplog.at_level(logging.WARNING):
        make_app_spawn_callback(app)("hourly", tmp_path, tmp_path / "op.yaml", "digest")
    assert "nothing installed runs" in caplog.text


def test_the_player_digest_job_enqueues_through_the_dispatcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from podcast_scraper.server import app_digest_dispatch

    seen: list[tuple[Path, Path]] = []

    class _Result:
        total = 0
        errors: dict[str, str] = {}

        def summary(self) -> str:
            return ""

    def _enqueue(corpus_root: Path, data_dir: Path) -> _Result:
        seen.append((corpus_root, data_dir))
        return _Result()

    monkeypatch.setattr(app_digest_dispatch, "enqueue_all_due", _enqueue)
    runner = _installed("player").job_kinds["digest"]
    runner("hourly", tmp_path, SimpleNamespace(state=SimpleNamespace(app_data_dir=None)))
    assert seen == []
    runner("hourly", tmp_path, SimpleNamespace(state=SimpleNamespace(app_data_dir=str(tmp_path))))
    assert seen == [(tmp_path, tmp_path)]


def test_cli_subcommands_dispatch_to_the_extension_that_owns_them() -> None:
    from argparse import Namespace

    from podcast_scraper import cli

    ran: list[Namespace] = []

    def _run(args: Namespace, log: logging.Logger) -> int:
        ran.append(args)
        return 7

    ext = Extension(
        name="fake",
        cli_commands={
            "fake-cmd": CliCommand(
                parse=lambda argv: Namespace(command="fake-cmd", argv=list(argv)), run=_run
            )
        },
    )
    with use_extensions([ext]):
        args = cli.parse_args(["fake-cmd", "--x", "1"])
        assert args.argv == ["--x", "1"]
        assert cli.main(["fake-cmd", "--x", "1"]) == 7
    assert [a.argv for a in ran] == [["--x", "1"]]


def test_the_intelligence_extension_owns_the_topic_clusters_and_mcp_commands() -> None:
    from podcast_scraper import cli

    with use_extensions([_installed("intelligence")]):
        args = cli.parse_args(["topic-clusters", "--output-dir", "/tmp/x"])
        assert args.command == "topic-clusters" and args.output_dir == "/tmp/x"
        assert cli.parse_args(["mcp", "--corpus", "/tmp/x"]).command == "mcp"
    # Without it the word is no command at all: the platform reads it as the feed URL.
    with use_extensions([]):
        with pytest.raises(ValueError, match="RSS URL"):
            cli.parse_args(["topic-clusters", "--output-dir", "/tmp/x"])


def test_trending_blends_engagement_only_from_an_installed_source(tmp_path: Path) -> None:
    from podcast_scraper.server.app_momentum import _engagement_weekly_by_entity

    calls: list[tuple[Path, str | None]] = []

    def _series(data_dir: Path, user_id: str | None) -> dict[str, Any]:
        calls.append((data_dir, user_id))
        return {"entities": [{"kind": "topic", "entity_id": "topic:x", "weekly_counts": {"w": 2}}]}

    with use_extensions([]):
        assert _engagement_weekly_by_entity(tmp_path, None) == {}
    with use_extensions([Extension(name="fake", engagement_series=_series)]):
        assert _engagement_weekly_by_entity(tmp_path, "u1") == {("topic", "topic:x"): {"w": 2}}
        assert _engagement_weekly_by_entity(None, "u1") == {}
    assert calls == [(tmp_path, "u1")]


def test_real_sign_in_providers_come_only_from_an_extension(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from podcast_scraper.server.app_oauth import provider_from_env, providers_from_env

    monkeypatch.setenv("APP_OAUTH_PROVIDER", "google")
    monkeypatch.setenv("APP_OAUTH_PROVIDERS", "google,apple,other")
    monkeypatch.setenv("APP_OAUTH_GOOGLE_CLIENT_ID", "id")
    monkeypatch.setenv("APP_OAUTH_GOOGLE_CLIENT_SECRET", "secret")
    with use_extensions([]), caplog.at_level(logging.WARNING):
        assert provider_from_env() is None
    assert "no installed extension provides it" in caplog.text

    google = SimpleNamespace(name="google")
    apple = SimpleNamespace(name="apple")
    other = SimpleNamespace(name="other")
    fake = Extension(
        name="fake",
        oauth_providers={"google": lambda: google, "apple": lambda: apple, "other": lambda: other},
    )
    with use_extensions([fake]):
        # Only Apple may be added beside the primary, whatever else is installed.
        assert providers_from_env() == {"google": google, "apple": apple}
        # Apple only ever sits beside a real primary, never as the primary itself.
        monkeypatch.setenv("APP_OAUTH_PROVIDER", "apple")
        assert providers_from_env() == {}
    with use_extensions([_installed("mcp")]):
        monkeypatch.setenv("APP_OAUTH_PROVIDER", "google")
        monkeypatch.setenv("APP_OAUTH_PROVIDERS", "")
        assert type(provider_from_env()).__name__ == "GoogleProvider"
