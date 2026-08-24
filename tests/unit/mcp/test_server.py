"""MCP registration smoke test (RFC-095). The MCP SDK ships in ``[dev]`` (always present)."""

from __future__ import annotations

import pytest

from podcast_scraper.mcp.cli_handlers import parse_mcp_argv
from podcast_scraper.mcp.server import build_server

pytestmark = pytest.mark.unit


def test_build_server_registers_tools(tmp_path) -> None:
    server = build_server(tmp_path)
    names = {tool.name for tool in server._tool_manager.list_tools()}
    assert names == {
        # slice 1
        "resolve_entity",
        "search_corpus",
        # RFC-093 LITM briefing pack (added 2026-06-25)
        "corpus_briefing_pack",
        # cross-surface refresh: momentum (RFC-103)
        "corpus_trending",
        # slice 2 — relational (RFC-094 traversals)
        "person_positions",
        "who_said_about_topic",
        "cross_show_synthesis",
        "insights_about_entity",
        "topic_entities",
        "related_insights",
        "show_episodes",
        # cross-surface refresh: pivot bridge
        "insight_detail",
        # slice 3 — CIL intelligence
        "person_profile",
        "topic_timeline",
        "position_arc",
        # cross-surface refresh: temporal + centrality
        "topic_conversation_arc",
        "topic_perspective_leaders",
        # cross-surface refresh: GI / grounded-insight
        "explore_insights",
        "episode_insights",
        "compare_subjects",
        # connectivity / neighborhood (#1054)
        "entity_neighborhood",
        "person_topics",
        "co_occurring_entities",
        "bridge",
        "related_topics",
        # cross-surface refresh: multi-hop + clusters
        "ego_network",
        "topic_clusters",
        # cross-surface refresh: search result-set operators
        "cluster_search",
        "consensus_search",
        # cross-surface refresh: enrichment + speaker
        "corpus_enrichment_signals",
        "episode_enrichment_signals",
        "episode_speaker_roster",
        # RFC-118: corpus derivation freshness + the re-derive levers
        "corpus_status",
        "reenrich",
        "reindex",
        # cross-surface refresh: composite dossiers
        "entity_dossier",
        "episode_digest",
        # slice 3 — catalog / navigation
        "list_feeds",
        "list_episodes",
        "episode_detail",
        "top_people",
    }


def test_registered_tools_have_descriptions(tmp_path) -> None:
    server = build_server(tmp_path)
    by_name = {t.name: t for t in server._tool_manager.list_tools()}
    # Agent-facing descriptions must be non-empty (the resolve-first guidance, etc.).
    assert by_name["resolve_entity"].description
    assert by_name["search_corpus"].description


def test_parse_mcp_argv_requires_corpus() -> None:
    args = parse_mcp_argv(["--corpus", "/some/dir"])
    assert args.command == "mcp"
    assert args.corpus == "/some/dir"
    with pytest.raises(SystemExit):
        parse_mcp_argv([])


# --- RFC-112 slice 2: transport selection + auth-wrapped HTTP app ---


def test_parse_mcp_argv_transport_defaults_stdio() -> None:
    args = parse_mcp_argv(["--corpus", "/d"])
    assert args.transport == "stdio"


def test_parse_mcp_argv_http_flags() -> None:
    # A test string asserting CLI parse — no actual bind (the container binds 0.0.0.0 by design,
    # loopback-published on the host). nosec per the repo convention for legitimate 0.0.0.0.
    args = parse_mcp_argv(
        ["--corpus", "/d", "--transport", "http", "--host", "0.0.0.0", "--port", "9"]  # nosec B104
    )
    assert args.transport == "http" and args.host == "0.0.0.0" and args.port == 9  # nosec B104


def test_run_server_rejects_unknown_transport(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import pytest

    from podcast_scraper.mcp.server import run_server

    with pytest.raises(ValueError):
        run_server(tmp_path, transport="carrier-pigeon")


def test_run_server_stdio_dispatches(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from podcast_scraper.mcp import server as srv

    called: dict = {}
    monkeypatch.setattr(srv, "run_stdio", lambda c: called.setdefault("corpus", c))
    srv.run_server(tmp_path, transport="stdio")
    assert called["corpus"] == tmp_path


def test_build_http_app_is_auth_wrapped(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from podcast_scraper.mcp.auth import McpAuthMiddleware
    from podcast_scraper.mcp.server import build_http_app

    app = build_http_app(tmp_path)
    assert isinstance(app, McpAuthMiddleware)  # the HTTP transport is gated


def test_transport_security_admits_public_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: behind the edge the forwarded Host is the public name, not loopback.

    FastMCP auto-enables DNS-rebind protection with loopback-only ``allowed_hosts`` when the
    default ``host`` is 127.0.0.1, so the deployed server 421'd every request under
    ``Host: mcp.closelistening.app``. ``_transport_security`` must derive the public host from
    ``APP_MCP_RESOURCE_URL`` and admit it while still rejecting arbitrary hosts.
    """
    from mcp.server.transport_security import TransportSecurityMiddleware

    from podcast_scraper.mcp.server import _transport_security

    monkeypatch.setenv("APP_MCP_RESOURCE_URL", "https://mcp.closelistening.app")
    monkeypatch.delenv("APP_MCP_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("APP_MCP_ALLOWED_ORIGINS", raising=False)

    mw = TransportSecurityMiddleware(_transport_security())
    assert mw._validate_host("mcp.closelistening.app") is True  # the public edge host
    assert mw._validate_host("127.0.0.1:8000") is True  # container healthcheck still works
    assert mw._validate_host("evil.example.com") is False  # protection intact


def test_transport_security_host_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``APP_MCP_ALLOWED_HOSTS`` adds hosts even when no resource URL is set."""
    from mcp.server.transport_security import TransportSecurityMiddleware

    from podcast_scraper.mcp.server import _transport_security

    monkeypatch.delenv("APP_MCP_RESOURCE_URL", raising=False)
    monkeypatch.setenv("APP_MCP_ALLOWED_HOSTS", "mcp.example.test, alt.example.test")
    monkeypatch.delenv("APP_MCP_ALLOWED_ORIGINS", raising=False)

    mw = TransportSecurityMiddleware(_transport_security())
    assert mw._validate_host("mcp.example.test") is True
    assert mw._validate_host("alt.example.test") is True
    assert mw._validate_host("127.0.0.1:8000") is True
