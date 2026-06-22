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
        # slice 2 — relational (RFC-094 traversals)
        "person_positions",
        "who_said_about_topic",
        "cross_show_synthesis",
        "insights_about_entity",
        "topic_entities",
        "related_insights",
        "show_episodes",
        # slice 3 — CIL intelligence
        "person_profile",
        "topic_timeline",
        "position_arc",
        # connectivity / neighborhood (#1054)
        "entity_neighborhood",
        "person_topics",
        "co_occurring_entities",
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
