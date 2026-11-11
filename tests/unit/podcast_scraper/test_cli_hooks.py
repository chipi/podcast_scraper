"""The command line reaches stage-owned subcommands and batch-end refreshes through cli_hooks
(E04 S7), kept out of the registry corpus migrations reach."""

from __future__ import annotations

from typing import Any, List
from unittest.mock import patch

import pytest

from podcast_scraper import builtin_cli_hooks, cli, cli_hooks

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("name", ["gi", "gi-repair", "identity-ledger"])
def test_each_stage_subcommand_is_owned_by_the_built_in_stage(name: str) -> None:
    assert cli_hooks.cli_command(name) is builtin_cli_hooks.CLI_COMMANDS[name]


def test_the_cli_parses_gi_through_its_owner() -> None:
    args = cli.parse_args(["gi", "validate", "/tmp/sample.gi.json", "--strict"])
    assert args.command == "gi"


def test_without_the_stage_the_subcommand_is_not_the_clis() -> None:
    with patch.object(builtin_cli_hooks, "CLI_COMMANDS", {}):
        assert cli_hooks.cli_command("gi") is None


def test_the_batch_refresh_runs_the_built_in_then_each_extension() -> None:
    from podcast_scraper.extensions import Extension, use_extensions
    from podcast_scraper.stage_hooks import PipelineHooks

    calls: List[Any] = []
    ext = Extension(
        name="x",
        pipeline_hooks=PipelineHooks(
            name="x", batch_completed=(lambda p, c, log: calls.append(("ext", p)),)
        ),
    )
    with (
        patch.object(
            builtin_cli_hooks,
            "BATCH_COMPLETED",
            (lambda p, c, log: calls.append(("builtin", p)),),
        ),
        use_extensions([ext]),
    ):
        cli_hooks.batch_completed("/corpus", "cfg", None)
    assert calls == [("builtin", "/corpus"), ("ext", "/corpus")]


def test_the_built_in_refreshes_are_the_guarded_ones() -> None:
    with (
        patch(
            "podcast_scraper.identity.refresh_guard.refresh_ledger_after", return_value=True
        ) as after,
        patch("podcast_scraper.identity.refresh_guard.refresh_ledger_after_batch") as batch,
    ):
        assert cli_hooks.ledger_refresh("/c", after="enrichment") is True
        builtin_cli_hooks._refresh_ledger_after_batch("/c", "cfg", None)
    after.assert_called_once_with("/c", after="enrichment", log=None)
    batch.assert_called_once_with("/c", "cfg", None)


def test_without_a_ledger_stage_the_refresh_has_nothing_to_do() -> None:
    with patch.object(builtin_cli_hooks, "LEDGER_REFRESH", None):
        assert cli_hooks.ledger_refresh("/c", after="enrichment") is True


def test_an_extension_owns_the_refresh_once_the_built_in_is_gone() -> None:
    """The ledger moves to Common; its refresh then comes from the extension's pipeline hooks."""
    from podcast_scraper.extensions import Extension, use_extensions
    from podcast_scraper.stage_hooks import PipelineHooks

    calls: List[Any] = []

    def refresh(root: Any, *, after: str, log: Any = None) -> bool:
        calls.append((root, after))
        return False

    ext = Extension(name="x", pipeline_hooks=PipelineHooks(name="x", ledger_refresh=refresh))
    with patch.object(builtin_cli_hooks, "LEDGER_REFRESH", None), use_extensions([ext]):
        assert cli_hooks.ledger_refresh("/c", after="upgrade run") is False
    assert calls == [("/c", "upgrade run")]


def test_upgrade_run_refreshes_through_the_seam() -> None:
    import inspect

    from podcast_scraper.upgrade import cli_handlers

    text = inspect.getsource(cli_handlers)
    assert "cli_hooks.ledger_refresh(" in text and "refresh_guard" not in text


@pytest.mark.parametrize(
    ("name", "module", "parse_fn", "run_fn"),
    [
        ("gi", "podcast_scraper.insight_stage.cli", "_parse_gi_args", "_run_gi"),
        (
            "gi-repair",
            "podcast_scraper.insight_stage.cli",
            "_parse_gi_repair_argv",
            "_run_gi_repair_cli",
        ),
        (
            "identity-ledger",
            "podcast_scraper.identity.ledger_cli",
            "parse_identity_ledger_argv",
            "run_identity_ledger_cli",
        ),
        (
            "enrich-edges",
            "podcast_scraper.insight_stage.edges_cli",
            "parse_enrich_edges_argv",
            "run_enrich_edges_cli",
        ),
        (
            "cluster-corpus-topics",
            "podcast_scraper.kg.topic_clustering_cli",
            "parse_cluster_corpus_topics_argv",
            "run_cluster_corpus_topics_cli",
        ),
        (
            "insight-clusters",
            "podcast_scraper.insight_stage.cluster_cli",
            "parse_insight_clusters_argv",
            "run_insight_clusters_cli",
        ),
    ],
)
def test_each_stage_verb_parses_and_runs_in_its_stage(
    name: str, module: str, parse_fn: str, run_fn: str
) -> None:
    command = cli_hooks.cli_command(name)
    assert command is not None
    with (
        patch(f"{module}.{parse_fn}", return_value="parsed") as parse,
        patch(f"{module}.{run_fn}", return_value=0) as run,
    ):
        assert command.parse(["--x"]) == "parsed"
        assert command.run("args", None) == 0
    assert parse.call_args.args[0] == ["--x"]
    assert run.called
