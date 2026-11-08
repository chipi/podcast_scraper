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
