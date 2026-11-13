"""The command line reaches stage-owned subcommands and batch-end refreshes through cli_hooks
(E04 S7), kept out of the registry corpus migrations reach."""

from __future__ import annotations

from typing import Any, List
from unittest.mock import patch

import pytest

from podcast_scraper import builtin_cli_hooks, cli_hooks

pytestmark = pytest.mark.unit


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


def test_a_command_no_one_provides_is_named_not_read_as_a_feed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without the extension that owns it, ``identity-ledger`` fails as a missing command. Read as
    a feed, its flags failed as the pipeline's ("--corpus" matched --corpus-media-link-mode)."""
    from podcast_scraper import cli
    from podcast_scraper.extensions import use_extensions

    with patch.object(builtin_cli_hooks, "CLI_COMMANDS", {}), use_extensions([]):
        with pytest.raises(SystemExit) as exc:
            cli.parse_args(["identity-ledger", "--corpus", "/c"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "no command named 'identity-ledger'" in err
    assert "RSS URL" not in err


@pytest.mark.parametrize(
    "argv",
    [
        ["https://example.com/feed.xml", "--dry-run"],
        ["doctor"],
    ],
)
def test_feeds_and_built_in_commands_never_reach_the_missing_command_message(
    argv: List[str], capsys: pytest.CaptureFixture[str]
) -> None:
    from podcast_scraper import cli
    from podcast_scraper.extensions import use_extensions

    with use_extensions([]):
        try:
            cli.parse_args(argv)
        except SystemExit:
            pass
    assert "no command named" not in capsys.readouterr().err
