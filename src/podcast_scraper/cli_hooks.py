"""Where the command line reaches a stage it does not own (E04 S7).

Kept apart from :mod:`podcast_scraper.stage_hooks` on purpose: the pipeline and the corpus
migrations reach every stage through that registry, and a migration's measured code (the outcome
hook's import closure) is everything it can reach. A subcommand or a batch-end ledger refresh is
not something a migration runs, so it lives here, reached only from ``cli``, ``service`` and the
enrichment CLI.

Built-ins first (:mod:`.builtin_cli_hooks`, today's in-tree code), then each installed extension:
its ``cli_commands`` and its ``pipeline_hooks.batch_completed`` / ``ledger_refresh``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def cli_command(name: str) -> Any:
    """The stage-owned ``extensions.CliCommand`` for subcommand *name*, or None."""
    from .builtin_cli_hooks import CLI_COMMANDS

    return CLI_COMMANDS.get(name)


def batch_completed(corpus_parent: Any, cfg: Any, log: Any) -> None:
    """Once a multi-feed batch with results is finalized (the identity ledger refresh)."""
    from .builtin_cli_hooks import BATCH_COMPLETED
    from .extensions import load_extensions

    for hook in BATCH_COMPLETED:
        hook(corpus_parent, cfg, log)
    for ext in load_extensions():
        if ext.pipeline_hooks is not None:
            for hook in ext.pipeline_hooks.batch_completed:
                hook(corpus_parent, cfg, log)


def ledger_refresh(root: Path | str, *, after: str, log: Optional[Any] = None) -> bool:
    """Refresh the identity ledger after a run of *after* (``enrichment``). True when written, or
    when no stage keeps a ledger (nothing to refresh)."""
    from .builtin_cli_hooks import LEDGER_REFRESH

    if LEDGER_REFRESH is None:
        return True
    return bool(LEDGER_REFRESH(root, after=after, log=log))


__all__ = ["batch_completed", "cli_command", "ledger_refresh"]
