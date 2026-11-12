"""Public's own stage subcommands and batch-end refreshes on the CLI hooks (E04 S7): none today.

The GI, identity-ledger, enrich-edges and clustering subcommands and the identity ledger refreshes
moved to the Close Listening stages extension with the E04 move (``Extension.cli_commands``,
``pipeline_hooks.batch_completed`` / ``ledger_refresh``).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from .extensions import CliCommand

CLI_COMMANDS: Dict[str, CliCommand] = {}
BATCH_COMPLETED: Tuple[Callable[[Any, Any, Any], None], ...] = ()
LEDGER_REFRESH: Optional[Callable[..., bool]] = None
