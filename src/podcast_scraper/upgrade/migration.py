"""Migration step abstractions (#862).

A ``Migration`` is an opaque, ordered, idempotent unit of corpus-upgrade work. It
knows nothing about *where* state is stored — the runner records the ledger via a
``StateStore`` — so the same step interface works whether the corpus is files on
disk today or a database later. A step that rewrites files now and one that targets
a DB later are indistinguishable to the runner.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass
class MigrationContext:
    """Inputs a migration needs to run, decoupled from storage specifics.

    ``options`` is the per-migration override bag (e.g. an explicit index path) and
    the natural extension point for a future DB handle — adding one means adding a
    field here, not changing every migration signature.
    """

    corpus_root: Path
    dry_run: bool = False
    options: Dict[str, Any] = field(default_factory=dict)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("upgrade"))

    def log(self, message: str) -> None:
        """Emit an info-level progress line (prefixed in dry-run)."""
        self.logger.info("%s%s", "[dry-run] " if self.dry_run else "", message)


@dataclass
class MigrationResult:
    """Outcome of a single migration's ``apply``."""

    migration_id: str
    applied: bool
    dry_run: bool = False
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class Migration(ABC):
    """One ordered, idempotent corpus-upgrade step.

    Subclasses set ``id`` (stable, lexicographically ordered, e.g.
    ``"0001_faiss_to_lance"``), ``to_version`` (the corpus version this step brings
    the corpus to), and ``description``; and implement ``apply``. ``apply`` must be
    safe to run more than once — the ledger prevents re-runs, but defence in depth
    (e.g. merge-insert writes) keeps a manual re-run harmless.
    """

    id: str = ""
    to_version: str = ""
    description: str = ""

    @abstractmethod
    def apply(self, ctx: MigrationContext) -> MigrationResult:
        """Perform the migration (or, when ``ctx.dry_run``, only report the plan)."""
        raise NotImplementedError

    def verify(self, ctx: MigrationContext) -> Tuple[bool, str]:
        """Check the migration's effect is present; ``(ok, message)``.

        Default: no verification (always ``True``). Steps with a checkable outcome
        (row counts, file existence) should override.
        """
        return True, "no verification defined"

    def plan(self, ctx: MigrationContext) -> str:
        """One-line human description of what ``apply`` would do for this corpus."""
        return self.description
