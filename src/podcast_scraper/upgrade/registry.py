"""Ordered registry of corpus-upgrade migrations (#862).

Future migrations (index rebuilds, the entity canonical-map rebuild #852, schema
deltas, or a files → DB move) append here with the next ``mNNNN_`` id. The runner
applies any registered migration whose id is not yet in the corpus ledger.
"""

from __future__ import annotations

from typing import List

from .migration import Migration
from .migrations.m0001_faiss_to_lance import FaissToLanceMigration

# Source of truth, declared in intended apply order.
_MIGRATIONS: List[Migration] = [
    FaissToLanceMigration(),
]


def get_migrations() -> List[Migration]:
    """All registered migrations, sorted by id (lexicographic == apply order)."""
    return sorted(_MIGRATIONS, key=lambda m: m.id)
