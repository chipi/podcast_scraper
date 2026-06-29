"""Ordered registry of corpus-upgrade migrations (#862).

Future migrations (index rebuilds, the entity canonical-map rebuild #852, schema
deltas, or a files → DB move) append here with the next ``mNNNN_`` id. The runner
applies any registered migration whose id is not yet in the corpus ledger.
"""

from __future__ import annotations

from typing import List

from .migration import Migration
from .migrations.m0001_faiss_to_lance import FaissToLanceMigration
from .migrations.m0002_two_tier_native_reindex import TwoTierNativeReindexMigration
from .migrations.m0003_gi_v3_typed_mentions import GiV3TypedMentionsMigration

# Source of truth, declared in intended apply order. 0001 migrates from FAISS when
# present; 0002 builds natively only when 0001 left no index — together they
# guarantee a two-tier index via the cheapest path. The entity canonical map (#852)
# is intentionally NOT a migration: it is computed live at graph-build, not persisted.
# 0003 retrofits the RFC-097 v3 GI schema migration into the framework — previously
# only runnable via the standalone scripts/migrate_gi_to_v3.py and easy to forget.
_MIGRATIONS: List[Migration] = [
    FaissToLanceMigration(),
    TwoTierNativeReindexMigration(),
    GiV3TypedMentionsMigration(),
]


def get_migrations() -> List[Migration]:
    """All registered migrations, sorted by id (lexicographic == apply order)."""
    return sorted(_MIGRATIONS, key=lambda m: m.id)
