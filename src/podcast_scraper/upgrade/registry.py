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
from .migrations.m0004_insight_type_reindex import InsightTypeReindexMigration
from .migrations.m0005_gi_v3_1_route_and_tag import GiV31RouteAndTagMigration
from .migrations.m0006_kg_v2_typed_entities import KgV2TypedEntitiesMigration
from .migrations.m0007_scope_bare_person_names import ScopeBarePersonNamesMigration

# Source of truth, declared in intended apply order. 0001 migrates from FAISS when
# present; 0002 builds natively only when 0001 left no index — together they
# guarantee a two-tier index via the cheapest path. The entity canonical map (#852)
# is intentionally NOT a migration: it is computed live at graph-build, not persisted.
# 0003 lands the RFC-097 v3 GI schema migration in the framework (the canonical home for every
# migration — see migrations/README.md); it wraps migrate_gi_document_v3.
# 0004 reindexes the two-tier LanceDB index when its schema predates the insight_type
# column (LANCE_SCHEMA_VERSION 3) so the Search v3 §S8 compare insight_types filter
# works — a fresh id because 0002 is already in every upgraded corpus's ledger.
# 0005 stamps GI 3.0 -> 3.1 (ADR-135/#1191); 0006 lands the RFC-097 v2 KG typed-entities
# migration. All three replace the former standalone scripts/migrate_*.py one-offs.
_MIGRATIONS: List[Migration] = [
    FaissToLanceMigration(),
    TwoTierNativeReindexMigration(),
    GiV3TypedMentionsMigration(),
    InsightTypeReindexMigration(),
    GiV31RouteAndTagMigration(),
    KgV2TypedEntitiesMigration(),
    ScopeBarePersonNamesMigration(),
]


def get_migrations() -> List[Migration]:
    """All registered migrations, sorted by id (lexicographic == apply order)."""
    return sorted(_MIGRATIONS, key=lambda m: m.id)
