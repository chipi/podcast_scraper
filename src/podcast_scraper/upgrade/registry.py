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
from .migrations.m0008_object_entity_kind import ObjectEntityKindMigration
from .migrations.m0009_backfill_speaker_roles import BackfillSpeakerRolesMigration
from .migrations.m0010_canonical_person_names import CanonicalPersonNamesMigration
from .migrations.m0011_shared_artwork_store import SharedArtworkStoreMigration
from .migrations.m0012_org_speakers_removed import OrgSpeakersRemovedMigration
from .migrations.m0013_artwork_thumbnails import ArtworkThumbnailsMigration
from .migrations.m0014_eponymous_hosts_restored import EponymousHostsRestoredMigration
from .migrations.m0015_unpublishable_speaker_names_removed import (
    UnpublishableSpeakerNamesRemovedMigration,
)
from .migrations.m0016_ad_reader_names_removed import AdReaderNamesRemovedMigration
from .migrations.m0017_speaker_names_canonicalised import SpeakerNamesCanonicalisedMigration
from .migrations.m0018_org_speakers_removed_residue import OrgSpeakersRemovedResidueMigration
from .migrations.m0019_descriptor_speaker_names_removed import (
    DescriptorSpeakerNamesRemovedMigration,
)
from .migrations.m0020_titled_person_ids_remerged import TitledPersonIdsRemergedMigration
from .migrations.m0021_backfill_feed_language import BackfillFeedLanguageMigration
from .migrations.m0022_derived_speaker_surfaces_resynced import (
    DerivedSpeakerSurfacesResyncedMigration,
)
from .migrations.m0023_transcript_speaker_prefixes_resynced import (
    TranscriptSpeakerPrefixesResyncedMigration,
)
from .migrations.m0024_shared_removed_speaker_prefixes import (
    SharedRemovedSpeakerPrefixesMigration,
)
from .migrations.m0025_one_person_one_entry import OnePersonOneEntryMigration
from .migrations.m0025_artwork_medium import ArtworkMediumMigration
from .migrations.m0026_missing_covers_stored import MissingCoversStoredMigration

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
    ObjectEntityKindMigration(),
    BackfillSpeakerRolesMigration(),
    CanonicalPersonNamesMigration(),
    SharedArtworkStoreMigration(),
    OrgSpeakersRemovedMigration(),
    ArtworkThumbnailsMigration(),
    EponymousHostsRestoredMigration(),
    UnpublishableSpeakerNamesRemovedMigration(),
    AdReaderNamesRemovedMigration(),
    SpeakerNamesCanonicalisedMigration(),
    OrgSpeakersRemovedResidueMigration(),
    DescriptorSpeakerNamesRemovedMigration(),
    TitledPersonIdsRemergedMigration(),
    # 0021 backfills each show's declared RSS <language> onto its episodes (#2173).
    # It FETCHES -- the first migration here that does. Every pre-#2172 artifact
    # carries the run config in feed.language, and skip_existing is GUID-keyed, so a
    # normal pipeline run never rewrites them: this is the only path.
    #
    # RENUMBERED THREE TIMES, and the reason is worth keeping because it will happen again. It
    # was 0011 until `main` landed its own m0011 (shared artwork store), 0015 until `main` landed
    # m0015-m0018 (the post-deploy speaker-name cleanups) on 2026-10-03, and 0019 until `main`
    # landed m0019-m0020 (descriptor names, titled person ids) on 2026-10-04. Two migrations
    # cannot share an id — the ledger records the id as applied, so a duplicate makes a corpus's
    # migration history ambiguous about which one ran — and `get_migrations()` sorts by id, so a
    # collision also silently reorders the sequence. A long-lived branch that adds a migration
    # must re-check the number at every merge; this one has never run in production, which is the
    # only reason renumbering is free.
    BackfillFeedLanguageMigration(),
    # 0022 makes context.json and the speaker diagnostics follow what 0012-0019 rewrote.
    DerivedSpeakerSurfacesResyncedMigration(),
    # 0023 re-renders the text transcripts from the repaired segments, offsets carried across.
    TranscriptSpeakerPrefixesResyncedMigration(),
    # 0024 renames what 0023 refused: a removed name on 2+ voices becomes SPEAKER (#2294).
    SharedRemovedSpeakerPrefixesMigration(),
    # 0025 repairs one person published twice (a title or a respelling) as c05cc0273 now avoids.
    OnePersonOneEntryMigration(),
    # 0025 writes the ≤1024px player copy of every stored cover (originals run to 3000px).
    ArtworkMediumMigration(),
    # 0026 stores covers that were only a feed-host URL (FETCHES), so every slot gets a downscale.
    MissingCoversStoredMigration(),
]


def get_migrations() -> List[Migration]:
    """All registered migrations, sorted by id (lexicographic == apply order)."""
    return sorted(_MIGRATIONS, key=lambda m: m.id)
