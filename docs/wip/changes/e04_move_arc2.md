---
id: e04_move_arc2
class: light
status: open
issue: none
touches:
  - src/podcast_scraper/upgrade/migrations/m0006_kg_v2_typed_entities.py
  - src/podcast_scraper/upgrade/registry.py
  - src/podcast_scraper/upgrade/migrations/m0003_gi_v3_typed_mentions.py
  - src/podcast_scraper/upgrade/migrations/m0005_gi_v3_1_route_and_tag.py
  - src/podcast_scraper/upgrade/migrations/m0009_backfill_speaker_roles.py
  - src/podcast_scraper/upgrade/migrations/m0012_org_speakers_removed.py
  - src/podcast_scraper/upgrade/migrations/m0014_eponymous_hosts_restored.py
  - src/podcast_scraper/upgrade/migrations/m0015_unpublishable_speaker_names_removed.py
  - src/podcast_scraper/upgrade/migrations/m0016_ad_reader_names_removed.py
  - src/podcast_scraper/upgrade/migrations/m0017_speaker_names_canonicalised.py
  - src/podcast_scraper/upgrade/migrations/m0023_transcript_speaker_prefixes_resynced.py
  - src/podcast_scraper/upgrade/migrations/m0025_one_person_one_entry.py
  - src/podcast_scraper/upgrade/migrations/m0029_naming_arc_voices_repaired.py
  - src/podcast_scraper/upgrade/migrations/m0030_adfree_cut_resynced.py
  - src/podcast_scraper/upgrade/migrations/m0031_named_voices_shown.py
  - src/podcast_scraper/upgrade/migrations/m0035_voice_identity_names.py
  - src/podcast_scraper/upgrade/migrations/m0036_bridge_identities_restored.py
  - src/podcast_scraper/upgrade/migrations/m0038_raw_label_speakers_removed.py
  - src/podcast_scraper/upgrade/migrations/m0041_joined_names_respelled.py
  - src/podcast_scraper/upgrade/migrations/m0043_non_participant_names_removed.py
  - src/podcast_scraper/speaker_detectors/*.py
  - src/podcast_scraper/providers/ml/diarization/*.py
  - src/podcast_scraper/gi/pipeline.py
  - src/podcast_scraper/workflow/adfree_transcript.py
---

# e04_move_arc2 — the E04 move's second arc: public's copy of the moving set is deleted

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md), light class.

## 0. Why light

Arc 2 of the E04 move (SPLITTING_A_SURFACE_INTO_A_PRIVATE_REPO.md) deletes from public the code
that arc 1 copied into Common (`closelistening_common.stages`, Common branch
`docs/e04-ip-extraction-plan`), and moves with it the corpus migrations that import that code, so
public still loads. Nothing here writes a corpus file or changes what a migration does to one:

- the deleted modules run unchanged from Common, registered by its stages extension through the
  same seams (stage, read and CLI hooks, provider methods, enrichers, corpus migrations);
- `m0006` (a KG format migration, which stays public) changes only where it imports
  `migrate_kg_document_v2` from: the two KG transforms leave `migrations/gil_kg_identity_migrations`
  (whose GI transforms move) for `migrations/kg_document_migrations`, byte for byte;
- the 18 migrations that import moving code move to Common with their ids, outcomes, verdicts
  and lists, so prod's applied state reads the same ids.

Effect on landed outcomes: every migration's measured closure covers the whole package, so each
stored code digest moves with any commit (measured 2026-11-11: all 14 shipped outcomes are
already stale on origin/main). A moved migration's closure becomes Common's; its outcome is
re-recorded before it is next run or rerun, as for any code change.

Evidence: arc 2 proves equivalence with gate a (the same fixtures through old public, and through
new public with Common installed, compared artifact by artifact) and a public-only run (gate c),
both recorded in the move's commits.
