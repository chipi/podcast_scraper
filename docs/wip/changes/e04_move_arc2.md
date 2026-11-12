---
id: e04_move_arc2
class: light
status: open
issue: none
touches:
  - src/podcast_scraper/upgrade/migrations/m0006_kg_v2_typed_entities.py
  - src/podcast_scraper/upgrade/registry.py
  - src/podcast_scraper/upgrade/migrations/voice_rewrite_derived.py
  - src/podcast_scraper/upgrade/migrations/m0018_org_speakers_removed_residue.py
  - src/podcast_scraper/upgrade/migrations/m0019_descriptor_speaker_names_removed.py
  - src/podcast_scraper/upgrade/migrations/m0020_titled_person_ids_remerged.py
  - src/podcast_scraper/upgrade/migrations/m0022_derived_speaker_surfaces_resynced.py
  - src/podcast_scraper/upgrade/migrations/m0024_shared_removed_speaker_prefixes.py
  - src/podcast_scraper/upgrade/migrations/m0033_hand_removed_names_refused.py
  - src/podcast_scraper/upgrade/migrations/m0034_talk_floor_names_removed.py
  - src/podcast_scraper/upgrade/migrations/m0037_named_voices_corrected.py
  - src/podcast_scraper/upgrade/migrations/m0039_adfree_segment_offsets_realigned.py
  - src/podcast_scraper/upgrade/migrations/m0040_gi_quote_offsets_realigned.py
  - src/podcast_scraper/upgrade/migrations/m0042_insight_cluster_quotes_resynced.py
  - src/podcast_scraper/upgrade/migrations/m0044_title_spelling_voices_renamed.py
  - src/podcast_scraper/upgrade/migrations/m0045_episode_voices_corrected.py
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
  - src/podcast_scraper/upgrade/outcome_hook.py
---

# e04_move_arc2 — the E04 move's second arc: public's copy of the moving set is deleted

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md), light class.

## 0. Why light

Arc 2 of the E04 move (SPLITTING_A_SURFACE_INTO_A_PRIVATE_REPO.md) deletes from public the code
that arc 1 copied into Common (its stages package, Common branch
`docs/e04-ip-extraction-plan`), and moves with it the corpus migrations that import that code, so
public still loads. Nothing here writes a corpus file or changes what a migration does to one:

- the deleted modules run unchanged from Common, registered by its stages extension through the
  same seams (stage, read and CLI hooks, provider methods, enrichers, corpus migrations);
- `m0006` (a KG format migration, which stays public) changes only where it imports
  `migrate_kg_document_v2` from: the two KG transforms leave `migrations/gil_kg_identity_migrations`
  (whose GI transforms move) for `migrations/kg_document_migrations`, byte for byte;
- 31 migrations move to Common with their ids, outcomes, verdicts
  and lists, so prod's applied state reads the same ids: the 18 that import moving code, the 10
  speaker-naming repairs that import those (0018, 0019, 0020, 0022, 0024, 0033, 0034, 0037, 0044,
  0045) and the GI and ad repairs 0039, 0040, 0042 (E04 IP plan item 4: Close Listening's corpus
  repairs leave), with the helper voice_rewrite_derived.

Effect on landed outcomes: every migration's measured closure covers the whole package, so each
stored code digest moves with any commit (measured 2026-11-11: all 14 shipped outcomes are
already stale on origin/main). A moved migration's closure becomes Common's; its outcome is
re-recorded before it is next run or rerun, as for any code change.

Evidence: arc 2 proves equivalence with gate a (the same fixtures through old public, and through
new public with Common installed, compared artifact by artifact) and a public-only run (gate c),
both recorded in the move's commits.

## 1. The change method's scoreboard (operator decision, 2026-11-12: option B)

The corpus scoreboard (`quality/corpus_scoreboard`) moves with the naming and insight reports; the
change gate (`scripts/measure/corpus_change_gate.py`) and the outcome hook stay public. They read
the scoreboard through a seam, `Extension.corpus_scoreboard` (`quality/scoreboard.py`):

- the gating metrics leave `config/corpus_change_gate.json` for the scoreboard that defines them;
  Common pins the same nine (`test_stages_scoreboard_is_installed.py`);
- with no scoreboard installed, the gate and the hook refuse a corpus holding `*.gi.json` or
  `*.adfree.txt` (`quality.scoreboard.SCORED_FILES`) rather than reading every number as unchanged;
- with Common installed, the numbers are the same as before: the same `score_episode` runs.

## 2. Also in this arc (operator decisions, 2026-11-12)

- Scripts that drive moved code: 17 research scripts (the guard's former "until the move" list)
  and 9 GI/attribution reports go to the eval repository; 9 corpus repair and ops tools and the
  GIL quality gate go to Common's `scripts/`, with their tests. Their Make targets and public's
  GIL CI step leave public; Common's CI runs the GIL gate.
- Diarization's dependencies (pyannote.audio, torchaudio, torchcodec with its caps) leave public's
  `[ml]` extra for Common's `[diarization]` extra; the pyannote model preload goes with them.
  Prod's pipeline image (`stack-pipeline-llm`, `INSTALL_EXTRAS=llm`) never had them, so the prod
  image does not change.
- Common's images check the new `stages` extension loads (`docker/check_extensions.py`).

## 3. Not done in this arc

- Gate a (equivalence) and gate c (public-only install, end to end) are not run yet.
- Common's own suite has not run through its runner against this public commit (it needs the
  commit), so its copies of the split tests are proven only where run in place.
- Diarization, cleaning and the LLM bundle modes are still configurable in public (`diarize`,
  `diarization_*`, `transcript_cleaning_strategy`, `llm_pipeline_mode`) though the code behind them
  is Common's; without Common they do nothing or fall back. Moving that configuration needs a seam
  for extension-owned config (operator raised, 2026-11-12; not started).
- The eval repository does not yet depend on Common, so its moved scripts that import Common's
  modules run only beside a Common checkout; `scripts/validate/snapshot_quality.py` there calls
  Make targets the eval repository does not have (its GIL and KG numbers report `make-failed`).
- Docs that name moved paths, Make targets or scripts are not updated (among them the speaker
  attribution trace guide and the post-deploy runbook's transcript-pairing step).
- `config/corpus_change_gate.json`'s `content_code` still lists paths that left public; Common's
  copies of them have no edit gate yet.
- The English artifact golden's manifest half (its `turns` block is the speaker stage's report)
  is kept as stored by public's regeneration; Common's copy compares it but has no regeneration
  of its own yet.
- The remaining migrations of IP plan item 4 (0004, 0007, 0010), the prompts (item 6), the
  docs about private parts (item 7), Player's import sites and the viewer's GI panels.
