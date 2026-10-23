---
id: 0035_voice_identity_names
class: full
status: built, not applied
issue: none (the operator opens issues; naming arc, 2026-10-23)
touches:
  - src/podcast_scraper/upgrade/migrations/m0035_voice_identity_names.py
---

# 0035 — the voice identity rung on served episodes that carry fingerprints

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md). Plan:
podcast-scraper-eval-data `docs/wip/NAMING-GATES-2026-10-22.md`, the coding plan, item C8. A wrong
name is worse than a hidden insight (operator, 2026-10-19).

## 1. Intake

- **The pipeline change.** New episodes get the voice identity rung (`voice_identity_rung`, naming
  ladder doc v26 / v29). It compares an episode's voice fingerprints with the same feed's other
  episodes:
  - a voice that matches a fingerprint group named by vote takes that name;
  - a name leaves a voice that is clearly not that person.
- **Why served episodes need a migration.** Served episodes were named before the rung. They
  already carry what it needs, so it runs from text on disk with no DGX:
  - the diarizer's fingerprints (`voice_embeddings` in the speaker-diagnostics sidecar, captured
    since #2339);
  - the feed's published names.
- **Scope.** On prod, 413 of 2,939 served episodes (14%) carried fingerprints on 2026-10-23, all
  written since 2026-10-12. Episodes without fingerprints are not touched.

## 2. Outcome and baseline

- **The outcome:** a fingerprinted voice is served with its feed's voted name when it matches that
  group, and loses a name when it is clearly not that person, only where a review confirms it.
- **Measured** (eval repo `autoresearch/voice_embedding_match`, pre-registered val_v4, 10 feeds
  never used to shape it): wrong names 23 -> 17 (p = 0.031), correct names 70 -> 77 (p = 0.014).
- **Baseline, 10-17 snapshot copy (`lost/gateworkB6c`, read only).** `find_changes` lists 15
  voices: 12 named, 3 unnamed. Some look suspicious on sight (for example "Willem Els ->
  Christine", "Misha Glenny -> Melvyn Bragg"); the review decides each one.
- **Prod:** not counted yet. The list is made on a fresh prod snapshot (section 5). The 10-17 copy
  predates most fingerprinted episodes, so the prod list will be larger.

## 3. Diagnosis

- **The detector is the pipeline's own rule.** `episode_changes` calls the rung's
  `voice_identity.feed_history`, `build_groups` and `decide` on the stored fingerprints, with the
  pipeline's defaults (threshold 0.7, 60 s minimum talk for a group member). Nothing is
  re-implemented.
- **Served names, not ladder names.** The voice's current name and role come from the served
  segments (0034's `served_by_voice`), so the list describes what the reader sees today.

## 4. Design

- **A frozen, reviewed list**, as 0034's.
  - `find_changes` replays the rung over a corpus copy.
  - `write_list` freezes it in `upgrade/lists/0035_voice_identity_names.json`.
  - The migration applies that list and nothing else.
- **The review.** The reviewer reads each listed voice:
  - a RENAME is judged by its new name: `CORRECT` / `CORRECT_MISSPELLED` applies it;
    `WRONG_PERSON` leaves the voice as it is;
  - an UNNAMING is judged by the name it takes off: `WRONG_PERSON` applies it; `CORRECT` keeps
    the name;
  - a listed voice without a verdict fails `verify`, and without a verdicts file nothing is
    written.
- **The rewrite.** Through 0029's voice-by-voice rewrite with 0031's derived surfaces (0034's
  episode class): transcript prefixes and offsets, turns, diagnostics, context, ad maps, metadata
  counts, GI, KG and bridge. All or none per episode; an episode whose stored name or role moved on
  is refused untouched.
- **Undo:** from `.podcast_scraper/upgrade-backups/0035/`, while each file is still as written.
- **Not in the registry yet**, for the same reason as 0034.
- **Tests** (`tests/unit/podcast_scraper/upgrade/test_m0035_voice_identity_names.py`):
  - the detector names a host's unnamed voice from the feed's fingerprints;
  - an episode without fingerprints is not touched;
  - a reviewed rename is written on every surface, verified and undone;
  - a rename read as `WRONG_PERSON` is not written;
  - review names: a rename by its new name, an unnaming by its old one;
  - a listed voice without a verdict fails verify and nothing is written;
  - not registered.

## 5. Still to do before apply

1. Make the list on a fresh prod snapshot copy (`--find <copy>`).
2. Do the full read audit: read every listed voice's stretch, and judge the downstream LLM impact
   by identity.
3. Have a fresh reviewer write the verdicts file.
4. Produce the clone outcome table: before and after on the snapshot copy, every surface.
5. Decide the order against 0034 (both touch names; 0034 first means 0035's list is made after
   the floor is applied).
6. Register the migration, with its version.
7. Get the operator's recorded go.
8. Apply with the corpus-migration skill.

## Not covered by the tests

- An unnaming on a real corpus: the test checks its review semantics, not a rewrite. The rewrite
  path is 0034's, which its own tests cover with an unnaming.
- A rename to a name already on another voice in the same episode. `decide` is the pipeline's; the
  migration does not add a guard of its own.
- The 10-17 detector output has not been read voice by voice yet; that is step 2.
