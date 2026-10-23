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
- **Baseline, 10-17 snapshot copy (`lost/gateworkB6c`, read only), after the review fixes:
  10 voices (7 named, 3 unnamed).**
  - The first run, before the fixes, listed 15. The 5 it no longer lists all hold 1.1-4.3% of
    their episode's talk, under the floor (M2). They include the two that looked wrong on sight
    (a guest's name replaced by another person's; a sub-5% voice given a host's name).
  - The absolute-path fix (M1) added no voice on this copy: 121 of its 2,685 served metadata
    files store an absolute path, and none of them changes a group here. Prod's share is not
    measured.
  - None of the 10 has been read yet (section 5, step 2).
- **Prod:** not counted yet. The list is made on a fresh prod snapshot (section 5). The 10-17 copy
  predates most fingerprinted episodes, so the prod list will be larger.

## 3. Diagnosis

- **The detector is the pipeline's own rule.** `episode_changes` calls the rung's
  `voice_identity.feed_history`, `build_groups` and `decide` on the stored fingerprints, with the
  pipeline's defaults (threshold 0.7, 60 s minimum talk for a group member). Nothing is
  re-implemented.
- **Served names, not ladder names.** The voice's current name and role come from the served
  segments (0034's `served_by_voice`), so the list describes what the reader sees today.
- **The talk floor comes after the rung.** In the pipeline a voice under 5% of the talk ends
  unnamed whatever the rung says. A host-read ad carries the host's fingerprint, so the rung
  alone would put the host's name on it, and a review could only answer "it is the host's
  voice". Voices under the floor are not listed; they are 0034's.
- **Siblings with an absolute stored path.** Prod metadata often stores the transcript path as
  written at ingest (`/app/output/...`). The rung's feed history now maps it into the corpus
  being read (`voice_identity.sibling_diagnostics_path`), so a snapshot copy sees every sibling.

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
- **The list is bound.** The verdicts header's `list_sha256` is the hash of the WHOLE list
  (every field of every entry: both names, both roles, the evidence), the list file is part of
  the migration's digest (`list_file`), and the receipts header records the list's hash. An
  entry edited after the review refuses the load.
- **A renamed voice's source is `voice_identity`**, as the pipeline records the rung's names, so
  a name this migration applies never votes in the feed bank (`voice_identity._NON_VOTING_SOURCES`).
  That includes a respelling of the same person ("Jon Smith" -> the group's "John Smith"), which
  the shared rewrite would otherwise leave on its old source (`RESPELLING_TAKES_RENAMED_SOURCE`).
- **Placeholder labels are no names.** A served label "SPEAKER_01", "Speaker", "Speaker 2" or
  "Unknown Speaker" counts as an unnamed voice (0034's `_LABEL`), so the review never judges one.
- **The rewrite.** Through 0029's voice-by-voice rewrite with 0031's derived surfaces (0034's
  episode class): transcript prefixes and offsets, turns, diagnostics, context, ad maps, metadata
  counts, GI, KG and bridge. All or none per episode; an episode whose stored name or role moved on
  is refused untouched.
- **Undo:** from `.podcast_scraper/upgrade-backups/0035/`, while each file is still as written.
- **Not in the registry yet**, for the same reason as 0034.
- **Tests** (`tests/unit/podcast_scraper/upgrade/test_m0035_voice_identity_names.py`; each fix
  from the review mutation-checked: reverting it fails its test):
  - the detector names a host's unnamed voice from the feed's fingerprints;
  - an episode without fingerprints is not touched;
  - a reviewed rename is written on every surface, verified and undone;
  - a rename read as `WRONG_PERSON` is not written;
  - review names: a rename by its new name, an unnaming by its old one;
  - a listed voice without a verdict fails verify and nothing is written;
  - a voice under the floor is not listed, even when it is the host's voice;
  - siblings stored with an absolute path still vote;
  - a name this migration applied does not vote afterwards, a respelling included;
  - not registered.

## 5. Still to do before apply

1. Make the list on a fresh prod snapshot copy (`--find <copy>`).
2. Do the full read audit: read every listed voice's stretch, and judge the downstream LLM impact
   by identity.
3. Have a fresh reviewer write the verdicts file.
4. Produce the clone outcome table: before and after on the snapshot copy, every surface.
5. Make 0034's and 0035's lists on the SAME snapshot. 0035 lists no voice under the floor, so
   the two lists do not share a voice: a voice on both would refuse its episode in whichever
   applies second (all or none), and that migration's `verify` would then fail on it.
6. Register the migration, with its version.
7. Get the operator's recorded go.
8. Apply with the corpus-migration skill.

## Not covered by the tests

- An unnaming on a corpus: the test checks its review semantics, not a rewrite. The rewrite
  path is 0034's, which its own tests cover with an unnaming on every surface.
- The surface check reads a quote's speaker through its `speaker_id` and the Person node, as the
  readers do (`gi/load.py`, `server/app_gi_view.py`): the rewrite clears the authored
  `speaker_name`. The fixture's Person node for the raw id `SPEAKER_00` stays; real GI makes no
  Person from a raw id (`gi/speakers.py`).
- A rename to a name already on another voice in the same episode. `decide` is the pipeline's; the
  migration does not add a guard of its own.
- The 10-17 detector output has not been read voice by voice yet; that is step 2.
