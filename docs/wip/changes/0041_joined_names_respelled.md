---
id: 0041_joined_names_respelled
class: full
status: landed
issue: none (RFC-131 plan V3, D2 items 8 and 9; operator, 2026-10-31)
touches:
  - src/podcast_scraper/upgrade/migrations/m0041_joined_names_respelled.py
  - src/podcast_scraper/upgrade/registry.py
  - src/podcast_scraper/identity/respell.py
  - src/podcast_scraper/identity/ledger.py
  - src/podcast_scraper/kg/entity_clusters.py
  - src/podcast_scraper/upgrade/migrations/m0025_one_person_one_entry.py
  - src/podcast_scraper/upgrade/migrations/m0029_naming_arc_voices_repaired.py
  - src/podcast_scraper/upgrade/migrations/m0031_named_voices_shown.py
---

# 0041 — a joined speaker's name in the episode files is the group's one spelling, and stays so

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Plan V3, D2 item 9 (transcript
names follow the group spelling, by a migration) with item 8 (one person listed twice in 5 episodes)
folded in. Person names are kept out of this public record; the measurements and per-episode lists
are in eval-data `data/eval/name-consolidation/d2i9/`.

## 1. Intake

- **What is wrong.** After S6a–S6d every served surface shows a joined person under the group's
  spelling (the ledger's `display_name`), at read time. The transcript does not: its speaker
  labels and prefixes keep each episode's own spelling. The operator excluded the transcript from
  the read-time respelling and asked for the files to be fixed instead (2026-10-31), so the
  transcript and every other surface agree.
- **The operator's decisions (2026-10-31):**
  - a migration rewrites the names in the episode files;
  - the ledger step respells the files again after every build in which a group's spelling
    changes, text-only, with receipts ("Ledger step respells");
  - item 8's 5 episodes are folded in the same migration, with S6's spelling rule.
- **Reading list:**
  - `upgrade/migrations/m0025_one_person_one_entry.py`: the name-map rewrite (`_Episode.plan`'s
    name path): metadata, segments, the three transcripts in place with every GI quote and
    ad-free offset carried (m0023), GI, KG, bridge, node labels, diagnostics voices, turns
    (rebuilt by the pipeline's builder or the episode is refused), context digest. `_record`
    merges two entries that end with one name. `_id_map` moves an id whenever a name's slug
    changes.
  - `m0017_speaker_names_canonicalised.py`: the five-surface `rewrite(names, ids)`; names and ids
    are separate maps; a KG `mentioned` entity's name is not rewritten, only a speaker's.
  - `voice_rewrite_derived.py`: ad maps carried and the metadata counts resynced after a rewrite
    (0031's derived surfaces).
  - `m0029`, `m0035`, `m0037`: the migrations built on that rewrite, their receipts, backups and
    undo (`upgrade/file_rewrite.py`).
  - `identity/ledger.py`: `build_ledger`, `write_ledger`, `refresh_ledger` (called after enrich,
    an upgrade run, and enrich-edges), `restore_previous`, the change log.
  - `identity/guards.py`: the evidence the ledger reads — KG Person names, the roster
    (`content.speakers` with the segment labels' voices), and the diagnostics' publisher text
    (`tried.known_hosts`, `metadata_named`, `detected_guests`); a publisher name belongs to a
    node by its folded name or by the id it mints.
  - `identity/roster_provenance.py`: `content.detected_hosts` / `detected_guests` are the
    pre-diarization hint from the feed and show notes.
  - `server/segments_view.py`, `routes/app_episodes.episode_segments`: segments are read from
    disk per request, and the label is shown as text.
  - `perf_cache.corpus_mtime`: caches key on the run summaries and the ledger file, not on
    episode files.
  - ADR-166 (a spelling never renames an id), ADR-167 (guards), S6a's display rule.
  - D2 item 8's cause (plan V3): 0034 and 0035 wrote a second spelling of a person into 5
    episodes that already listed the person; 0025's verify fails on them.

## 2. Outcome and baseline

Snapshot-prod-20261030, the ledger built in memory at S6d (46667141a), every served episode
(`d2i9/population.py`, per episode in `d2i9/population_episodes.jsonl`):

- 658 episodes have a joined Person node (KG or GI) whose name differs from its group's spelling;
- 106 of them have speaker segments under that name.

The fields that carry such a name today:

| surface | fields |
|---|---|
| segments `speaker_label` (`.segments.json` / `.adfree.segments.json`) | 25,900 / 25,714 |
| transcript speaker prefixes (`.txt` / `.adfree.txt` / `.cleaned.txt`) | 2,472 / 2,470 / 2,020 |
| turns `speaker_label` (`.turns.json` / `.adfree.turns.json`) | 869 / 867 |
| KG Person `name` / `label` | 839 / 839 |
| GI Person `name`; GI Insight `speaker` | 258; 1,444 |
| bridge `display_name` | 839 |
| context `people` / `basic.hosts` / `basic.guests` | 843 / 57 / 59 |
| metadata `content.speakers[].name` | 117 |
| diagnostics `voices[].resolved_name` | 112 |
| metadata `detected_hosts` / `detected_guests` (publisher hint) | 10 / 20 |

- A name is linked to a group through every Person node of that name in the episode, KG and GI
  together. No name on the snapshot has an unjoined node or two group spellings under it (0
  refused).
- **The plan's earlier figure** (24,933 segments in 86 episodes, `s6/segments_surface.py`) linked
  through GI nodes only. KG and GI together give 25,900 in 106 (`d2i9/reconcile.py`). The KG lists
  1,681 people the GI does not, the GI 14 the KG does not.
- 0025's dry run on the snapshot (`d2v/m0025_dry.py`): 5 episodes still to rewrite, item 8's.

## 3. Diagnosis

- The ledger decides a group's spelling (S6a); nothing writes it back to the episode files. Read
  time covers every surface but the transcript, by the operator's choice.
- Item 8: the guards keep two ids out of one group when they sit in one episode (0 groups have two
  ids in one episode, `d2i9/folds.py`), so the 5 duplicates are two roster entries and two nodes of
  one person that the ledger does not join. 0025's name-map rewrite merges such entries, but by its
  own spelling rule.

## 4. Prediction

Dated: 2026-10-31

### The change

1. **`identity/respell.py`: one home, used by the migration and the ledger step.**
   - `episode_renames(meta, view)`: the episode's name map. A Person node name (KG and GI
     together) is respelled when every node of that name is joined and all take one group
     spelling different from the name. Otherwise it is refused and counted.
   - **Every joined Person node, speaker or mentioned.** m0017 renames only a speaker's node (its
     rule was for cleaning titles off voices). Here the name is the person's, wherever the file
     names them: a mention-only node's `name` / `label`, its bridge `display_name`, and the
     context digest's `people`, which MCP and the digest's readers take raw. A pass after m0017's
     rewrite does the mentioned nodes. The spoken text is never changed.
   - **Item 8's fold:** a roster entry whose name 0025's `same_person` matches to a respelled
     name, or to the group spelling, under another spelling, takes the group spelling too. Its
     node, if it has one under another id, folds into the joined id. That is the one case where
     an id moves; elsewhere the id map is empty (ADR-166).
2. **The rewrite** is 0025's name-map path through 0031's derived surfaces (ad maps, metadata
   counts), with three overrides:
   - `_id_map`: only the fold above;
   - no host promotion;
   - `_record` merges entries only in an episode with a fold, so other episodes keep their roster
     order and entry ids.
   - **Kept unchanged, as evidence:** metadata `detected_hosts` / `detected_guests` (put back
     after m0017's rewrite, which respells them), the diagnostics' publisher text, decision
     traces and model outputs, and every spoken word.
3. **Migration `0041_joined_names_respelled`:** the first full run over the corpus. Plan, apply
   with backups and receipts (`joined_names_respelled.jsonl`, tag `0041`), verify (nothing left to
   respell and nothing refused), undo from receipts. It reads the served ledger. With no valid
   ledger it writes nothing and verify says so.
4. **The ledger step respells:** `refresh_ledger` builds the ledger, respells the files to the new
   build's spellings, then writes the ledger.
   - **The order is deliberate:** the ledger write moves `corpus_mtime`, which invalidates every
     cache. Respelling after the write would leave a window in which a cache holds the old files
     under the new token.
   - **Its own backups and receipts per build:** tag `ledger-respell/<built_at>`, receipts
     `ledger_respell.jsonl` with the build stamp per row, so the undo of one respell restores
     the state just before it.
   - **A refused or failed episode** is logged and counted, and the ledger is still written.
     Read time keeps every other surface right; the transcript lags until the next build.
   - **Atomic writes and the corpus owner's ownership:** the shared writer's
     (`file_rewrite.write_with_backup`).
   - `restore_previous` (the ledger's undo) does not respell back; the files then show the
     newer spelling until the next build. Stated, not changed.

### Predicted on snapshot-prod-20261030 (a clone, the ledger built and written first)

- **Episodes written by 0041:** 658, plus any of item 8's 5 not among them; the dry run's count is
  explained before step 7 if it differs.
- **Fields after:** every count in the section 2 table but the publisher hint goes to 0, through
  the real read of the clone after (`population.py` rerun).
- **Item 8:** the 5 episodes fold; 0025's verify passes on the clone after.
- **Refused:** 0 names (section 2). Episodes refused by the transcript or turns guards are listed
  with the reason; each one is read.
- **Must not move:**
  - every person id outside the 5 folds; every insight, quote and edge count outside them;
  - the spoken text: the transcript with speaker prefixes removed is identical, byte for byte;
  - the ledger: rebuilt on the clone after, its joins equal the joins before, and every group's
    `display_name` is unchanged (the fixpoint: the files now carry the group spelling, and
    publisher text is untouched);
  - `detected_hosts` / `detected_guests` and the diagnostics' publisher text;
  - every read-time surface's output (S6a–S6d gates rerun on the clone: the same shown names);
  - the search index: see 4b (a correction).
- **A second run** of 0041 and of the ledger step writes nothing.
- **The ledger step on a later spelling change:** a synthetic corpus whose group spelling changes
  between two builds; the second build respells exactly that group's episodes, and its undo
  restores them.

### 4b. Rows added after the fresh read

**The search index.** A correction: section 4 said its transcript rows "hold text, not names". That
was false; I wrote it without checking. Transcript chunks are slices of the analysis transcript,
speaker prefixes included (`search/chunker.py` slices `text[c_start:c_end]`), with their offsets.
A respelled prefix changes chunk text and shifts every later offset in the episode; Quote rows
carry offsets too.
- The index's per-episode fingerprint hashes every embeddable row's `(doc_id, text)`
  (`two_tier_indexer._episode_fingerprint`). So the next `reindex` re-embeds exactly the episodes
  whose rows changed, offsets with them: ladder step 4, the affected episodes only, as D1's run
  steps did (deploy, ledger, migration, enrich, reindex).
- Predicted: the reindex after 0041 touches the episodes 0041 writes and no others. Measured on
  the clone with the fixture-free count of changed rows per episode (step 7).
- Between the migration and the reindex, a transcript hit's lift reads offsets that no longer
  match the file: stated, and the window is the run's own (minutes).

**The ledger fixpoint, measured before any code** (`d2i9/fixpoint.py`: the ledger rebuilt with its
own inputs respelled as 0041 leaves them, publisher text kept):

| | before | after |
|---|---|---|
| joins | 840 | 840 |
| person groups | 475 | 474 |
| groups whose canonical id moves to another member | | **92** |
| groups whose members change | | 1 (one variant added by chaining) |
| group gone | | 1 (a two-person pair a guard now blocks: a correction) |
| `display_name` changed | | **0** |
| `display_evidence` changed | | 5 (frequency becomes publisher, or the publisher episode count grows; the name is the same) |
| blocked joins | 9 | 10 (the pair above) |

- **The cause of the 92:** `_pick_canonical` breaks a frequency tie by the longest name, then the
  name. With every member under one spelling, the tie falls to the file order. A canonical id is
  what person pages, links and follows key on, so the reader was right: an argument was not a
  fixpoint.
- **Added to the change: the ledger keeps the previous build's canonical id when it is still a
  member of the group** (`build_ledger` reads the served ledger first). That holds person URLs
  across every build, not only this one; today an ingest that shifts a frequency can move them
  too. Predicted: the fixpoint test with it moves 0 canonical ids, the gone pair aside.
- The member added and the pair removed are listed for reading at step 8.
- **Must not move (added):** `blocked_joins` beyond the pair; canonical ids; `display_name`.

**Item 8's fold, defined.**
- The survivor is the id in a ledger group; its variant is the other id of the pair.
- **Refused** when both are in groups, or when neither is.
- **Refused** when the variant has episodes outside this one, since its group membership
  elsewhere would lose the episode.
- **Refused** when `same_person` matches only by a bare name: the fold needs two full names.
- The fold is an identity decision on the episode's own evidence (the roster lists one person
  twice), which is 0025's precedent, not a spelling. It is the only id move. Scoped ids
  (`is_scoped_person_id`) never move.
- Edges duplicated by the move are de-duplicated (m0017's rewrite does). `_record` merges the
  two entries once their names are equal.
- **Step 8, for each of the 5:**
  - both nodes' voices and segments read (one voice?);
  - the variant's episode count elsewhere;
  - node, edge, quote and insight counts before and after;
  - 0025's verify;
  - no id outside the 5 moved.

**The ledger step.**
- **Receipts per build:** `ledger_respell/<stamp>.jsonl`, the stamp filesystem-safe (`20261031T120000Z`),
  with the backup tag `ledger-respell-<stamp>`. The undo takes the stamp. A retry of the same build
  after a partial failure uses a new stamp, so a kept backup is never stale.
- **It writes only when a group's spelling changed** against the previous ledger. It then reads
  only those groups' episodes, which the ledger's members list. No full scan per build.
- **Under the corpus write lock** (`utils/corpus_lock`): a caller already holding it passes
  through; otherwise the step takes it.
- **The server can read an episode mid-rewrite** (files go one at a time, segments before the
  transcript). m0025 has the same window; it is per episode and milliseconds. Stated.
- **A refused or failed episode** is counted in the ledger's change log (`respell_lagging`). No code
  reads the files as proof of the ledger.
- **`restore_previous` and the file undo stay two operations.** After a ledger restore, the files
  keep the newer spelling until the next build respells them. Stated.

**Must not move, made checkable:**
- Spoken text: per transcript line, the line count is equal, and each line's text after its
  leading `name:` label is equal (all three variants). Segments: `text`, `start` and `end` equal.
- GI quote spans still slice to the same text after m0023's offset carry.
- Mention-only and every other node id unchanged (outside the 5 folds); non-Person KG entities
  unchanged; `speaker_id` counts in GI, bridge and context unchanged.
- `detected_*` and the diagnostics' publisher text unchanged.
- **The diagnostics:** `voices[].resolved_name` is respelled, being the roster's name for the voice.
  The voice's `source` stays (the evidence for a respelling still holds; m0025's
  `RESPELLING_TAKES_RENAMED_SOURCE` is off). Decision traces stay.

**Derived outputs that carry names:**
- `guest_coappearance` and `topic_consensus` read KG names; the enrich run after the migration
  rebuilds them, as in D1's run steps. Read time spells them meanwhile (S6d).
- `person_web` keys on ids, and `insight_clusters` on insight ids; neither is changed.
- No cache keys on episode-file mtimes for these files: segments are read per request
  (`episode_segments`); the others key on `corpus_mtime`, which the ledger write moves.

### 4c. Revision after the two gate runs

Dated: 2026-11-01T09:15

What changed since the measurement at 2026-10-31T21:35, each recorded in section 6 or its own record:
- 0025 defers to the ledger's spelling; 0041's fold finds a survivor under a respelled name;
- the display rule never shows a bare name over a full one;
- **the publisher's own text is the reference spelling** (record `s6e_publisher_spelling`, the
  operator's decision): 10 groups move;
- the frozen-list migrations read a group spelling as the listed name (`m0029.as_listed`);
- **0039** realigns the one episode whose ad-free offsets a hand repair broke, before 0041;
- **0041 also carries 0025's pending renames of unjoined names.** Where 0025's own decision for an
  episode is renames only (no swap, no voice rename, no refusal, nothing left), each rename of a
  name no ledger group holds is applied. That is the feed's stated host spelling, which is the
  publisher's text, as the operator decided. A name's node folds into the stated host's node when
  both are in the episode, as 0025 folds them. This covers item 8's last 4 episodes.

Predicted on the clone (snapshot-prod-20261030, 0038 and 0039 applied, the ledger rebuilt with this
code):
- `upgrade verify` after 0041: **0 failing** (it was 3: 0025, 0035, 0041);
  - 0035's 5 episodes read as done;
  - 0041's 1 refusal is gone (0039);
  - 0025's 4 are applied through 0041.
- 0041 refuses 0 episodes and leaves 0 joined names unrespelled (`population.py` after: every
  respelled field 0, the publisher hint kept);
- the ledger rebuilt after 0041: joins equal (840), 0 canonical ids moved, every `display_name`
  equal;
- the scoreboard's gating numbers do not move (as in both runs);
- the number of episodes written and per-field counts: read from the gate report and explained in
  section 7.

### 4d. Rows added after the revision's fresh read

Dated: 2026-11-01T09:15

- **0025's whole decision is carried, or none of it.**
  - Carried: its renames, its `dropped` unplaced twins and its host promotions (`promote`), with
    0025's own `_record` merging and renumbering the roster.
  - Not carried: an episode where 0025 renames a voice (`voice_names`, which also holds
    REPLAYED_VOICES), leaves a pair, or refuses.
- **Only publisher-text targets.** Every carried target must be in the episode's `known_hosts` or
  `metadata_named`. A target only in `detected_guests` (NER over the description) is not carried,
  and the episode is counted as skipped.
- **Ids follow 0025's own rule,** so 0025's verify and 0041 agree. The old name's id moves to the new
  name's id where they differ, never a scoped id (0025's `_id_map`). A name held by any ledger group
  is never carried, so no joined id moves.
- The carried names are 0025's output, its `canonical_person_name` included.
- **Predicted (rows):**
  - the carried episodes are item 8's 4 (`d2i9/m0025_after_0041.txt`), each listed with its carry or
    skip reason;
  - none has a target outside `known_hosts` or `metadata_named`;
  - `upgrade verify --only 0025_one_person_one_entry` after: OK;
  - a second 0041 apply writes nothing, and 0025's `decide` on the 4 finds nothing;
  - joins 840 and every group's `display_name` unchanged after the rebuild;
  - the 4's roles before and after are listed (`promote`).

### 4e. Revision after the third gate run (`gate0041c`)

Dated: 2026-11-01T10:42

`upgrade verify` after 0041: 38 ok, failing 0035 alone (4 episodes, item 8's last 4), a regression.
- **The cause:** two publisher spellings of one guest disagree. The episode's metadata says one;
  0035 renamed the voice to its cross-episode voice group's name, the person's spelling in their
  other episodes, reviewed CORRECT or CORRECT_MISSPELLED. 0041 carried 0025's rename of that voice to
  the episode's metadata spelling, and 0035 then wants its own name back.
- **The rule (the operator's, with s6e's tie-break):** publisher texts that disagree are decided by
  the person's episodes, so the reviewed cross-episode name stands. 0025 now leaves a voice whose
  diagnostics `source` is `voice_identity` (`m0025._identity_named`), as it leaves the ledger's
  spelling. These 4 episodes then have nothing for 0025 to do: there is no duplicate roster entry;
  0025's rename came from the metadata's participant list.
- **Narrowed after the fresh read:** `voice_identity` is also what the pipeline's fingerprint rung
  writes for an unreviewed name. 0025 now defers only to a name 0035's review confirmed for that
  voice (a rename judged `CORRECT` / `CORRECT_MISSPELLED`; `m0025._reviewed_identity_names`). An
  unreviewed fingerprint name stays open to 0025's correction.
- **Measured** (`d2i9/identity_deference.py`, the clone before 0041): 0025 defers in **4 episodes**,
  item 8's 4. All 4 names are in 0035's reviewed list, and none carries a title.
- **Predicted on the clone:** before 0041, 0025's verify fails on 1 episode (measured: the one
  ledger-joined pair), which 0041 folds. After 0041, `upgrade verify` has **0 failing**. 0041
  carries 0025's decision in 0 of the 4 (their names stay 0035's). Every other 4c and 4d row stands.

### 4f. Revision after the fourth gate run (`gate0041d`)

Dated: 2026-11-01T11:55

The gate is green: `upgrade verify` 39 ok, none failing, no gating number moved, 689 episodes
written. The after-checks (`d2i9/after_checks.py`, `after_checks_gate0041d.txt`) passed every
must-not-move row but one:
- **One ledger group changed membership** on the rebuild. The full-name display rule gave a group
  the spelling of its full-name member, and 0041 wrote that two-word name over 5 episode-scoped bare
  mentions (`person:unresolved-*`). On the rebuild those were full names, eligible for the join test,
  and the group chained into another group of a similar name: a join the respelling created. Two
  members split off under guards. The predicted merge of one group's spelling variants happened too.
- **The change:** 0041 never respells a name held by an episode-scoped placeholder id. Those ids are
  kept bare on purpose (`identity/bare_name_scope`). Read time still shows them under the group
  spelling (S6), so no surface changes. The name is refused and counted.
- **Also measured, not caused by 0041:**
  - 24 GI quotes in 11 episodes do not slice to their transcript text, the same before and after
    (`d2i9/quote_spans.py`), listed for the operator as a separate finding;
  - one episode's shown names changed from an authored quote name to the group spelling, the
    intended consistency.

Predicted on the next gate run: `upgrade verify` 0 failing; the rebuilt ledger's joins 840, its
groups' members unchanged but for the predicted variant merge, `display_name` 0 changed; the
written-episode count lower than 689 by the episodes where only scoped names changed.

### 4f-b. Rows after 4f's fresh read, measured before the gate

Dated: 2026-11-01T12:02

- **The check is in `episode_renames`' main loop** (`_placeholder`): an episode-scoped id
  (`is_scoped_person_id`), or an unresolved-speaker placeholder of any shape
  (`graph_id_utils.is_unresolved_speaker_placeholder`: `person:speaker-<ep>-<n>`, a legacy role id).
  A name held by one is refused whole, and counted only when it is joined. Tested per shape.
- **Measured** (`d2i9/dry_counts.py`, 0041's dry run on the clone):
  - **573 episodes to write** (689 before the rule: 116 changed scoped names only);
  - **83 names refused** as episode-scoped, 0 of them mixed with a real joined id;
  - 3 refused folds ("a fold needs two full names"); 0 episodes refused.
- **15 of the 83 are speaker labels in a transcript.** They keep the episode's bare name there, while
  read time shows the group's full spelling elsewhere. This is accepted and stated: a full spelling
  written over a scoped placeholder is what created the wrong join. Listed for the operator's read.
- **Predicted on the gate:** 573 episodes written; `upgrade verify` 0 failing; the rebuilt ledger's
  joins 840; `display_name` 0 changed; groups whose members change: 1 (the predicted variant merge),
  with the chain into another group gone; blocked joins as before the respelling (9) or the one
  two-person pair more (10).

### 4g. Revision after the final batch's gate (`gate-0041_joined_names_respelled`)

Dated: 2026-11-01T14:49

The gate is green: `upgrade verify` 40 ok, none failing; 573 episodes, 2,864 files; no gating number
moved. The after-checks (`d2i9/after_checks_final.txt`, base-0041 against base-0042, which is 0041's
after-state) pass every must-not-move row but the ledger's:
- **Joins 658 → 631 on the rebuild** (`d2i9/join_diff.py`): 28 lost, 1 gained, 5 moved, 3 groups'
  members changed, display names 0 changed.
- **The cause, general and not one case:** 0041 rewrote the names the ledger reads as evidence.
  - A one-word, unscoped node took its group's two-word spelling. Its 26 bare scoped
    mentions then no longer matched it.
  - A different person's name then matched the new spelling: the one gained join, a wrong-person
    risk.
  - 4f's placeholder rule closed one path of this, not the cause.
- **The change: evidence and presentation are kept apart.** When 0041 respells a name, it keeps the
  episode's own spelling beside it:
  - `respelled_from` on the Person node (KG, GI) and on the roster entry (`content.speakers`);
  - the first original is kept when a later build respells again.
- The ledger's readers take `respelled_from` when present: the candidates
  (`kg/entity_clusters.collect_entity_candidates`) and the episode evidence
  (`identity/guards.collect_episode_evidence`: people and roster). So the ledger reads exactly the
  names it read before 0041, and the fixpoint holds by construction. Every surface still shows the
  respelled name.
- **Predicted:** the rebuild after 0041 equals the ledger before it. That is joins 658, 0 changed,
  0 groups' members changed, `blocked_joins` equal, `display_name` 0 changed. Everything else in
  4c-4f-b stands (573 episodes, 0 failing).

### 4g-b. Rows after 4g's fresh read

Dated: 2026-11-01T14:50

- **The label too:** the candidate reads `name or label`, and 0041 respells both, so `respelled_from`
  is read first, before either.
- **Covered through the two readers:** `_name_counts`, a candidate's representative name,
  `_add_display_spellings` and `_publisher_spellings` (`person_names`).
- **Not ledger inputs (checked):**
  - `identity/intra_episode_merge` runs only in the ingest pipeline (`workflow/metadata_generation`);
  - no ledger reader opens the bridge.
- **The roster's voices:** `entry_voices` gets the entry unchanged (its respelled name matches the
  respelled segment labels for an older record without `voices`). Only the name in the evidence's
  `(name, voices)` pair is `respelled_from`.
  - Where a roster entry is respelled and no segment label is, an older record had no label of that
    name before either; nothing changes.
- **The key survives:** later rewriters edit `properties` in place (m0017, m0025, m0029), so the key
  is kept. A reprocess rebuilds the node from fresh text, with no key, which is that episode's new
  own spelling.
- **Baseline for the prediction:** the pre-0041 ledger of base-0041 (joins, canonical ids,
  `blocked_joins`), diffed with `d2i9/join_diff.py`.

## 5. Acceptance

Fresh reader, 2026-10-31, a sonnet reviewer subagent with read-only tools, about 1.5 minutes, before
any code: "not complete". Every row is taken in 4b, and two were measured before answering (the
index fingerprint, the ledger fixpoint). The fixpoint measurement added the canonical-id stability
to the change.

Spot checks for step 8: item 8's five, as listed in 4b; the chained member and the removed pair;
one respelled episode read end to end (transcript, segments, player view); one reindexed
episode's hits.

Revision, fresh reader (a sonnet reviewer subagent with read-only tools, about 1.5 minutes, before the
revision's code):
Dated: 2026-11-01T09:14
"Not complete": `dropped` and `promote` were missing from the carry, `detected_guests` is not the
publisher's statement, and 0025 moves ids in more cases than "both nodes in the episode". Every row
is taken in 4d.

Fourth revision (4e), fresh reader (a sonnet reviewer subagent with read-only tools, about 30 s):
Dated: 2026-11-01T10:42
"Not complete": `voice_identity` also marks unreviewed names; the deference's reach was not
counted; a titled name was not checked. Taken: the deference is narrowed to reviewed names and
measured (4 episodes, all reviewed, none titled).

Fifth revision (4f), fresh reader (a sonnet reviewer subagent with read-only tools, about 30 s):
Dated: 2026-11-01T11:55
"Not complete": the check belongs in the main loop; other placeholder shapes; the counts and the
speaker-label question. All are taken in 4f-b, with the counts measured.

Sixth revision (4g), fresh reader (a sonnet reviewer subagent with read-only tools, about 45 s):
Dated: 2026-11-01T14:49
"Not complete": the label fallback; the intra-episode merge and the bridge (checked: not ledger
inputs); the roster's voices; the key's survival; the baseline. All are taken in 4g-b.

## 6. Build

### Part A: the ledger keeps its canonical ids

Tests, written first (`tests/unit/podcast_scraper/identity/test_ledger.py`, invented names; the first
and third failed before the code):
- a rebuild that shifts the frequencies keeps the served canonical id;
- with no served ledger, the canonical is the rule's;
- one spelling for every member keeps the canonical (the tie 0041 makes).

Code:
- `kg/entity_clusters`: `build_entity_canonical_map(keep_canonical=)`. `_pick_canonical` prefers a
  kept id, then frequency, the longest name, the name, and the **id**. The id is new, so the pick
  never rests on the input order.
- `identity/ledger.build_ledger`: passes the served ledger's canonical ids.
- A test spy on `_pick_canonical` takes the new argument.

Measured (`d2i9/fixpoint.py`, the "before" ledger served to the rebuild, as on prod):

| | before | after the respelling |
|---|---|---|
| joins | 840 | 840 (1 moved) |
| person groups | 475 | 474 |
| canonical ids moved | | **0**, apart from the one group that merges |
| `display_name` changed | | **0** |
| `display_evidence` changed | | 6 (publisher evidence gained; same names) |
| blocked joins | 9 | 10 (a two-person pair a guard now names; it was not joined before either) |

- The merge: two small groups of one person's spelling variants chain into one through a
  respelled member. The merged-away group's members take the surviving group's spelling. Listed
  for the step-8 read.
- **The id tie-break on a build with no served ledger** (`d2i9/ledger_dump.py`, against 46667141a):
  joins and groups are equal (840, 575); 67 canonical ids differ, all in scoped bare-name groups
  whose members tie on frequency and name, where the old pick took the file order. On prod the
  served D1 ledger keeps every one.
- Unit: identity and KG areas, `921 passed`.

### Part B: migration 0041

**Written beside the code, not first:** the code came before its tests. Every test was then checked
by a mutant, and three that the first mutants survived were strengthened.

Tests (`tests/unit/podcast_scraper/upgrade/test_m0041_joined_names_respelled.py`, on 0025's fixture,
invented names):
- a joined speaker takes the group spelling on every surface (segments, transcript prefixes, turns,
  metadata roster, KG and GI names, diagnostics voices), keeping its id;
  - the spoken words are unchanged, including the name inside them; segments' text and times are
    unchanged; quotes still slice to their text;
  - the roster keeps its order and entry ids; `detected_hosts` and the diagnostics' publisher text
    are unchanged;
  - verify passes after, a second run writes nothing, and undo restores the transcript;
- a mentioned person's KG node and bridge identity are respelled, and the transcript, segments and
  turns are untouched, even where the turns would no longer rebuild;
- a name with an unjoined node is refused and left;
- with no ledger, nothing is written and verify says so;
- item 8: one person listed twice folds into the joined id, the variant's edges moving to it;
  refused when the variant has episodes elsewhere, and when a name is one word (with 0025's
  predicate forced to match one, which it does not today).

Code: `upgrade/migrations/m0041_joined_names_respelled.py` (registered after 0038, `to_version`
2.7.33):
- `episode_renames` (the decision) and `LedgerView`;
- `_Episode` over 0031's derived surfaces, with `_id_map` (folds only), `_record` (merge only on a
  fold or a name now listed twice), `_node_labels` (every joined Person node and bridge identity;
  the publisher hint put back), and `_turns` (only when a speaker label changed).

Mutants, all killed: the id map moving ids by slug; the publisher hint not put back; `_record`
always merging; the elsewhere-episodes refusal off; the one-word refusal off; the unjoined-node
refusal off; mentioned nodes not respelled; turns rebuilt for a mention-only respelling.

### Part C: the ledger step respells

Tests (`tests/unit/podcast_scraper/identity/test_respell.py`, a three-episode corpus):
- a build respells the files and records it in the change log; a second build writes nothing,
  and the ledger is the same (the fixpoint); the run's undo restores the files;
- a held corpus lock defers the respelling, the ledger is still written, and the next build
  catches up;
- a failing respelling never blocks the ledger; the failure is in the change log;
- the CLI respells unless `--no-respell`.

Code:
- `identity/respell.py`: `respell_files` (the corpus lock taken without waiting; 0041's decision and
  rewrite over every served episode, judged against its files; backups and receipts per run under a
  filesystem-safe stamp) and `undo_respell(stamp)`;
- `identity/ledger.py`: `refresh_ledger` builds, respells, then writes; `write_ledger(extra=)`
  records the counts under `respell`;
- `identity/ledger_cli.py`: respells by default, `--no-respell` to skip.

Mutants, all killed: the refresh not respelling; a deferred respelling still writing; a failure
raised instead of counted; the CLI ignoring `--no-respell`.

Unit: identity, upgrade, enrichment, search and KG areas, `2189 passed`; flake8 and mypy clean.

### Found by the first gate run, and changed

The first gate run (`gate0041`, its report discarded) applied 0041 to the clone. Its verify passed,
but 0025's verify failed on 10 episodes instead of item 8's 5 (`d2i9/m0025_after_0041.txt`):
- **One of item 8's 5 folded**; the other 4 remained (below).
- **6 new.** 0041 wrote a group's spelling over a voice whose own episode's publisher text spells
  the person another way, and 0025's rule then renames the voice toward the episode's spelling.
  The two migrations would undo each other.

Three changes, each with tests and a mutant:
1. **0025 defers to the ledger** (`m0025._Episode._ledger_spells`). A placed voice whose every
   Person node carries its group's spelling in the served ledger is not renamed toward the episode's
   stated spelling. The ledger owns a joined person's spelling across episodes: the arc's agreed goal
   is one person, one name, on every surface and across shows. With no ledger, 0025 is unchanged.
2. **0041's fold finds the survivor under a respelled name.** The publisher's entry can match the
   group spelling and not the voice's old one; the voice's id is filed under the old spelling.
3. **The display rule never shows a bare name over a full one** (`ledger._add_display_spellings`).
   The repo's rule (m0017 `display_target`: "a name reduced to one word is never a repair") was not
   applied to the group spelling. One scoped bare id that publisher text names in one episode
   outranked a full name in 26 episodes. Publisher text now picks among the full names when a member
   has one. Measured (`d2i9/spelling_full_name.py`):
   - 5 of 475 groups move;
   - the truth set is unchanged (train 13 of 35, held-out 9 of 33, dropped 5 of 29 wrong, as S6a);
   - of the five, three read as corrections, one as plausible, and one belongs to a join to read at
     step 8.
   - **Titles stay** in a display name, as m0017 decided ("a change that is nothing but a title is
     not a rename").

Quick check after the three (`d2i9/quick_ten.py`, the 10 episodes in a copy-on-write clone): 0025
would change nothing in the 6. Item 8's remaining 4 are not ledger groups. Their ids are unjoined, so
0041 does not touch them. What is left there is 0025's own rule: a placed name takes the feed's
stated host spelling, which is publisher-first, as S6's rule is. They are read at step 8, with
0035's review verdicts on two of them.

**Not built, an open decision for the operator** (the S6 result's "linking the show text to the
group by a fuzzy name match"): the publisher's own spelling, matched to a member by 0025's
predicate. Measured (`d2i9/spelling_publisher_match.py`):
- 7 of 475 groups move, the truth set unchanged;
- 6 of the 7 read as the show's correct spelling of a public figure, and 1 gains a title;
- it would also fix a group whose every member is an ASR misspelling of a host the feed names.

### Found by the second gate run (`gate0041b`), and changed

`upgrade verify` after 0041 failed on 0025 (item 8's 4, below), 0035 (5 episodes) and 0041 itself (1):
- **0035, and every frozen-list migration (0029, 0031, 0033, 0034, 0035, 0037):** each judges a listed
  voice by its exact stored name. 0041 wrote a group spelling over 5 listed names, and 0035 saw neither
  its before nor its done state. A stored name that is the served ledger's group spelling of a listed
  name now reads as that listed name (`m0029.as_listed`, `ledger_spellings`; both decides).
- **0041's one refusal** was m0023's guard: the episode's ad-free segments already did not slice to
  their text, the work of a 2026-10-06 hand repair. Fixed at its own home, migration **0039**
  (record `0039_adfree_segment_offsets_realigned`), registered before 0041. 0041's `to_version` moved
  to 2.7.34.
- **The operator's decision, 2026-11-01:** the publisher's own text is the reference spelling (record
  `s6e_publisher_spelling`). It changes the ledger's spelling of 10 groups, and so what 0041 writes.

**The deploy order this adds.** The ledger CLI now respells. D2's ledger rebuild therefore runs with
`--no-respell`, so the respelling happens inside 0041's own run, with its receipts and the outcome
hook. The ledger step respells only from then on.

**Found at step 14's planning (2026-11-02), and fixed:** the code did not hold to that order.
`refresh_ledger` respelled on every build, and `upgrade run` calls it after any run that applies a
migration, and `enrich` after every run. On prod, the 0039 run, or any enrich before 0041, would
have done 0041's whole respelling outside its measured run, and 0041 would then have found an empty
plan. The clone chains applied migrations directly, never through `upgrade run`, so no gate saw it.
- **The fix:** `identity/refresh_guard.refresh_ledger_after`, used by `upgrade run` and `enrich`.
  While 0041 is not applied it builds and writes the ledger and defers the respelling,
  `{"respell": {"deferred": true, ...}}` in the change log. From 0041 on, it is `refresh_ledger`.
  Tests: `tests/unit/podcast_scraper/identity/test_refresh_guard.py`.
- **Outside the measured code:** the guard is in no D2 migration's import closure (a test pins it),
  so the four code digests stay `bc3c9bb670fd` and the gates and outcomes stand.
- **Not guarded:** `enrich-edges` (`search/cli_handlers`). It is inside the closure and refreshes
  only when a pass rewrites artifacts. The run steps require that no `enrich-edges` pass runs
  between the deploy and 0041's run.

## 7. Measure

The gates and after-checks, newest last.

### Final batch

Gate (`0041_joined_names_respelled.gate.json`, the final batch, code digest `e0bedfef5a62`, 2026-11-01; each gate on its own base: snapshot-prod-20261030 with 0038, the D2 ledger built with the final rules (`--no-respell`), and the migrations before it applied):
- computed class full; gating moves none;
- **573 episodes, 2,864 files**;
- `upgrade verify` **40 ok, none failing** (0025 included: item 8 folded, and the 4 with reviewed
  names left to 0035).

After-checks (`d2i9/after_checks_final.txt`, base-0041 against base-0042, its after-state): **every
must-not-move row passes**, the ledger's included:
- joins 658 → 658 (0 changed); canonical ids, group members, `display_name`, `display_evidence`: 0
  changed; blocked joins 9 → 9;
- spoken text per line, segment text and times, publisher evidence: unchanged; 0 person ids added;
  1 removed (the folded duplicate);
- 25,629 segment labels and 107 transcripts respelled;
- search: 107 episodes with changed chunks (1,700 of 2,757 chunks) re-embed on the next reindex;
- the 2 quotes off their text are among 0040's 6 listed, unchanged by 0041;
- 1 episode's shown names move from an authored quote name to the group spelling (intended).


### Rerun after s5d (the ledger's near-spelling rule and reviewed list)

0041's code is unchanged. Its input changed: D2's ledger after record s5d (sections 4–4d), 632
joins instead of 658. The gate and the after-checks were rerun on the same chain of bases, each
ledger rebuilt with `--no-respell`.

**Gate** (`0041_joined_names_respelled.gate.json`, code digest `2b6374388bbc`, 2026-11-01T21:50):
- computed class full; gating moves none;
- **556 episodes, 2,805 files.** That is s5d 4d's prediction (`d2i9/s5d_4d_evidence.py`) exactly.
  - The 17 fewer episodes come from the joins s5d removed: its two-people pairs, whose spelling
    0041 would have written over another person's mentions, and the one unclear member
    (`person:yu-shang`).
- `upgrade verify` **40 ok, none failing**.

**After-checks** (`d2i9/after_checks_s5d.txt`, base-0041 against base-0042): **every must-not-move
row passes**. The script's verdict line reads "AFTER CHECKS: FAIL 0". It prints FAIL whenever a
quote span is off its text, here the 2 below, and 0 is the count of must-not-move breaches.
- **The ledger fixpoint:**
  - joins 632 → 632, 0 changed;
  - blocked joins 39 → 39;
  - canonical ids gone or new, group members, `display_name`, `display_evidence`: 0 changed.
- Spoken text, segment text and times, publisher evidence: unchanged.
- 1 person id removed (the folded duplicate). 25,629 segment labels and 107 transcripts respelled.
- Search: 107 episodes with changed chunks (1,700 of 2,757), re-embedded on the next reindex.
- The 2 quotes off their text are among 0040's 6 listed.

**Bare speaker labels kept** (`d2i9/placeholder_labels.py`): **4** episode-scoped names that are
speaker labels in their segments. Each keeps the episode's bare name in the files, and read time
shows its group's spelling. All 4 read right: one show's two co-hosts in three episodes, and one
analyst's surname.
- The earlier figure, 15 (4f-b), was measured on the ledger before s5d with `d2i9/dry_counts.py`.
  That script joins the metadata's transcript path, which is absolute on prod, so it missed
  episodes.
- The two counts differ in both ledger and method, and the difference is not traced by id.

### Final rerun on D2's final ledger (s5d through 4e, s6g, s6h; 2026-11-02)

Code digest `bc3c9bb670fd` (the shared digest moved with s6h's `identity/ledger` and `identity/display`). Chain eval-data `d2out/scripts/chain5.sh`, each gate on its own base: snapshot-prod-20261030 with 0038, the ledger rebuilt `--no-respell` with the final rules, the migrations before it applied. The gate report is from this run. Record s6h section 7 has the whole chain.
- **Gate:**
  - **467 episodes, 2,289 files**, as s6h's 4b predicted. s6g's run gave 501 / 2,525 (record
    s6g section 7); s6h's 12 groups with no group spelling are respelled no more.
  - names refused 179 (173 episode-scoped, 6 a fold needs two full names), folds 1;
  - `upgrade verify` 40 of 40 ok, none failing; no gating moves.
- **After-checks** (eval-data `d2out/after_checks_s6h.txt`): 0 must-not-move breaches.
  - The verdict line reads "FAIL 0". FAIL is printed for the 2 quote spans off their text, both
    among 0040's 6 and byte-identical before and after (`d2out/quote_spans_both.out`).
  - joins 631 → 631, blocked 40 → 40, `display_name` and `display_evidence` changed 0, members and
    canonical ids 0.
  - 1 GI person id removed and one episode's shown names moved: the fold. Its 65 quotes now show
    the group's spelling (`d2out/shown_moved.out`). This row was 1 in every run since s5d, and the
    section above did not list it.
  - 84 transcripts and 19,563 segment labels respelled; 84 episodes with changed chunks (1,273 of
    2,136).
- **Bare speaker labels kept** (`d2i9/placeholder_labels.py` on this base →
  `d2out/placeholder_labels_s6h.out`): **2**, one name in two episodes. Read time shows its group
  spelling. The section above counted 4 on the s5d ledger; the difference is not traced by id.

## 8. Verdict

**Builder's verdict, before the review (2026-11-02): GO WITH CONDITIONS.** 467 episodes, 2,289 files, 179 refused, 1 fold, as s6h predicted; 40 of 40 verify ok; after-checks 0 must-not-move breaches; second apply 0; undo byte-identical (2,289 files). Gate report on committed code `b64e08e04` (clean, code digest `bc3c9bb670fd`, worktree run eval-data `d2out/scripts/chain7.sh`), equal in outcome to the final rerun in section 7. Second apply and full-copy undo: record d2_name_consolidation, Undo.
Conditions: `make ci-fast` and `make test-integration` green on the committed tree (plan step 13);
at the apply, prod's dry-run plan diffed against the clone's (method step 10: prod has 43 episodes
the snapshot lacks).

Verdict read 2026-11-02 by a fresh agent (reviewer, Sonnet; not the builder, not a step-5 reader), inputs as file paths (brief eval-data `d2out/scripts/step8_prompt.txt`), about 10 minutes for all four, ran nothing over the corpus. Its block, condensed:

```text
REVIEW 0041_joined_names_respelled 2026-11-02 commit b64e08e04 reviewer Sonnet (fresh) minutes 4
PREDICTION vs GATE: 11 rows compared (s6h 4b, the latest prediction), 11 equal
REQUIRED ITEMS: 0 episodes refused, 179 names refused (counted; the plan itemises 30); code clean b64e08e04; undo recorded (2,289 restored, byte-identical); second apply recorded (0); census vs gate suffixes differ: written .txt / .adfree.txt / .cleaned.txt 86 / 86 / 81 against the after-checks' changed 84 / 84 / 79, unexplained; not-covered present; ci-fast log missing
AFTER-STATE READ: the fold's 65 quotes read right; the 2 bare labels read right; the step-5 spot checks (item 8's five, one episode end to end in the player, one reindexed episode's hits) not present as reads
OPEN CALLS: the 4 vs 2 bare labels not traced by id (accepted); the refused names itemised at the prod apply; frozen copies keep the old spelling (disclosed)
DOUBTED NUMBER: the 2-per-kind transcript gap (86 vs 84); particular to this change; one read of the receipts with the operator's go
VERDICT: GO WITH CONDITIONS  every dated prediction row met; conditions ci-fast and test-integration, and the transcript-count gap explained
```

Conditions, each answered:

1. ci-fast and test-integration on the committed tree: plan step 13, next. **Open.**
2. The transcript-count gap (written 86 / 86 / 81, after-checks changed 84 / 84 / 79): not yet
   explained. The re-measure is one read of 0041's receipts against the after-checks' episode
   loop. It walks no corpus and takes seconds. It needs the operator's go (the method's
   recalculation rule). **Open.**
3. The step-5 spot checks as reads (one episode end to end in the player, one reindexed
   episode's hits): the player read needs the deployed stack. They are made on prod after the
   apply (step 10 of the method), where the reindex runs. **Open, at the apply.**

Verdict: GO WITH CONDITIONS

- **Condition 1:** `make ci-fast` and `make test-integration` green on the committed tree (plan step 13).
- **Condition 2:** The transcript-count gap (86 / 86 / 81 written, 84 / 84 / 79 changed) explained, with the operator's go for the read.
- **Condition 3:** At the apply, prod's dry-run plan diffed against the clone's, and the spot checks read on prod after the reindex (method step 10).

Outcome: `src/podcast_scraper/upgrade/outcomes/0041_joined_names_respelled.json`, sha256
79274a8794f82d042c173eaddb01216e7664f94d05d2b97dfa7c64573bae5c35 (`scripts/measure/build_outcome.py`, from the gate report measured on
b64e08e04 and this verdict).

**Condition 2, answered after the verdict (operator's go, 2026-11-02: "I take your
recommendations"):** the 2-per-kind gap is 2 episodes whose metadata stores an absolute
transcript path. My after-check script skips such an episode (eval-data `d2out/gap_0041.py` →
`.out`).
- The two are substack `deda024b` "0005" and simplecast `999571cd` "0007".
- The after-checks' must-not-move rows, run on both runs' episodes with the transcript resolved
  by name (`d2out/gap_0041_checks.py` → `.out`, "FAILS 0"):
  - publisher evidence equal;
  - spoken text equal on `.txt`, `.adfree.txt` and `.cleaned.txt`;
  - segment text and times equal;
  - 0 GI quotes off their text (224 and 79 checked).
- The gap is in my measuring script, not in 0041.

## 10. Before apply / deploy

- **Prod's plan against the clone's:** 479 episodes (the clone 467), 183 names refused (179). The
  outcome hook's plan diff: 14 episodes, each given a cause from prod's rows and the snapshot:
  - 10 were ingested after the snapshot;
  - 1 differs only in the ledger file's sha among its inputs;
  - 3 come from joins prod's newer episodes made possible (`donald-j-trump`, `robert-carrow`,
    `louis-guerra`).
  - In all 4 rows checked, every file sha equals the snapshot's.
  - The accept file is on prod: `.podcast_scraper/upgrade-outcomes/0041_joined_names_respelled.accept.json`.
    It is also in eval-data `d2out/accept_0041_plan_diff.json`, without the go.
- **Run steps, as run on prod (2026-11-02, each on the operator's go typed in prod's session):**
  1. deploy-all-prod on `sha-8906696` (run 41489217128, both live smokes).
     - The operator waived Docker Build & Test and CodeQL Advanced, which did not run on 890669690:
       their path filters skip a test-only commit.
     - `deploy.py verify`: VERIFY PASS, 10 of 10 containers.
  2. The ledger rebuilt with D2's rules, `identity-ledger --no-respell` (pipeline-llm,
     18:53–18:56Z): 644 joins (the clone 631), 40 blocked (the clone 40). Its 5 canonical changes
     are the clone's.
  3. 0039, 0040, 0041 and 0042, one corpus-migration job each, in order.
     - One pre-apply snapshot, before 0039; the other three ran with `--no-snapshot`.
     - The ledger refresh after each run deferred the respelling until 0041 was applied
       (`identity/refresh_guard`).
  4. `enrich`, then the incremental `index` (not `index-two-tier`, which drops the index), both in
     pipeline-llm.
  5. The bounded verify of 0039–0042, the near-spelling report, and spot checks through the API.

## 11. After apply / deploy

Prod, 2026-11-02, each step on the operator's go typed in prod's session.

- **Apply** 20:10:29–20:39:55Z, exit 0, `--no-snapshot`, `--accept-plan-diff` (operator_go "Go"):
  "rewrote 479 episode(s)", with the dry run's counts.
  - The outcome hook: verdict PASS, failures none, clone_deltas and prod_deltas empty.
  - Now at 2.7.35.
  - The upgrade process peaked at 450 MB RSS.
- **The first real respell,** the refresh after it: 0 files written, as predicted.
- **Verify** `--only 0041`: OK.
- **Ownership:** FAIL on 5 root-owned backup directories under `upgrade-backups/0041/…/transcripts`.
  - The cause is m0023's `_write_text_with_backup`, which does not chown the directories it creates
    (handover item).
  - They were chowned on the operator's go; the recheck gave PASS.
- **Derived surfaces:**
  - `enrich` 21:12:54–21:29Z: exit 0, 9 enrichers, insight_density 481 of 481; the refresh after it
    respelled 0;
  - incremental `index` 21:30:39–22:44:27Z: exit 0;
  - the 2026-11-03 nightly ran after it, exit 0.
- **Bounded verify** (`upgrade verify --only 0039,0040,0041,0042`, capped container, after the
  nightly): all four OK.
- **Near-spelling report:** 15 lines, 11 pairs, each read as one person in prod's transcripts. The
  0 expected was the clone's: prod's ledger has joins the clone did not. The verdicts went into the
  reviewed list for the next deploy.
- **Spot checks** (the API, not the player UI):
  - a respelled episode's search hits and speaker labels read right;
  - an empty `person_profile` (`person:robert-caro`) is empty on the D1 snapshot too: a separate
    bug, not D2.
- **Rollback:** this migration's receipts and backups under `.podcast_scraper/upgrade-backups/`. The
  pre-apply snapshot and the rollback image holds were removed on the operator's go, 2026-11-03.
- **Operator accepted:** 2026-11-03.

Status `landed` (2026-11-03): applied on prod, so its outcome is no longer compared with later code.

## Not covered

- Frozen copies (highlights, favourites, sent emails).
- The ledger pooling one bare first name across a show's episodes (103 groups on the snapshot):
  found while measuring this, reported to the operator, not changed here.
