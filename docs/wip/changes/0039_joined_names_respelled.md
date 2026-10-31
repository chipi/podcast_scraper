---
id: 0039_joined_names_respelled
class: full
status: open
issue: none (RFC-131 plan V3, D2 items 8 and 9; operator, 2026-10-31)
touches:
  - src/podcast_scraper/upgrade/migrations/m0039_joined_names_respelled.py
  - src/podcast_scraper/upgrade/registry.py
  - src/podcast_scraper/identity/respell.py
  - src/podcast_scraper/identity/ledger.py
  - src/podcast_scraper/kg/entity_clusters.py
---

# 0039 — a joined speaker's name in the episode files is the group's one spelling, and stays so

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
3. **Migration `0039_joined_names_respelled`:** the first full run over the corpus. Plan, apply
   with backups and receipts (`joined_names_respelled.jsonl`, tag `0039`), verify (nothing left to
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

- **Episodes written by 0039:** 658, plus any of item 8's 5 not among them; the dry run's count is
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
- **A second run** of 0039 and of the ledger step writes nothing.
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
- Predicted: the reindex after 0039 touches the episodes 0039 writes and no others. Measured on
  the clone with the fixture-free count of changed rows per episode (step 7).
- Between the migration and the reindex, a transcript hit's lift reads offsets that no longer
  match the file: stated, and the window is the run's own (minutes).

**The ledger fixpoint, measured before any code** (`d2i9/fixpoint.py`: the ledger rebuilt with its
own inputs respelled as 0039 leaves them, publisher text kept):

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

## 5. Acceptance

Fresh reader, 2026-10-31, a sonnet reviewer subagent with read-only tools, about 1.5 minutes, before
any code: "not complete". Every row is taken in 4b, and two were measured before answering (the
index fingerprint, the ledger fixpoint). The fixpoint measurement added the canonical-id stability
to the change.

Spot checks for step 8: item 8's five, as listed in 4b; the chained member and the removed pair;
one respelled episode read end to end (transcript, segments, player view); one reindexed
episode's hits.

## 6. Build

### Part A: the ledger keeps its canonical ids

Tests, written first (`tests/unit/podcast_scraper/identity/test_ledger.py`, invented names; the first
and third failed before the code):
- a rebuild that shifts the frequencies keeps the served canonical id;
- with no served ledger, the canonical is the rule's;
- one spelling for every member keeps the canonical (the tie 0039 makes).

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

## Not covered

- Frozen copies (highlights, favourites, sent emails).
- The ledger pooling one bare first name across a show's episodes (103 groups on the snapshot):
  found while measuring this, reported to the operator, not changed here.
