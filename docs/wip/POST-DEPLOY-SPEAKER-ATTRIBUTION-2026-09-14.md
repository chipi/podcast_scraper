# Post-deploy: repairing host/guest attribution on production

**Context:** PR #2065 fixes host/guest attribution at cause. This note is what to run on prod
*after* that deploys, and what each step can and cannot fix.

**Written 2026-09-14.** Numbers below are measured on a 287-artifact production staging copy, not
on the live corpus. Re-measure before trusting the counts.

---

## Why the deploy is not the end of it

Two different things are broken and they need different repairs.

- **The pipeline** was computing the wrong answer. PR #2065 fixes it — but only for episodes
  ingested *from now on*.
- **The artifacts already on disk** have the wrong answer stored. Code cannot reach back and
  change them.

And the stored artifacts are broken in two distinct ways, which is why there are two repair
passes and not one:

| | what is wrong on disk | repair | cost |
|---|---|---|---|
| **A** | the roster in `content.speakers` is CORRECT; the graph never received it | migration m0009 | no LLM, no GPU, minutes |
| **B** | the roster is **itself wrong** (`['Host']`, an org, a show name) | `pipeline_stage=relabel_only` | LLM for the GI/KG cascade; no ASR, no GPU diarization, no audio download |

A migration cannot fix class B. There is no correct answer on disk for it to copy.

---

## Step 0a — FIRST: measure the idx-collision damage (#2082)

**Operator decision 2026-09-16: the order is deploy -> REPAIR #2082 -> quality work.** Not deploy
-> quality. Every quality metric here reads `content.speakers`, and on an affected episode that
field describes a different episode — so quality measured before the repair is partly measuring the
wrong episodes, and a quality change cannot be told apart from the contamination. Measurements
already taken during the #2075 arc should be re-derived after repair, or restricted to episodes this
audit clears.


**Run this before anything else, and do not run a migration until you have the number.** It costs
one read-only pass and it decides whether the rest of this document is measuring a corpus or a
corrupted one.

On a reprocess, `idx` used to come from the episode's filename prefix. Every run directory numbers
from `0001`, so a feed with fourteen run dirs has fourteen "episode 1"s, and `idx` keys per-episode
state and output filenames — so they collided. Measured on the 2026-09-14 snapshot:

```
metadata files pointing at ANOTHER episode's transcript   275 of 2,256  (12.2%)
  of those, roster matches the WRONG transcript             119
  roster matches its own                                      0
  inconclusive                                              146

repairable in place (own transcript still on disk)           42
needs re-download + re-ASR + re-diarize                     233
```

119 episodes credit their quotes to people from a different episode — "DHH's new way of writing
code" is credited to Addy Osmani; "How Kent Beck shapes the software engineering industry" to Grady
Booch. Those names reach `content.speakers`, the KG `Person` nodes, `SPOKEN_BY` and search.

```bash
make transcript-pairing-audit CORPUS_DIR=<prod corpus>
```

Read-only. **Exits 1 when any mismatch is found**, so it can gate this step. It reports three
verdicts, because they need different work:

| verdict | meaning |
| --- | --- |
| `roster_matches_wrong` | the speakers are in the transcript it points AT and not its own — **confirmed misattribution** |
| `roster_matches_own` | the pointer is stale but the roster is right |
| `inconclusive` | names in neither or both — usually a roster built from metadata. **NOT proven safe**, just unclassified |

and two repair routes: `repairable` (own transcript still on disk — scoped `relabel_only`, no
audio, no GPU) and the rest (needs re-download + re-ASR + re-diarize).

Do not quote the confirmed count as a total until the unclassified ones have been looked at.

Why this comes first:

1. **Every measurement below is taken from `content.speakers`.** On an affected episode that field
   describes a different episode, so the coherence before/after number, the migration previews and
   the "roles look like this afterwards" tables are all reading corrupted rows until this is known.
2. **m0009 promotes and demotes roles from those same rosters.** Running it first bakes the wrong
   attribution deeper and makes the damage harder to see.
3. **The 42 repairable ones are cheap** — a scoped `relabel_only`, no audio, no GPU. Do them before
   the migrations so the migrations see the corrected rows.
4. **The 233 are not cheap** and are a separate decision: their transcript does not exist anywhere
   in the corpus, because `relabel_only` overwrites the transcript it picks, so where episode A was
   relabelled onto B's file, A's was never written and B's was overwritten with A's labels. Only a
   full re-ingest recovers them. Size that GPU bill deliberately; do not let it start by accident.

The code fix (unique per-run `idx`, `Episode.on_disk_idx` kept only for the legacy search) is on
`fix/duplicate-people-variant-resolution` and stops NEW damage. It repairs nothing already written.

## Step 0 — Pre-flight

```bash
make upgrade-status CORPUS_DIR=<prod corpus>
make speaker-migration-preview CORPUS_DIR=<prod corpus>
make speaker-coherence CORPUS_DIR=<prod corpus>          # record the BEFORE number
```

**EXPECT ALL NINE MIGRATIONS PENDING, not two.** An earlier version of this file said "confirm
pending is exactly [0008, 0009]". That is wrong: production carries **no `upgrade_ledger.json` at
all**, so `upgrade status` reports `Applied: none` and all nine pending, at version `2.7.0.dev0`.
The artifacts are nonetheless already modern — kg schema `2.0`, gi schema `3.1`, m0007 scoped ids
present — because the pipeline writes those natively and re-ingestion satisfied the migrations
without recording them. Rehearsed against the real `snapshot-prod-20260914` corpus (2,257
artifacts) on 2026-09-14.

What each pending migration actually does on that corpus:

| migration | effect |
| --- | --- |
| 0001, 0004 | no-op |
| 0002 | **no-op on prod** — skips when a healthy index with its sidecar exists. It reports a build locally only because the backup deliberately prunes `search/` as regenerable |
| 0003, 0005, 0006 | genuine no-ops — 2,257 already-current |
| **0007** | **rewrites 153 episodes** (18 healed, 255 scoped) — see the numbers below, this is not a formality |
| **0008** | **rewrites all 2,257 artifacts** for a one-character change — see below |
| **0009** | 3,177 promoted, 195 demoted, 1,844 artifacts |

## Step 1 — Deploy

Images publish only from `main`, and one tag pins all three services. This fixes every future
ingest and changes nothing already stored.

## Step 2 — Migrate (fixes class A)

```bash
# SNAPSHOT_DIR must be a persistent path — the default is a sibling of the corpus root, which is
# container-ephemeral on a volume mount. Capture the path it prints.
make upgrade-corpus CORPUS_DIR=<prod corpus> SNAPSHOT_DIR=<persistent path>
```

Confirm no ingest job is running (`GET /api/jobs`) first: a mid-ingest snapshot is inconsistent.

### 0008 is the BIGGEST write and does the LEAST work — expect it

It stamps `schema_version` `2.0` -> `2.1` and changes nothing else. Readers already accept both
(`kg/schema.py`: `_ACCEPTED_SCHEMA_VERSIONS = {"2.0", "2.1"}`), so nothing is gated on it and it
fixes no violations. It is still correct to run — the stamp is what makes the corpus honestly say
it can hold `Object` nodes (#2057) — but know two things before watching it:

- it **rewrites every artifact in the corpus** (2,257) for a one-character change, which is real
  I/O and a full-corpus churn in the next backup snapshot;
- it invalidates the entire `corpus_delta` fingerprint set, so the NEXT pipeline run re-upserts all
  2,257 episodes into the search index instead of skipping unchanged ones. Budget for that on the
  DGX; it is a cost, not a defect.

Work done per artifact written, across the three:

| migration | artifacts written | violations fixed |
| --- | ---: | ---: |
| 0007 | 153 | 2,327 |
| 0008 | 2,257 | 0 |
| 0009 | 1,844 | 227 |

### 0007 does more good than 0009 on the coherence measure — do not skip it

Measured on the real corpus, in order:

```
coherence violations   3,783   before anything
                       3,556   after m0009 alone     (-227)
                       1,229   after m0009 + m0007   (-2,554 total, -67%)
```

The 2,702-violation class m0009 cannot touch is `quote_two_speakers` — a quote pointing at BOTH a
bare id and its episode-scoped twin (`person:selena` and
`person:unresolved-selena-<episode>`), which is bare-name-scope residue, not a role problem. 0007
is the migration for exactly that.

### What the roles look like afterwards

| role | after | before the arc |
| --- | ---: | ---: |
| mentioned | 67.6% | 89.5% |
| host | 21.5% | 9.9% |
| **guest** | **10.9%** | **0.6%** |

That guest column is the operator's original report, fixed at production scale.

### Two classes needing a human, from the real preview

- **49 SUSPECT demotions** — the node had a voice, which is also what an eponymous-show host looks
  like. All 49 on this corpus are genuine show names: `Machine Learning Street` ×29,
  `Conversations with Tyler` ×18, `Turkey Book` (feed *Turkey Book Talk*), `Africa Tech Summit`.
  **If a real person appears in this list, stop.**
- **2 AMBIGUOUS nodes**, left untouched and both correctly refused: `William de Rimpel` (an ASR
  mangling of *William Dalrymple*, but the roster also holds a bare `person:william`) and
  `Jill Lepore` (the roster holds `jill-laporte` AND `jill-lapour`). Guessing either would have
  been wrong.

## Step 3 — Re-label the rest (fixes class B)

`relabel_only` re-resolves speaker NAMES on the **frozen** diarization and cascades GI/KG. No
audio download, no re-ASR, no re-diarization.

```
POST /api/jobs?pipeline_stage=relabel_only[&feed=<feed>][&profile=<dgx profile>]
```

It gets `--reprocess-existing-only` automatically, so the corpus can only shrink, never grow.
Point `profile=` at a DGX profile so the GI/KG cascade runs on local vLLM rather than a paid API.

**Scope it.** After step 2, 50 of 287 sampled episodes (17.4%) across 18 feeds still violate
coherence. Worst offenders in the sample:

| episodes | feed |
|---|---|
| 6 | The China-Global South Podcast |
| 6 | Biz Talk |
| 5 | The Daily |
| 5 | Colombia Calling |
| 5 | Round Table China |
| 4 | Conversations with Tyler |
| 3 | The Naked Pravda |

Run per-feed on the affected feeds rather than the whole corpus.

> `reprocess_dgx_no_llm` is **not** a cheaper substitute here — it sets `generate_metadata: false`,
> which skips the very stage the roster lives in.

## Step 4 — Verify

Every step here is a runnable command. An earlier version of this section named a function with no
CLI and told you to "re-run the migration", which the upgrade ledger blocks — it was a description,
not a procedure (advisor S5).

```bash
# What m0009 would do, bypassing the ledger. Run this BEFORE step 2 as the pre-flight, and again
# after step 3 to confirm it has nothing left to do.
make speaker-migration-preview CORPUS_DIR=<prod corpus>

# Is the corpus self-consistent now?
make speaker-coherence CORPUS_DIR=<prod corpus>
```

**Read the preview's two human classes before running anything.** They exist because the
predicates are imperfect and say so:

- **SUSPECT demotions** — the node being demoted also had a voice. That is exactly what a host of
  an eponymous show (`Lex Fridman Podcast`, `Rich Roll Podcast`) looks like. On the 287-artifact
  sample all 6 are genuine show names; on a corpus containing such a feed, the host would appear
  here. **If a real person is in this list, stop.**
- **AMBIGUOUS nodes** — matched more than one roster entry, so they were left untouched. 0 on the
  sample; the guard is for the 85% of production not sampled.

**Read the ROLE TRANSITION table, not the violation count.** The coherence checks share the
migration's own predicates, so a wrongful demotion scores as a violation **FIXED**, not introduced.
`82 -> 56 violations` cannot detect the damage this migration is most likely to cause. The
transition table reports what changed per node:

```
mentioned -> guest      195
mentioned -> host       185
host      -> mentioned   21     <- the only destructive direction
guest     -> mentioned    0
```

Expected on the sample: 380 promoted, 21 demoted. Anything in `guest -> mentioned`, or a demotion
count far above 21, means look before proceeding.

**Restart the API after step 2.** Not optional. The migration rewrites `*.kg.json` and the upgrade
ledger; `perf_cache.corpus_mtime` now includes the ledger so the mtime-tokened projections do
invalidate, and `corpus_graph` / `cil_queries` now carry the token too — but a restart is still the
only thing that guarantees every in-process cache is gone.

**Reindex only if needed.** `relabel_only` reindexes incrementally on its own (advisor S7): the
finalize path calls `maybe_index_corpus`, and `prod_dgx_full.yaml` sets `vector_search: true` with
no skip flag. Check `vector_index_seconds` in the run summary; a full `make index-two-tier-docker`
is only needed if that is 0.

## Known limits — read before declaring victory

- **An org in the host seat whose name is not the show's** survives step 2 entirely: `China Plus`,
  `Mercatus Center at George Mason University`, `Brandon Anderson, RJ Honicky, and Latent.Space`.
  No predicate can tell these from a person. They are the bulk of the 56 remaining violations and
  they need step 3.
- **39 roster names have no matching node** in their episode's graph — near-miss spelling variants
  (`bernt børnich` vs `bernt bornich`). The migration reports them and deliberately does not insert
  a node, because that would put the same human in the graph twice. Needs step 3 or #2056's
  variant resolver.
- **Whether a `relabel_only` run rebuilds the search index by itself is UNVERIFIED.** Assume not;
  do step 4.3 explicitly.
- **The 17.4% figure is from the 287-artifact sample.** Prod is ~678 episodes / ~953 artifacts and
  the real proportion has not been measured.
- **The full `make upgrade-corpus` chain was never run end to end during development** — migration
  0002 rebuilds the Lance index and needs `sentence_transformers`, unavailable on the dev Intel
  Mac. m0009's `apply` was driven directly with a real `MigrationContext` and `dry_run=False`.
- **Three feeds will light up the `empty_host_anchor` rework queue at once, and that is the fix
  working.** The feed-host work may now only REMOVE a host, so 63 episodes across The Rest Is
  History (`Norman Conquest`, 84 voices), Latin America in Focus (`Americas Online`, 32) and
  Colombia Calling (`Emily Hart`, 8 — a wrong name on Richard McColl's voice) move from a junk host
  name to no host at all. 126 files, 63 episodes; 20 other episodes GAIN a named host in the same
  change. Nothing regressed: every name lost there was junk, an organisation, or the wrong person.
  Whoever watches that queue after deploy should not read the spike as new damage.
- **A relabelled episode's record used to keep the OLD names.** Until `51a202bb` no reprocess stage
  rewrote `content.speakers` — verified by diffing before/after on two DGX episodes, byte-identical.
  Every episode repaired by a relabel BEFORE that fix therefore has a transcript with new labels and
  a record with old names, and every number read from such a record describes the pre-repair state.
  The same defect wrote a SECOND record into the reprocess's own fresh run directory, which then
  made `rederive_only` refuse the episode. Both are fixed; the corpus cleanup is tracked on #2097.

## Rollback

**The role changes are reversible (#2069).** m0009 writes `speaker_roles_ledger.json` at the corpus
root recording every `(episode, node, role_before, role_after, route)` transition, and:

```bash
make upgrade-undo-roles CORPUS_DIR=<prod corpus>              # replay every role_before
python scripts/ops/undo_speaker_roles.py --corpus-dir <c> --show   # read it, change nothing
```

Verified twice. On a 287-artifact staging copy with nothing else touching it: 401 changes applied,
401 restored, 0 refused, corpus sha256 **byte-identical**.

And on the real `snapshot-prod-20260914` corpus (2,257 artifacts), migrations run in the ORDER a
real `upgrade run` uses — 0007, 0008, 0009:

```
ledger rows : 3,370
restored    : 3,370
skipped     :     0
refused     :     0
```

A clean, complete rollback at production scale.

**An earlier version of this section claimed 54 refusals as "the realistic case". That was wrong,
and the error was mine:** I had rehearsed 0009 BEFORE 0007, so 0009 recorded hashes that 0007 then
invalidated. In the real order 0009 runs last and its hashes are current. Re-measured above.

The refusal path is still real and still matters — but the thing that triggers it is **step 3**, a
`relabel_only` re-enrich, which rewrites kg.json after the migration. That is why the rule below
is: undo BEFORE step 3, or not at all. When it does refuse, it refuses correctly: a role-only
check would have seen matching roles, demoted everything, reported zero refusals and called it a
clean rollback while destroying the newer work.

**Restart the API after an undo, exactly as after the migration — and do not rely on the token
here.** The undo un-records 0009 from `upgrade_ledger.json`, which moves the `perf_cache` token,
but only when it actually rewrote something: `record_reverted` returns False without writing if
nothing was restored, or if 0009 was not in `applied` to begin with. A run that restores 0 rows
therefore leaves the token where it was. That is the correct behaviour — nothing changed — but it
means "the undo moved the token" is not something to assume. Restart.

**The undo now REFUSES rather than racing.** If an ingest or a migration holds the corpus lock,
`undo_speaker_roles.py` prints `REFUSED: …` and exits 1 having changed nothing. Stop the other job
and re-run. Previously it detected the contention and then proceeded unlocked, which is the one
case the lock existed for. `upgrade run` takes the same lock in the forward direction, around
**both** the pre-upgrade snapshot and the migrations — the snapshot is minutes of `copytree` and
is the rollback you are being handed, so it must not be copied out from under an ingest.

**What that lock does and does not cover, precisely:**

- **Covered:** a multi-feed ingest (`cli.py`, `service.py`) — the shape prod runs — and a
  concurrent `upgrade run` or undo. All take the same parent lock, so contention is detected.
- **NOT covered:** a single-feed ingest with `single_feed_uses_corpus_layout`, which locks
  `<root>/feeds/<slug>` instead — a different lock file. A migration and such an ingest can still
  run concurrently with no refusal. Prod's schedule is multi-feed, so this is a gap on paper today.
- **Expect the reverse, too:** a scheduled ingest that STARTS during the migration window now
  fails loudly rather than interleaving (the CLI raises, `service.run` returns `success=False`).
  That is the intended behaviour — **re-trigger the ingest after the migration finishes.**
- **Verify on the box:** if `PODCAST_SCRAPER_CORPUS_LOCK=0` is set in the deploy environment, every
  lock above is inert. Check before relying on any of this.

**`make upgrade-verify CORPUS_DIR=…` now means something for 0009.** It checks the ledger's rows
against the artifacts: `401 of 401 recorded role(s) still present` after a run, `0 of 401` after an
undo. Previously it returned "no verification defined".

**So: undo BEFORE step 3, or not at all.** After a re-enrich the ledger's episodes are refused by
design, and that refusal is correct — the re-enriched answer is the better one.

Three outcomes, not two: `restored` / `skipped` (already back where it started) / `refused`
(something else owns that file now). The CLI exits non-zero when anything is refused, because a
partial rollback must not read as a complete one.

**Byte-identity has a scope.** A kg.json last written by a MIGRATION comes back byte-identical. One
last written by the PIPELINE (a new ingest, or a `rederive_only` since the last migration) uses a
different serialiser — those files are refused by the sha check anyway, so the undo never touches
them.

**That changes the decision.** Running the migration is no longer a one-way door: the question is
"is the damage bounded and recorded", not "can we prove zero damage in advance". Read the SUSPECT
and AMBIGUOUS lists first anyway — the ledger makes a mistake recoverable, not free.

### Step 2 snapshot (still recommended)

`upgrade run` snapshots the whole corpus before any migration, and aborts if the snapshot fails.
That covers migrations with no ledger; the role ledger is the finer-grained undo for m0009
specifically. Keep both until step 4 passes.

Step 3 regenerates artifacts from audio-derived data that is not itself modified, so re-running it
is safe; there is no ledger for it — take a snapshot first if the corpus state matters.
