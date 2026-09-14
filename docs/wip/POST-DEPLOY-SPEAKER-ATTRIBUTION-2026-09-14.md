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
| 0008 | stamps all 2,257 as schema 2.1 |
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

## Rollback

**The role changes are reversible (#2069).** m0009 writes `speaker_roles_ledger.json` at the corpus
root recording every `(episode, node, role_before, role_after, route)` transition, and:

```bash
make upgrade-undo-roles CORPUS_DIR=<prod corpus>              # replay every role_before
python scripts/ops/undo_speaker_roles.py --corpus-dir <c> --show   # read it, change nothing
```

Verified twice. On a 287-artifact staging copy with nothing else touching it: 401 changes applied,
401 restored, 0 refused, corpus sha256 **byte-identical**.

And on the real `snapshot-prod-20260914` corpus with 0007 run afterwards — the realistic case:

```
ledger rows : 3,372
restored    : 3,277
refused     :    54     <- the episodes 0007 rewrote after m0009
```

Those 54 refusals are CORRECT and are the whole point of the file-hash predicate. 0007 rewrote 153
episodes after m0009 wrote them; a role-only check would have seen matching roles, demoted all of
them, reported zero refusals and called it a clean rollback — destroying 0007's repair. The
rollback is partial, and says so; the CLI exits non-zero.

Undo refuses an episode whose FILE sha256 differs from what the migration wrote — not merely a node
whose role changed. Role-equality is blind to "something rewrote this file and happened to agree",
and that is the likely sequence here, not a corner case: **step 3 is a re-enrich**, `rederive_only`
and `rediarize_only` cascade to GI/KG, and the rebuilt graph now reads the roster — so it writes
`host` for most of the same nodes m0009 promoted. A role-only check would demote all of them,
report zero refusals, and call it a clean rollback.

**Restart the API after an undo, exactly as after the migration.** The undo un-records 0009 from
`upgrade_ledger.json`, which moves the `perf_cache` token — but the token-less in-process caches
are only guaranteed gone on a restart.

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
