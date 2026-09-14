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
make upgrade-status CORPUS_DIR=<prod corpus>          # confirm pending is EXACTLY [0008, 0009]
make speaker-migration-preview CORPUS_DIR=<prod corpus>
```

An unstamped ledger would run all nine migrations, including the 0002 Lance index rebuild that has
never been exercised on this corpus. And read the SUSPECT / AMBIGUOUS lists before step 2, not
after.

## Step 1 — Deploy

Images publish only from `main`, and one tag pins all three services. This fixes every future
ingest and changes nothing already stored.

## Step 2 — Migrate the corpus (fixes class A)

```bash
# --snapshot-dir MUST be a persistent path. The default is a sibling of the corpus root, which is
# container-ephemeral on a volume mount — the rollback story is only real if the snapshot outlives
# the container (advisor S8). Capture the path it prints.
make upgrade-corpus CORPUS_DIR=<prod corpus> SNAPSHOT_DIR=<persistent path>
```

Confirm no ingest job is running (`GET /api/jobs`) before snapshotting: a mid-ingest snapshot is an
inconsistent one.

Reads `content.speakers` and rewrites Person roles in `kg.json`. Takes a pre-upgrade snapshot by
default and **aborts before touching anything if the snapshot fails**, so it is recoverable — keep
the snapshot until step 4 passes.

Expected (287-artifact sample):

```
promoted 380 · demoted 21 (17 not-a-person, 4 roster-denied) · 239 artifacts written
roles   mentioned 89.5% → 68.1%   host 9.9% → 19.7%   guest 0.6% → 12.2%
coherence  82 violations → 56     26 FIXED, 0 NEW
```

Idempotent — a second run writes nothing (verified byte-identical).

**Sanity check before trusting it:** run `--dry-run` first and read the demotion split in the
summary line. If `not a person` is ~80% of demotions, that matches the sample. If the
`roster accounts for every voice without naming` number is large, stop and look at them
individually — that is the route that can, in principle, unseat a real speaker.

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

Verified on the 287-artifact staging copy: 401 changes applied, 401 restored, 0 refused, corpus
sha256 **byte-identical** to before the migration.

Undo REFUSES any node whose current role is not the `role_after` the ledger recorded — that means
something else wrote it since (a `relabel_only` re-enrich, a later migration, a previous undo), and
replaying over it would overwrite newer and better work. Refused nodes are listed, not forced. This
is also why a second undo is a harmless no-op rather than a corruption.

**That changes the decision.** Running the migration is no longer a one-way door: the question is
"is the damage bounded and recorded", not "can we prove zero damage in advance". Read the SUSPECT
and AMBIGUOUS lists first anyway — the ledger makes a mistake recoverable, not free.

### Step 2 snapshot (still recommended)

`upgrade run` snapshots the whole corpus before any migration, and aborts if the snapshot fails.
That covers migrations with no ledger; the role ledger is the finer-grained undo for m0009
specifically. Keep both until step 4 passes.

Step 3 regenerates artifacts from audio-derived data that is not itself modified, so re-running it
is safe; there is no ledger for it — take a snapshot first if the corpus state matters.
