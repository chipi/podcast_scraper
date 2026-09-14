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

## Step 1 — Deploy

Images publish only from `main`, and one tag pins all three services. This fixes every future
ingest and changes nothing already stored.

## Step 2 — Migrate the corpus (fixes class A)

```bash
make upgrade-corpus CORPUS_DIR=<prod corpus>
```

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

1. Re-run the migration. Expect **0 promoted / 0 demoted** — it is idempotent, so anything else
   means step 3 changed rosters and you should re-read the diff.
2. Re-run the coherence check over the corpus (`kg.speaker_coherence.check_corpus`). Violations
   should approach 0. That is the pass/fail signal, not the migration's own counters.
3. Rebuild the search index — step 3 regenerates insights. `make index-two-tier-docker` (the host
   target cannot run on an Intel Mac: no x86_64 ML wheels).

---

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

Step 2 is in-place but snapshotted: restore the pre-upgrade snapshot directory the run reports.
Step 3 regenerates artifacts from audio-derived data that is not itself modified, so re-running it
is safe; there is no snapshot for it, so take one first if the corpus state matters.
