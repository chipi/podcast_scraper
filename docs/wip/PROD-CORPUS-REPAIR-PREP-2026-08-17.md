# Production corpus repair — prep sheet (2026-08-17)

**What this is:** the concrete pre-flight for running
[CORPUS_INTEGRITY_REPAIR_RUNBOOK.md](../guides/CORPUS_INTEGRITY_REPAIR_RUNBOOK.md) against the
**production** corpus. The runbook is the procedure; this sheet is what must be true before it
starts, what is already proven, and what is still unknown.

Written after #1661 merged (`f6c77fcd`) + hotfix `7127e396`.

---

## Operator decisions already made

| Decision | Value |
| --- | --- |
| **Step 6 ASR budget** | **APPROVED** — spend on Deepgram, or run ASR on DGX. 2026-08-17. |
| Missing ffmpeg | FATAL, never a silent degrade (#26) |
| Missing optional ML package | DEGRADES, loudly, with a ledger row (#1661) |
| Aged-out episode fix | PARKED — measured, not needed (see below) |

---

## Preconditions — ALL must hold before step 1

- [ ] **The merged code is deployed to production.** `f6c77fcd` + `7127e396` are on `main` but a
      deploy has NOT been run. Without it production is still running the pre-fix pipeline and a
      repair would be pointless — worse, it would re-damage what it repairs.
      Deploy is `workflow_dispatch` on `deploy-all-prod.yml`, confirmation string
      `DEPLOY_ALL_PROD`, rolling all three planes in order.
- [ ] **The image carries `git_sha`.** ADR-132's exact-code backstop only exists if the image was
      built with the `GIT_SHA` build arg (#30). Verify inside the deployed image before trusting
      any provenance the repair writes.
- [ ] **`DEEPSEEK_API_KEY` is set** (task #23). Every configured failover ladder points at the
      deepseek tier; without the key the ladder detects failures and recovers nothing. A startup
      pre-flight prints `FAILOVER LADDER BROKEN` at every run start until it is set. Compose
      already forwards the variable — only the value is missing.
- [ ] **A corpus snapshot/backup exists.** Steps 3 and 6 write in place.

---

## Pre-flight measurements — read-only, run these FIRST

These answer the questions nobody can answer from here. None of them writes anything.

```bash
make corpus-gi-integrity-check   CORPUS_DIR=<prod-corpus>   # step 1 baseline
make corpus-preprocessing-audit  CORPUS_DIR=<prod-corpus>   # step 2 baseline
```

Record both verbatim. Steps 8–9 compare against them, and a count that does not move is the
failure mode the whole runbook exists to prevent.

**Open questions these settle:**

| Question | Current status |
| --- | --- |
| Are ~112 placeholders still in production? | **UNVERIFIED.** The figure predates the epic. Step 1 answers it. |
| How much of production was transcribed from unpreprocessed audio? | **UNVERIFIED.** The 60 % figure is from a *local pre-fix* corpus (9 of 15 runs), not production. Step 2 answers it. |
| Will step 6 silently no-op on aged-out episodes? | **MEASURED, no.** 9/9 of the damaged work-list still served by their feeds; archives run 71–2950 items. Re-run `scratchpad/check_feeds.py` against the *production* work-list before step 6 — production may hold older episodes. |

---

## What is proven, and where

Validated locally on real corpora with the merged code:

| Runbook step | Evidence |
| --- | --- |
| 1 — GI integrity gate | PASS on `pipeline-run/corpus-out`: 40 corpus members from 76 metadata files — the membership rule correctly excluded 36 superseded runs |
| 2 — preprocessing audit | Found **9 damaged runs** on `podcast-acceptance-corpus` (the pre-fix corpus) |
| 4 — work-list | 9 episode ids written, production-shaped (UUIDs + `substack:post:` ids) |
| 3 — `gi-repair` | Proven earlier in the epic (1 → 4 insights, gate FAIL → PASS). **Cannot be re-rehearsed locally: zero placeholders exist in any local corpus.** |
| 6 — re-transcribe | **NOT rehearsed.** Needs ASR; no whisper locally and it costs money. First real exercise will be production. |
| 7–9 | Not rehearsed — depend on 6 |

---

## Step 6 — the command, and the two traps in it

```bash
podcast-scraper --config <profile> --feeds-spec <corpus>/feeds.spec.yaml \
  --output-dir <corpus> --skip-existing --single-feed-uses-corpus-layout \
  --no-transcript-cache \
  --reprocess-episode-ids <corpus>/preprocessing_repair_worklist.txt
```

1. **`--no-transcript-cache` is not optional.** The cache key is the original media hash plus
   `preprocessing_fingerprint(cfg)`, and that fingerprint is computed from *config* — it reads
   `pp=on` whether preprocessing succeeded or fell back to raw audio. Neither key component
   changes between the damaged run and the repair run, so without this flag step 6 scores a cache
   hit and re-serves the exact transcript it was launched to replace, and step 8 goes green on
   unrepaired data. (Entries written *since* #1661 are safe — a run that falls back to raw audio no
   longer writes a cache entry at all. This flag covers everything written before.)
2. **`--single-feed-uses-corpus-layout` is required** or cross-run resolution never fires and every
   episode reports "no transcript".

`--reprocess-episode-ids` implies `--reprocess-existing-only`. Before that implication existed, a
one-episode work-list preprocessed **12 unrelated episodes** before being killed.

---

## Step 8 — a green audit is NOT sufficient

The preprocessing audit's damage rule is `completed < attempts`, and it reports `attempts == 0` as
*not damaged* — correctly, since a run that never attempted preprocessing damaged nothing. But a
step-6 run served entirely from cache **also** records `attempts: 0`. It is indistinguishable from
healthy while having repaired nothing.

So assert positively, against the run dirs step 6 created:

```bash
find <corpus> -name metrics.json -newermt '-1 hour' -exec sh -c '
  echo "$1: $(jq -c "{attempts: .preprocessing_attempts, completed: .preprocessing_count,
                      transcribed: .transcribe_count}" "$1")"' _ {} \;
```

Expected: `attempts >= 1`, `completed == attempts`, `transcribed >= 1`. All zeros means the cache
served it.

---

## Known limits going in — say these out loud rather than discover them

- **Step 6 has never been run.** Its first execution will be against production. Use
  `--max-episodes` for a cautious first pass.
- **A reprocess does NOT fix placeholder episodes.** Four flag combinations were rehearsed and
  every one skipped them: the skip predicates key on file *presence* and never look at GI. That is
  what `gi-repair` (step 3) exists for, and why it rewrites in place.
- **`make corpus-placeholder-check` is not a valid exit criterion.** It asks only "is the bad
  string absent?", which PASSES on a corpus whose artifacts were deleted and never regenerated.
  Use `corpus-gi-integrity-check`.
- **`metrics.json` is run-level.** A one-episode run attributes exactly; a multi-episode run can
  only say "this run has damage". Per-episode attribution needs the #22 ledger row, which by
  construction exists only on runs made *after* that change — never on the damaged ones.
- **Zero-insight artifacts are legal now.** 112 placeholders quietly becoming 112 *empty* artifacts
  would satisfy a naive check while having re-derived nothing. Step 1 reports that count
  separately — read it.
