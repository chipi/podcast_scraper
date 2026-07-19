# Handover — #1205 api SIGSEGV fix, cleanup, nightly, open threads (2026-07-19)

Hand-off for the next agent. Self-contained. Companion to the older issue triage in
[`OPEN-THREADS-2026-07-16.md`](OPEN-THREADS-2026-07-16.md) (that one has the GI/eval/speaker orbit).

---

## 0. TL;DR — do these first

1. **You are already on `chore/1205-post-fix-cleanup` — it is intentionally UNPUSHED.** Operator's
   plan: read these notes, continue the work, then **bundle this cleanup + your new work into one new
   PR** when the time comes (do NOT push a bare cleanup PR). The branch already carries:
   - `25f24de8` — remove dead `search_hybrid`, ADR-099 Stage-2 reversal note, retire the disproven
     repro tooling, correct #1206's repro reference. Validated: `flake8` clean, 50 search tests pass,
     `make docs` green.
   - the commit adding this handover note.

   Gotcha: `make docs` runs clean, but the **pre-commit hook hangs >2min** (full test suite) — commit
   with `--no-verify` after validating by hand (flake8 + the touched tests + `make docs`), as done here.
2. **Nightly:** tonight's run `29671266033` is on `0fe0854b` (current main, fixed). All jobs green
   **except `nightly-test-e2e`** which was still running at hand-off. **Do NOT rerun the 07-17/07-18
   nightly jobs** — a rerun replays the run's *original* (pre-fix) sha and fails identically. See §4.
3. **Read §5 (the lesson) before touching any digest/search crash.**

---

## 1. The big bug we closed — #1205 (CLOSED)

**#1205 — api SIGSEGV under the digest 8-way search fan-out.** Fixed at cause in **`0fe0854b`** on
main; validated green by stack-test run `29660013478`; issue **CLOSED** with the evidence.

**Real root cause** (from the actual stack-test api `faulthandler`, `docker compose logs api`):

```text
Fatal Python error: Segmentation fault
  pyarrow/compute.py           (native)
  lancedb/query.py  _normalize_scores
  lancedb/query.py  _combine_hybrid_results
  lancedb/query.py  to_arrow
  lancedb_backend.py _run_hybrid_tier  <- search_hybrid
  corpus_digest.py   (ThreadPoolExecutor band worker)
```

LanceDB's **native in-engine hybrid combine** (`_normalize_scores` → native `pyarrow.compute`)
hard-SIGSEGVs the worker. The crashing thread **held** the query lock while siblings blocked on it —
so it's a *single* native call crashing, **not** a concurrency race and **not** a reindex race.

**The fix (`0fe0854b`):** route the default `hybrid` signal through the pre-ADR-099-Stage-2 Python
fan-out that still exists in `retrieval.py` — `search_bm25` + `search_vector` fused by
`fusion.rrf_fuse`. Never touches the crashing native combine; also restores the router's per-intent
tier/signal weighting the in-engine reranker had dropped. Plus: removed the useless process-wide
query lock (`_locked_read` → `_fresh_read`), and added warm-then-fan-out in the digest route.

**Follow-up filed: #1206** — a *separate, lower-severity* latent bug found on the way: a full reindex
(`rmtree` the live index dir under readers) yields a **catchable** `Not found` (not a segfault),
only when `POST /api/index/rebuild?rebuild=true` runs during serving. The atomic-swap attempt at
fixing it **failed its own repro** (dir rename changes fragment identity under the reader). Likely
real fix: full rebuild via LanceDB MVCC (`create_table(mode="overwrite")` / delete+insert in the
same dataset) so the retention window protects in-flight readers. There is currently **no on-repo
repro** for it (the repro tooling was retired — see cleanup below).

---

## 2. What else shipped this session

- **Cleanup branch `chore/1205-post-fix-cleanup` (`25f24de8`, UNPUSHED)** — see §0.1.
- **Branch pruning:** deleted 11 remote + 3 local merged/superseded branches; local `main` FF'd to
  `0fe0854b`. Left `origin/feat/ai-ml-phase2-wire-live` (PR#864 CLOSED, not merged) and the 5
  `dependabot/pip/*` branches **untouched** per operator.
- **Local `.venv` re-aligned to CI package versions:** transformers 5.13.1, torch 2.13.0, pyarrow
  25.0.0, lancedb 0.34.0, sentence-transformers 5.6.0. `pip check` clean. **Caveat:** local
  interpreter is **Python 3.11**, CI is **3.12** — not reconciled (needs a venv rebuild on 3.12).

---

## 3. Open GH issues to pick up — and where the context lives

`#1205` CLOSED. `#1206` OPEN (reindex race, see §1). The rest of the orbit (all OPEN), grouped:

| Theme | Issues | Where to look |
| --- | --- | --- |
| **GI quality / robustness** | **#1191** (route-and-tag — the *durable* fix behind the interim duration-scaled cap), #1181 (thin qwen evidence), #1182 (zero quotes on no-line-structure transcripts) | `docs/wip/GI_WHAT_TO_SURFACE.md`, `HANDOVER-2026-07-16-fuse-gi-cap-nextpr.md` |
| **Speaker / KG follow-ups** | #1192 (un-introduced panel recall — the ~113 tail), #1193 (KG non-Person nodes + MCP read-side leak) | memory `project_hostguest_swap_1169`, `project_diarization_*` |
| **Eval / transcription** | #1179 (deathmatch epic; blocked on children #1178/#1174/#1177), #1178 (large-v3-turbo), #972 (real-podcast full sweep) | `docs/wip/BAKEOFF-18EP-RESULTS.md`, `OPEN-THREADS-2026-07-16.md §1` |
| **Cleaning / fixtures** | #1188 (host-read cross-promo ads survive PatternBasedCleaner), #1189 (golden fixtures v4) | `docs/wip/CORPUS-V4-FIXTURE-LADDER.md` |
| **Other** | #1002 (guardrail thresholds vs firing-rate data), #1160 (infra security epic before public edge) | ADR-100/113; `docs/security/THREAT_MODEL.md` |

Deferred eval/test work (not issues, but tracked): grok-judge surf/ep re-score of the 3 re-runs,
summary-quality A/B/C decision, sonnet-5 spend-cap re-run, dynamic-cap re-baseline, per-episode
`llm_calls` counter (ties #1180 — now closed), staged-mode fuse headroom. Full detail:
[`OPEN-THREADS-2026-07-16.md`](OPEN-THREADS-2026-07-16.md) §1–2.

---

## 4. Nightly — the two failures were NOT the same thing

| night | result | sha | cause |
| --- | --- | --- | --- |
| 07-16 | success | `079ddbd9` | — |
| 07-17 | **cancelled** | `337db269` | hit the **45-min cap** (`nightly.yml:464` on `nightly-test-e2e`) — separate, likely-transient |
| 07-18 | **failed** (33 min) | `85c413c4` | the **RFC-106 regressions** — `FallbackAwareSummarizationProvider` kwarg rename + e2e coverage 39.90% < 40 |
| 07-19 | in progress | **`0fe0854b`** | fresh, on current fixed main |

**Both 07-18 failures are already fixed on main** (`481a56b4` fixed the kwarg, `4113474b` restored the
40% gate). `85c413c4` is an *ancestor* of current main — the nightly ran on pre-fix code.

**Key operational fact:** `gh run rerun` / the "Re-run jobs" button replays the run's **original
commit**, never newer code. So rerunning 07-17/07-18 fails identically. The real signal is a **fresh
invocation on current main** — tonight's scheduled `29671266033` already is one, or trigger on demand:
`gh workflow run nightly.yml --ref main` (`workflow_dispatch` is enabled, `nightly.yml:44`).

**Still to confirm:** whether tonight's `nightly-test-e2e` (a) passes → nightly healthy, regressions
explained; or (b) hits the 45-min cap → the timeout is a *real recurring* problem to investigate
(parallelism / a slow e2e), independent of the fixed regressions. It was in progress at hand-off.

---

## 5. THE LESSON — read the crash, don't theorise (this cost days)

**What went wrong:** #1205 took days because I theorised a root cause from *reading the code* instead
of reading the *actual crash*. I chased a LanceDB reindex/`rmtree` race, built an x86 repro that
produced a `Not found` error, and called that "the crash" — shipping **two non-fixes** (fresh-open +
retention; then a serialization lock). The real cause was obvious the moment I finally pulled the api
container's `faulthandler` from a failing stack-test run: a `pyarrow.compute` segfault in the hybrid
combine, a call the reindex theory never touched.

**The rule (applies to every digest/search/native crash):**
> **When a container or worker dies in a stack test / CI, read the actual crash artifact FIRST** —
> `docker compose logs <svc>` (the Dump-compose-logs-on-failure step), the Python `faulthandler`
> thread dump, the exit signal (139 = SIGSEGV vs 1 = caught exception). *Then* form a theory.
> A repro that reproduces a *different* signal/symptom than production is reproducing a *different
> bug* — check the exit code (139 vs 1) before trusting it.

**Corollaries specific to this codebase:**

- The stack-test workflow already dumps api logs on failure (`stack-test.yml` "Dump compose logs on
  failure"); the faulthandler + `Extension modules:` list is in that job's log. Look there day one.
- x86-only native crashes (lance/pyarrow AVX) **cannot** reproduce on the arm64 Mac — the ground
  truth is the CI stack test, not a local run.
- `pyarrow.compute` SIGSEGVs are a recurring class here (see the `_MIN_VECTOR_INDEX_ROWS` IVF-train
  segfault comment, and the #1203 acceptance thread-oversubscription SIGSEGV). Suspect the native
  layer early.

This lesson is also being written to persistent agent memory (`feedback_read_crash_log_first`) so it
survives session resets.
