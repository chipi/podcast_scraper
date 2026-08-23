# Batch implementation plan — corpus-derivations + enrichment reliability + polish

**Started:** 2026-08-23 · **Source analysis:** `docs/wip/ARCH-CORPUS-DERIVATIONS-REFACTOR.md`
(twice advisor-verified, Fable 5). **Rule:** each item lands with its tests; run the item's test
target green before commit. Most items are `src/` → need the rebuild cycle to DEPLOY, but code +
tests are written and validated locally now. Box/CI-only items deploy without a rebuild.

Ordering follows the advisor: fix the **meta-defect (honest coverage) first** — it's what let a
partial enrichment report green and hid everything else.

Legend: **[R]** = needs image rebuild to deploy · **[B]** = box-deployable (no rebuild) ·
**[CI]** = CI/repo only. Status: ☐ todo · ◐ in-progress · ☑ done.

---

## STATUS — COMPLETE (2026-08-23)

All items landed on branch `hotfix/deploy-all-permissions` with tests; a four-reviewer deep
review + verification pass followed and its real findings were folded in. **Not deployed / not
pushed** — the `[R]` items deploy together in one rebuild cycle (operator-gated). Commits:

| Item | Commit | Item | Commit |
|---|---|---|---|
| E1 | `f299f5634` | E4 | `f21fe87cb` |
| C1 | `781b80b88` | E5 | `e636854e6` |
| #1810 (B3) | `5d8bfa475` | #1808 (B2) | `f6423cce4` |
| E2 | `f5c709b06` | P1/P2 | `33075c132` |
| #1809 (B1) | `5d2644fbb` | P3 | `c1af2343b` |
| #1789 (B5) | `5181feef5` | review-fixes | (this commit) |
| E3 | `5569106de` | | |

- **#1757 (B4)** — verified ALREADY DONE in the codebase (`93ecaa65e`); not re-implemented.
- **P4** — deliberately deferred (keep the box-log tees; revisit after Alloy capture is proven).
- **NOTE:** line/symbol refs in the item bodies below are PRE-FIX addresses (the files were
  restructured by the fixes). Read the commits above for the final code, not these line numbers.

---

## Phase 1 — Enrichment reliability (the 5-defect slice, #1811)

### E1 ☑ [R] Honest coverage — an enabled enricher that didn't run must NOT report `ok`
- **Why:** `registry.list_enabled` warn-skips leave NO row in `run_summary` and the run still says
  `status=ok` (`registry.py:75-96`, `run_summary.py:58-100`; `executor.py:861-880` only fixed the
  zero-enricher case). This let the partial reenrich report green.
- **Change:** every ENABLED-but-not-run enricher gets a per_enricher row with an explicit reason
  (`not_registered` / `not_admitted` / `timeout` / `failed`); the run aggregates to a non-`ok`
  status (`partial` or `failed`) when any enabled enricher is missing/failed.
- **Tests:** unit — (a) enabled+unregistered ⇒ row present + status≠ok; (b) all-ran ⇒ ok;
  (c) admitted-but-timeout ⇒ row + non-ok. Target: `tests/unit/**/enrichment/` executor+run_summary.
- **Validate:** the targeted pytest module green; mypy on touched files.

### E2 ☑ [R] Ship `gate_metrics.json` into the image (or fix `_default_eval_root`)
- **Why:** `.dockerignore:41` excludes `data/`; confirmed `/app/data/eval NOT in image` →
  `topic_consensus` (`on_missing_data="reject"`, `topic_consensus.py:129-132`) self-rejects
  in-container forever. `_default_eval_root` (`admission.py:122-124`) resolves to a site-packages
  path with no data.
- **Change (pick one, advisor to confirm):** ship the enrichment `gate_metrics.json` files into the
  wheel/package (MANIFEST/package_data) OR make `_default_eval_root` resolve a known in-image path.
- **Tests:** unit — admission finds the gate data at the resolved root; `topic_consensus` admits
  when metrics present. Integration if feasible.

### E3 ☑ [R+B] `reenrich-prod` `--with-ml` + survivable ML timeout + HF cache
- **Why:** ML enrichers need `--with-ml` (the CLI's designed contract; auto-path passes it). But the
  180s/120s `expected_duration_s` won't survive 678 episodes + a cold model download; no HF cache
  volume today.
- **Change:** [B] add `--with-ml` to `reenrich-prod.yml`; [B] add an HF cache volume to pipeline-llm
  in the prod compose; [R] raise/scale the two ML manifests' `expected_duration_s` (or a
  corpus-scope timeout override) — advisor to size it.
- **Tests:** manifest duration unit test; `actionlint` on the workflow; compose config validates.

### E4 ☑ [R] Enqueue-never-persists (defect 2) — root-cause THEN fix
- **Why:** pipeline logs "enqueued corpus_enrichment" but 0 rows persist (box-confirmed). Root cause
  open (not the coalesce theory — 0 queued to coalesce into).
- **Change:** get the next reprocess's job log (`orchestration.py:1923-1937` enqueued-vs-could-not);
  fix the actual cause. NO speculative fix.
- **Tests:** unit — enqueue persists exactly one queued row; regression for the found cause.

### E5 ☑ [B] Pause / staleness alert
- **Why:** `.viewer/jobs.paused` blocks drain with a 30s INFO nobody sees; 8-day-stale enrichment
  went unnoticed.
- **Change:** o11y alert — enrichment `run_summary` older than newest gi/kg mtime, and drain paused
  &gt; N min. VictoriaLogs/Grafana rule (box/homelab config).
- **Tests:** N/A (config) — validate the rule fires against a synthetic stale sample.

## Phase 2 — Topic clustering (Problem 1)

### C1 ☑ [R] scipy UPGMA swap + deterministic ordering + skip-gate + guardrail
- **Why:** O(n³) hand-rolled agglomerative merge hangs whole-corpus (`topic_clusters.py:257-305`).
- **Change:** replace `cluster_indices_by_threshold` internals with scipy
  `linkage(method="average", metric="cosine")` + `fcluster(criterion="distance", t=1-threshold)`
  (same signature); **deterministic cluster ordering** before slug assignment (prevents `tc:` id
  churn breaking stored per-user interests); a **skip-gate** (fingerprint topic rows; skip if
  unchanged → `skipped_unchanged` logged). Declare `scipy` in `[search]` extra.
- **Tests:** (a) partition equivalence vs the old algorithm on the validation fixture
  (`docs/wip/wip-topic-clusters-validation-reference.yaml`); (b) deterministic slugs across runs;
  (c) skip-gate skips when unchanged; (d) **Tier-2 wall-clock guardrail** row (ADR-095) — clustering
  on a corpus-scale fixture completes under a bound (regression guard for the O(n³) return).
- **Validate:** targeted pytest; confirm no schema change (consumers untouched).

## Phase 3 — Batch-B (related rebuild-gated issues)

- **B1 ☑ [R] #1809** — Deepgram retries billed per attempt, priced once (ledger undercounts). Tests: ledger counts each billed attempt.
- **B2 ☑ [R] #1808** — audio sweep cost never measured. Tests: sweep emits a cost figure.
- **B3 ☑ [R] #1810** — nothing stops two pipeline runs sharing one corpus. Tests: concurrent-run guard.
- **B4 ☑ [R] #1757** — cost cap didn't fire in a prior run. Tests: cap refuses before spend at the boundary.
- **B5 ☑ [R] #1789** — audio provenance only stamped by `archive backfill`, not download/reprocess. Tests: reprocess stamps provenance.
- (#1752 no-op run is subsumed by E1 honest-coverage + delta reporting — verify, don't re-fix.)

## Phase 4 — Polish (folded in per operator, 2026-08-23)

- **P1 ☑ [B]** `push.sh` — add `container_cpu_percent{app,name,box}` so a hung container's CPU is visible BY NAME in VictoriaMetrics (hang detection). Validate: run push.sh, metric appears.
- **P2 ☑ [B]** Alloy `operator.alloy` — extract `run_id` (and episode id when present) from pipeline log lines into `structured_metadata` (like the existing `trace_id` stage), so VictoriaLogs is filterable by run/episode. Validate: config reloads; a run's lines carry the label.
- **P3 ☑ [CI]** `check_prod_secret_staging.py` — make it proximity-aware (a re-stage must precede EACH post-session-boundary container-creation), so it can't false-green like it did for D5. Tests: the gate flags a stage-once-create-twice workflow.
- **P4 ☐ [B, optional]** Drop the box-log tees in `reprocess-prod.yml`/`sweep-prod-audio.yml` — ONLY if we decide the Alloy capture fully replaces the "logs survive the kill" net. Default: keep (defensive). Revisit.

---

## Execution notes
- Local test env: no complete `.venv` in this worktree; reuse a sibling repo venv
  (`podcast_scraper-infra/.venv` has pytest/mypy) for deterministic tests; ML-dependent tests may
  need `[ml,search]` — flag per item.
- One commit per item (or tight sub-steps), tests green before commit, never push without approval.
- The `[R]` items deploy together in one image rebuild → Stack test → deploy at the end of the batch.
