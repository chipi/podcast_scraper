# Handover: v2.5 corpus (DGX-local naming LLM) + the naming/labeling arc

**Author:** Claude (session 2026-07-30, post-#1355 merge)
**Purpose:** a cold-start plan for a fresh session. Read this top-to-bottom; it
carries the north star, the ordered plan, the decisions already locked, and the
open ones. Nothing here needs re-derivation — it's a seed, refine as you go.

---

## Where we are (baseline just landed)

- **PR #1355 merged to main** as squash commit `b24608fa` — the v2.4 arc:
  naming-4 speaker labeling, GI route-and-tag (3.1) + KG Voice node,
  per-episode observability manifest. Closes #1191, #1220, #1337 + the 7
  retrospective issues #1358–#1364.
- **The v2.4 corpus's naming/summarization LLM is Gemini** (cloud provider).
- Main CI on `b24608fa`: Python application + Docker + CodeQL + Snyk + MkDocs +
  obs-image-publish green; **Stack test** fires via `workflow_run` after Python
  application and must finish green. (Confirm before starting new work.)
- Local branches cleaned (feat/v23-turbo-asr + acl/litellm-gateway-ports deleted,
  both merged). This worktree parks on detached `origin/main`.

## North star — the 2.5 corpus

A corpus with **v2.4's pipeline and quality**, but the **Gemini** naming /
summarization LLM replaced by a **DGX-hosted local model**, kept **as close to
Gemini quality as possible**. Everything below is the path to freezing 2.5.

## Decisions locked (operator, 2026-07-30)

1. **2.5 model = decided by the loop.** Do NOT pre-commit the DGX model. The
   autoresearch prompt-tuning loop (Phase D) bakes off candidates
   (Cohere Command / Qwen3-30B-A3B / DeepSeek — all in the roster) and picks the
   winner. "column-based command-free" ≈ Cohere Command, but it's a candidate,
   not a given.
2. **2.5 ship gate = judge-panel parity.** A cross-vendor judge panel scores the
   DGX output vs the Gemini baseline; 2.5 ships only at statistical parity.
   Panel must be **disjoint-vendor** (silver + judge from a vendor NOT in the
   cohort — see `feedback_silver_judge_vendor_bias`), **scalar mode** (not
   pairwise — `feedback_scalar_over_pairwise_for_judge_trust`), and the score
   parser must strip `</think>` (`feedback_judge_reasoning_block_parsing`).
3. **Prompt-tuning precedes the swap.** Tune the naming/resolver LLM query
   per-provider (incl. DGX) BEFORE swapping, because the swap itself will force
   more prompt fine-tuning. This ordering is deliberate.

## The ordered plan

### Phase 0 — Baseline green (this session)
- [ ] Confirm main Stack test green on `b24608fa`. If red, fix at cause first —
      nothing else starts until main is green through stack-test.

### Phase B — Close tonight's gap (tiny, do first)
- [ ] **#1360 regression test.** The fix (str-coercing `run_id`/`episode_id`/
      `trace_id` in `CorrelationFormatter.format`, `utils/correlation.py`) landed
      at cause + verified by the full suite, but has **no dedicated test**. Add a
      ~10-line unit test: a non-string (e.g. a `Mock`) episode context → the
      stamped record field is a `str`. Honors the bug→repro-test rule.

### Phase A — Naming / labeling arc, top-to-bottom
The highest-value continuation of naming-4. Suggested order (bugs → recall →
heavy):
- [ ] **#1226** — merged diarization cluster claiming a host's name adds an
      uncapped 3rd host (bug)
- [ ] **#1228** — correctly-named co-host demoted to guest when the feed doesn't
      establish the host set (bug)
- [ ] **#1169** (memory thread) — roster host↔guest swap in guest-heavy
      interviews (intro-window argmax); was queued AFTER #1167
- [ ] **#1192** — name the ~113 truly-unknown panel-guest tail (recall lever
      beyond the intro)
- [ ] **#1286** — guest recall via re-diarization (corpus-v2.2)
- [ ] **#1170** (memory thread) — diarization handover (RTTM truth, count+DER
      scorer; Real-10 deferred; DGX `~/diar-sweep`)
- [ ] **#1285** — canonicalize ASR-garbled names in transcript BODY text
      (corpus-v3; heavier, do last)

### Phase C — Selected platform threads
- [ ] **Docs-hygiene audit** — OVERDUE (was flagged PRIORITY 2026-07-29). Plan at
      `docs/wip/DOCS-HYGIENE-AUDIT-PLAN.md`; ~439 docs; classify → approve →
      execute, UNCERTAIN stays. See `project_docs_hygiene_audit` memory.
- [ ] **Go-live (Goal-1)** — PAUSED on a native SIGSEGV (`project_goal1_golive_status`).
      NOTE a SIGSEGV cluster is now open: **#1323** (MiniLM/MPS 3rd loader),
      **#1345** (GlitchTip SIGABRT), and the search-indexing one isolated in
      **#1362**. Decide whether these gate go-live and/or the DGX indexing path.

### Phase D — Autoresearch prompt-tuning loop (the pivot, BEFORE the swap)
Tune the **naming / speaker-resolver LLM query** per-provider and pick the 2.5
model. This is the methodological heart of 2.5.
- [ ] Stand up the bake-off in the autoresearch harness (`autoresearch/`,
      `JUDGING.md`, `bundled_prompt_tuning/`).
- [ ] Candidates on DGX vLLM (:8003): Cohere Command, Qwen3-30B-A3B-Instruct-2507,
      DeepSeek — per `autoresearch/PER_MODEL_OPTIMAL_PARAMS.md` (consult BEFORE
      and update AFTER each sweep). Deep-research each model's serving knobs
      first (`feedback_deep_research_per_model`).
- [ ] Metric = **judge-panel parity vs the 2.4-Gemini baseline** (see locked
      decision 2). Baseline = the naming-4 relabel defect/named metrics from
      #1355 + the judge panel.
- [ ] Output: (a) the winning DGX model, (b) tuned per-provider prompts for the
      naming/resolver query, (c) the recorded parity result.

DGX access reminders: `gpu-mode-swap.sh research` (NEVER `code`/coder-next —
`feedback_never_use_coder_next`); ACL permits only :8003 / :11434
(`project_dgx_tailscale_acl`); model size class = TOTAL params, A3B active ≠
substitute (`feedback_model_size_class`).

### Phase E — Provider swap (Gemini → DGX-local)
- [ ] Wire the winning model via the existing DGX profiles: `prod_dgx_balanced`,
      `prod_dgx_full_with_fallback`, `cloud_with_dgx_primary` (config/profiles/).
- [ ] Expect iterative prompt fine-tuning as real output diverges from bake-off.
- [ ] Re-validate with the naming-4 harness + judge panel each iteration.

### Phase F — Freeze the 2.5 corpus
- [ ] Full reprocess under the DGX-local profile.
- [ ] Confirm judge-panel parity with 2.4 holds on the full corpus (ship gate).
- [ ] Commit all; write the 2.5 ADR (decision + alternatives + the parity
      evidence). Then next things.

## Assets you'll need
- **Profiles:** `config/profiles/prod_dgx_*.yaml`, `cloud_with_dgx_primary.yaml`,
  `experiment_dgx_*.yaml`, `local_dgx_*.yaml`.
- **Autoresearch:** `autoresearch/JUDGING.md`,
  `autoresearch/PER_MODEL_OPTIMAL_PARAMS.md`, `autoresearch/bundled_prompt_tuning/`.
- **naming-4 harness:** shipped in #1355 (the relabel defect/named/tape metrics).
- **Golden fixtures:** #1189 (one per show, real diarization + hand-labelled
  truth) — feeds the parity gate if judge-panel is complemented by human-GT.
- **Memory:** `project_autoresearch`, `project_autoresearch_programme`,
  `reference_per_model_optimal_params`, `reference_dgx_judge_deployment_quirks`,
  `project_dgx_vllm_distinction`, `reference_pyannote_cache_offline`.

## Open decisions (for the fresh session to resolve)
- Which Phase-C threads are on the critical path to 2.5 vs parallel/optional.
- Does go-live have to precede 2.5, or run in parallel?
- Is the SIGSEGV cluster (#1323/#1345/#1362) a hard prerequisite for the DGX
  indexing path or independent?
- Does the parity gate use judge-panel ONLY, or judge-panel + human-GT
  (#1189) as a second signal?

## Start here (fresh session)
1. Confirm main is green through Stack test on `b24608fa`.
2. Do Phase B (#1360 test) — warms you into the code.
3. Begin Phase A at #1226/#1228.
4. Keep Phase D's methodology in mind while doing A — the naming code you touch
   in A is the same code the DGX swap will re-tune.
