# Handover — v2.5 corpus phase (DGX-local LLM swap), then the v3+10k backlog

**Author:** Claude (session 2026-08-02, post docs-hygiene + arc reconciliation)
**Branch:** `feat/naming-arc-and-corpus-prep` (pushed; tip after this doc). Not a PR yet.
**Purpose:** cold-start plan for a fresh session. Read top-to-bottom; nothing here needs
re-derivation. **Priority order is deliberate: finish the ORIGINAL v2.5 arc (Part 1) FIRST,
then take on the NEW backlog items (Part 2).**

Canonical arc plan: `docs/wip/1000-EPISODES-REPROCESS-PLAN.md` (the single north-star doc —
v2.5 is its current stage). Backlog map: `docs/wip/OPEN-BACKLOG-v3-10k.md`.

## North star

**2.5 corpus = v2.4's pipeline/quality, with the Gemini naming/summarization LLM replaced by a
fully-local DGX model, shipping only at judge-panel parity vs the 2.4-Gemini baseline.**
v2.1–v2.4 are MERGED (#1335, #1355). v2.5 (the LLM swap) is the last single-variable step before
the frozen scale run.

## Decisions locked (operator)

1. **Model = decided by the bake-off loop** — do NOT pre-commit (candidates: Cohere Command /
   Qwen3-30B-A3B / DeepSeek — local Llama-class replacements for Gemini, served on the autoresearch
   vLLM `:8003`).
2. **Ship gate = judge-panel parity ONLY** (2026-08-02). Disjoint-vendor, scalar mode, strip
   `</think>`. **Human-GT / #1189 is NOT part of the parity gate** (it stays only the Phase-3
   reprocess acceptance gate).
3. **Prompt-tuning precedes the swap.**
4. **Sequencing = 2.5 FIRST, then go-live** (2026-08-02). Go-live is largely done and has no
   corpus dependency; it waits, does not block.
5. **DGX serving = the original plan** (2026-08-02) — v2.5 swaps ONLY the naming/summarization LLM
   onto the autoresearch vLLM `:8003`; the supporting services (ASR/diarization/failover) are fixed
   and unchanged. See the entry-state below + `docs/architecture/DGX_SERVING.md`.

## Verified DGX entry-state (read-only SSH as `ops@`, 2026-08-02) — READ BEFORE Phase D

The DGX serving stack is **intact** and matches the SSOT
`agentic-ai-homelab/infra/dgx/README.md`. (An earlier probe this day as the personal user
looked in the wrong home and wrongly reported the box "re-provisioned / bring-up path gone" —
that was a false alarm; corrected here.)

- **Access:** `ssh ops@dgx-llm-1` — the box runs as **`ops`** from `/home/ops/agentic-ai-homelab`
  (the personal home has nothing). `gpu-mode-swap.sh` is on PATH at
  `/usr/local/bin/gpu-mode-swap.sh` (absolute path in non-interactive shells).
- **GPU mode was `free`/idle**, which is why `:8003` served nothing. Bring the autoresearch
  vLLM up before the bake-off: `ssh ops@dgx-llm-1 /usr/local/bin/gpu-mode-swap.sh research`.
  So Phase D's original plan (autoresearch vLLM on `:8003`) works as written.
- **Running (supporting services, unchanged):** `faster-whisper` (:8000, ASR), `pyannote` (:8001,
  diarization), `moss` (:8004, ASR failover), ollama (:11434), obs stack. v2.5 swaps ONLY the
  naming/summarization LLM (on the autoresearch vLLM :8003); these support every run, unchanged.
- **Serving bridge (short → full):** `docs/architecture/DGX_SERVING.md` in this repo points to the
  homelab SSOT. Never touch `code`/coder-next (`feedback_never_use_coder_next`).

## Part 1 — the ORIGINAL v2.5 arc (do this FIRST)

Each step: what to do · goal optimized · metric evaluated against · operator gate.

- **Step 1 — Bring up the autoresearch vLLM + confirm the candidates load.**
  `ssh ops@dgx-llm-1 /usr/local/bin/gpu-mode-swap.sh research`, then load each local LLM candidate
  (Cohere Command / Qwen3-30B-A3B / DeepSeek) on `:8003` per `autoresearch/PER_MODEL_OPTIMAL_PARAMS.md`.
  *Goal:* the candidate set is servable on the autoresearch vLLM. *Metric:* each answers a
  chat/completion call (reachability + model identity). **GATE:** operator confirms which candidates
  are in/out. (Supporting services — ASR/diarization/failover — are already up; not part of this step.)
- **Step 2 — Bake-off harness + baseline.** Wire the autoresearch naming/resolver bake-off
  (`autoresearch/`, `JUDGING.md`, `bundled_prompt_tuning/`); pin the 2.4-Gemini baseline (relabel
  defect/named metrics from #1355 + judge panel); dry-run a few eps.
  *Goal:* comparable score per candidate. *Metric:* harness runs green on the 90-ep corpus via
  `scripts/backfill/relabel_corpus.py --llm none`. *Gate:* none.
- **Step 3 — Run bake-off, pick winner** (gap #1). Sweep candidates × prompt variants; score with
  the disjoint-vendor scalar judge panel; pick the winner.
  *Goal:* naming/resolver quality (correct speaker names + roles). *Metric:* judge-panel parity vs
  Gemini (winner ≥ Gemini within panel CI) + deterministic relabel defect/named parity.
  **GATE:** operator approves the winning model + parity result before any swap.
- **Step 4 — Provider swap (Phase E).** Wire the winner into `config/profiles/prod_dgx_balanced`,
  `prod_dgx_full_with_fallback`, `cloud_with_dgx_primary`; iterate prompts as real output diverges.
  *Goal:* same naming quality on the full pipeline. *Metric:* naming-4 harness + judge panel per
  iteration. *Gate:* none until Step 5.
- **Step 5 — Freeze 2.5 (Phase F).** Full reprocess of the 90-ep corpus under the DGX-local
  profile; confirm parity holds corpus-wide; write the 2.5 ADR (decision + alternatives + evidence).
  *Goal:* corpus-wide parity. *Metric:* judge-panel parity vs 2.4 across all 90 eps (ship gate).
  **GATE:** operator approves 2.5 ship.

### Assets verified on disk (this worktree)

- Harness: `autoresearch/JUDGING.md`, `autoresearch/PER_MODEL_OPTIMAL_PARAMS.md` (consult BEFORE +
  update AFTER each sweep), `autoresearch/bundled_prompt_tuning/`.
- Acceptance harness: `scripts/backfill/relabel_corpus.py` (`--llm none` = deterministic, no GPU).
- Parity baseline corpus: `.test_outputs/manual/prod-v2/corpus` (4.5G, 90 audio, 10 feeds).
- DGX profiles: `config/profiles/prod_dgx_*.yaml`, `cloud_with_dgx_primary.yaml`.
- naming-4 metrics: shipped in #1355 (relabel defect/named/tape).
- Memory: `project_autoresearch`, `feedback_silver_judge_vendor_bias`,
  `feedback_scalar_over_pairwise_for_judge_trust`, `feedback_judge_reasoning_block_parsing`,
  `feedback_deep_research_per_model`, `project_naming_arc_phase_a_outcome`.

## Part 2 — the NEW backlog items (only AFTER 2.5 freezes)

From `docs/wip/OPEN-BACKLOG-v3-10k.md`. The scale arc (Phase 3 → 4) and quality levers:

- **v4 fixture freeze (#1189)** — the reprocess acceptance gate (12 trap cases). Precedes the scale run.
- **Corpus growth to 500–1000 eps (#630)** — gap #3. **Needs operator input:** target size + curated-
  overlap-vs-broad-ingest strategy + onboarding mechanics. Forward dep: Corpus Scout (`PRD-037`/`RFC-088`).
- **Host identity build (EPIC-HOST-IDENTIFICATION)** — gap #4. Fully specced, zero code
  (`person→HOSTS→podcast`, `/api/relational/shows`, host scorecard). Needed before 10k scale.
- **Quality tier** (non-blocking, high-leverage): enricher hardening (#1168), speaker-resolution
  tail (~113 panel guests), diarization split/merge cause, graph perf #1219 (bites at 10k nodes).
- **Kept-open uncertain docs** (operator to review): `SPEAKER-PIPELINE-SUBSYSTEM-AUDIT.md`,
  `LORA_HYBRID_PIPELINE_PLAN.md`.

## Start here (fresh session)

1. Read `docs/wip/1000-EPISODES-REPROCESS-PLAN.md` (canonical) + this handover.
2. Confirm branch state; rebase on `origin/main` if stale.
3. **Step 1 — bring up the autoresearch vLLM** (`gpu-mode-swap research`) + confirm the candidates load; gate.
4. Proceed through Part 1 Steps 2→5. Do NOT open Part 2 until 2.5 is frozen.
