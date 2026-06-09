# Autoresearch next phase — 2-agent parallelization plan

**Generated:** 2026-06-09 (v2 — updated with #935-940 after v2.1 sweep)
**Scope:** the open work after the [v2.1 PR](#) ships. Companion to [`AUTORESEARCH_NEXT_PHASE_DEPENDENCIES.md`](AUTORESEARCH_NEXT_PHASE_DEPENDENCIES.md) — that doc maps the issue graph, this one maps the agent roles.

Designed for 2 agents working in parallel on the same branch (per operator's standing setup), with explicit file-ownership boundaries so the agents don't step on each other.

---

## TL;DR sequence

```text
Phase 0 — GATING (solo, ~1 day):
  One agent → #939 (upgrade silver to Opus 4.7)
  → 1 PR merged

Phase 0.5 — Prompt-tuning re-runs (parallel, ~2 days):
  Agent 1 → #935 (gemma3 prompts) + #936 (phi4 prompts)
  Agent 2 → #937 (hermes3 prompts) + #938 (mistral-small prompts)
  → 1-2 PRs merged

Phase 1 — Methodology + foundation (parallel, ~1-2 weeks):
  Agent 1 → #932 (finale tier code) + #940 Track 1 (R1-as-judge integration)
  Agent 2 → #921 (v3 fixtures)
  Either agent → #933 (prod-curated, when operator delivers backup)
  → 2 PRs merged

Phase 2 — Championships + R1 exploration (parallel, ~1-2 weeks):
  Agent 1 → #928 (summary/GI/KG championship)
  Agent 2 → #929 + #930 (transcription + diarization championships)
  Optional side-track → #940 Tracks 2-3 (entity-canon, grounding)
  → 3 PRs merged

Phase 3 — Synthesis (solo, ~1 week):
  One agent → #931 + #923 + ADR + final synthesis
  → 1 PR merged
```

**Total**: 8 PRs across 4-6 weeks, 2 agents most of the time. (Was 6 PRs; +2 for new prompt-tuning + silver upgrade work surfaced by v2.1.)

---

## Why the sequence changed

The v2.1 sweep + post-sweep discussion surfaced 6 new issues that change the dependency graph:

| Issue | What | Why it's where it is |
|---|---|---|
| **#939** Opus silver | Replace Sonnet 4.6 silver with Opus 4.7 | **Phase 0 — gates everything downstream.** Every comparison consumes silver. Doing it after Phase 1/2 means redoing the comparisons. Do it first, save the rework. |
| **#935-938** per-model prompts | Tune prompt templates for gemma3 / phi4 / hermes3 / mistral-small | **Phase 0.5 — short tasks, run against new silver.** v2.1 sweep used qwen3.5_9b prompts verbatim for these models; results were probably biased. ~half-day per model. |
| **#940** R1 alternative uses | Find better roles for DeepSeek-R1 distills (judge, entity-canon, grounding, decomposition) | **Track 1 in Phase 1 alongside #932** (R1-as-judge integrates directly into finale tier). **Tracks 2-5 in Phase 2 or later.** |

These insertions push the timeline from ~3-5 weeks to ~4-6 weeks but produce a substantially better methodology (better silver, better-tuned candidates, broader R1 utilization).

---

## Phase 0 — Silver upgrade (gating)

**One agent, ~1 day. No parallel work in this phase** — everything downstream consumes silver, so we don't want a half-applied upgrade.

### Single agent — #939 Opus silver upgrade

**Goal**: generate `silver_opus47_smoke_v1` and migrate all autoresearch references to it.

**MAY edit**:

- `data/eval/references/silver/silver_opus47_smoke_v1/` (NEW)
- `data/eval/configs/summarization/autoresearch_prompt_*.yaml` — update `reference: ...` field
- `data/eval/baselines/baseline_llm_ollama_*_smoke_paragraph_v1/` — re-score against new silver
- `docs/eval-history/SILVER_OPUS47_GENERATION_2026_06.md` (NEW — generation report)
- `docs/guides/eval-reports/EVAL_SMOKE_V2_DGX_REFRESH_2026_06.md` — append rescored numbers

**MUST NOT edit**:

- Old `silver_sonnet46_smoke_v1/` — keep for historical comparison
- Prompt templates (Phase 0.5)
- Anything in `src/podcast_scraper/evaluation/` (Phase 1)

**Deliverable**: one PR — new silver + all configs migrated + rescore report.

**Acceptance**:

- New silver generated for the 5-episode smoke_v1 dataset
- All v2 + v2.1 sweep outputs rescored
- Champion decision re-validated (qwen3.5:35b vs qwen3.6:latest may move)
- Old silver retained at original path

---

## Phase 0.5 — Per-model prompt tuning (parallel)

**Two agents, ~2 days total.** Touches `src/podcast_scraper/prompts/ollama/<slug>/summarization/` — model-specific dirs, zero contention.

### Agent 1 — gemma3 + phi4

**Goal**: hand-craft Gemma-native and Phi-native paragraph summarization templates; re-run sweep cells; update report.

**MAY edit**:

- `src/podcast_scraper/prompts/ollama/gemma3_27b/summarization/{system_v1.j2, long_v1.j2}` (REPLACE clones)
- `src/podcast_scraper/prompts/ollama/phi4_14b/summarization/{system_v1.j2, long_v1.j2}` (REPLACE clones)
- `data/eval/runs/llm_ollama_gemma3_27b_dgx_smoke_v2_1_tuned_*/` (NEW)
- `data/eval/runs/llm_ollama_phi4_14b_dgx_smoke_v2_1_tuned_*/` (NEW)
- Eval report addendum (own section per model)

**MUST NOT edit**:

- Other models' prompt dirs (Agent 2)
- DGX_MODEL_CATALOG / AI_PROVIDER_COMPARISON_GUIDE (synthesis phase)

**Deliverable**: 1 PR with both models retuned. Closes #935 + #936.

### Agent 2 — hermes3 + mistral-small

**Goal**: same shape as Agent 1, but for Hermes 3 (Nous Llama-3 fine-tune) and Mistral Small.

**MAY edit**:

- `src/podcast_scraper/prompts/ollama/hermes3_8b/summarization/{system_v1.j2, long_v1.j2}` (REPLACE clones)
- `src/podcast_scraper/prompts/ollama/mistral-small_24b/summarization/{system_v1.j2, long_v1.j2}` (REPLACE clones)
- `data/eval/runs/llm_ollama_hermes3_8b_dgx_smoke_v2_1_tuned_*/` (NEW)
- `data/eval/runs/llm_ollama_mistral-small_24b_dgx_smoke_v2_1_tuned_*/` (NEW)
- Eval report addendum (own section per model)

**MUST NOT edit**:

- Other models' prompt dirs (Agent 1)
- DGX_MODEL_CATALOG / AI_PROVIDER_COMPARISON_GUIDE (synthesis phase)

**Deliverable**: 1 PR with both models retuned. Closes #937 + #938.

### Why this phase exists

The v2.1 sweep result was biased — every new candidate ran against qwen3.5:9b's prompt format. Gemma 3's 0.207 RougeL surprise was almost certainly prompt-mismatch, not model failure. If even one of these models becomes a credible champion contender after tuning, the prod-LLM decision in #923 changes materially.

---

## Phase 1 — Methodology + v3 foundation (parallel)

**Two agents, ~1-2 weeks.** Both agents work simultaneously. Different file trees, zero contention.

### Agent 1 — Finale tier (#932) + R1-as-judge integration (#940 Track 1)

**Goal**: ship the G-Eval LLM-judge tier so future autoresearch verdicts have a real dimensional score, not just ROUGE. Integrate DeepSeek-R1 distill as a candidate judge alongside Sonnet 4.6 / Gemini 2.5 Pro.

**MAY edit**:

- `src/podcast_scraper/evaluation/g_eval.py` (NEW)
- `src/podcast_scraper/evaluation/finale_runner.py` (NEW)
- `src/podcast_scraper/evaluation/judges/{sonnet46, gemini25pro, deepseek_r1}.py` (NEW per-judge clients)
- `scripts/eval/finale_sweep.py` (NEW)
- `scripts/eval/explore_r1_as_judge.py` (NEW — Track 1 eval harness for #940)
- `data/eval/configs/finale/<sweep_tag>.yaml` (NEW)
- `tests/unit/evaluation/test_g_eval.py` (NEW)
- `tests/unit/evaluation/test_judge_clients.py` (NEW)
- `docs/guides/eval-reports/EVAL_FINALE_METHODOLOGY.md` (NEW)
- `docs/guides/eval-reports/EVAL_R1_AS_JUDGE_2026_XX.md` (NEW — Track 1 result)
- Minor edits to `src/podcast_scraper/evaluation/comparator.py`

**MUST NOT edit**:

- `tests/fixtures/v3/` (Agent 2's territory)
- DGX_MODEL_CATALOG / AI_PROVIDER_COMPARISON_GUIDE / profile yamls (Phase 3)
- Other #940 tracks (Phase 2)

**Deliverable**: one PR that:

- Implements G-Eval 4-dimension scoring (faithfulness / coverage / coherence / fluency)
- Stratifies finalists per #932 spec (cloud / MBP / DGX Spark / hybrid)
- Wires Sonnet 4.6 primary + Gemini 2.5 Pro cross-check on top-2
- Adds R1 distill as a tertiary judge with agreement-test eval
- Runs end-to-end against the v2 + v2.1 sweep outputs as smoke test (using new Opus silver from Phase 0)
- Defers prod-curated sanity check to a follow-up (gated on #933)

**Acceptance**:

- G-Eval runs against v2+v2.1 sweep outputs against the new Opus silver
- Judge-agreement Sonnet vs Gemini ≥80% on top-2
- R1 judge agreement vs Sonnet ≥75% → integrate as cost-saving judge slot
- Pareto chart generated

**Why fold #940 Track 1 in here**: R1-as-judge IS a finale-tier design question (which judge models to use). Splitting it out would mean #932 ships with Sonnet+Gemini only, then a separate PR adds R1 — wasteful.

### Agent 2 — v3 smoke fixtures (#921)

**Goal**: rebuild v2 smoke as v3 with all the failure modes from `AUTORESEARCH_LEARNINGS_FOR_V3.md` baked in, plus diarization-aware multi-voice TTS audio (per operator's standing diarization-fixture work).

**MAY edit**:

- `tests/fixtures/v3/` (NEW big tree — generator + ground truth + episodes)
- `scripts/build_v3_fixtures.py` (NEW generator)
- `data/eval/datasets/curated_5feeds_smoke_v3/` (NEW)
- `docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md` (mark items as "landed in v3")
- `docs/guides/eval-reports/EVAL_FIXTURES_V3.md` (NEW)
- Tests under `tests/integration/eval/test_v3_fixtures.py`
- The diarization e2e test pinned to v1 in this PR can be re-pointed to v3 once the multi-voice audio lands

**MUST NOT edit**:

- Anything under `src/podcast_scraper/evaluation/` (Agent 1's territory)
- Existing v2 fixtures (additive, don't break v2)
- Catalog / AI guide / profile yamls (Phase 3)

**Deliverable**: one PR (or series) that:

- Implements every failure mode tag in `AUTORESEARCH_LEARNINGS_FOR_V3.md` as a knob in the generator
- Ships ~30-50 v3 episodes with ground truth labels
- Includes diarization-aware multi-voice audio (resolves the v1-pinned test from this PR)
- Adds CI gate that runs autoresearch on v3 and asserts result-shape consistency

**Acceptance**: each failure-mode tag exercised by ≥1 episode; smoke v3 runs to completion against v2's existing model matrix without errors; diarization e2e test passes against v3 audio.

### Synchronization mid-phase

- When operator supplies prod backup → **#933 prod-curated set** curated (half-day task, either agent picks it up between active subtasks)
- Once #933 lands → Agent 1 wires the top-2 sanity check into the finale tier
- Once #921 lands → diarization e2e test re-points from v1 to v3 (smallest PR)

---

## Phase 2 — Championships + R1 exploration (parallel)

**Two agents, ~1-2 weeks.** Both agents run their championship sweeps + reports in parallel. Each touches its own eval domain.

### Agent 1 — Summary/GI/KG championship (#928) + R1 entity-canon track (#940 Track 2)

**Goal**: run the cloud-vs-DGX summary championship using the finale tier from Phase 1. Side track: explore R1 distill for entity-canonicalization decisions (Track 2 of #940).

**MAY edit**:

- `scripts/eval/championship_summary.py` (NEW)
- `scripts/eval/r1_entity_canon_eval.py` (NEW — Track 2)
- `data/eval/runs/championship_summary_*/` (NEW)
- `data/eval/configs/championship/summary_*.yaml` (NEW)
- `docs/guides/eval-reports/EVAL_CHAMPIONSHIP_SUMMARY.md` (NEW)
- `docs/guides/eval-reports/EVAL_R1_ENTITY_CANON_2026_XX.md` (NEW — Track 2)
- Possibly minor edits to GI/KG eval helpers + entity-canon predicates

**MUST NOT edit**:

- Transcription / diarization eval code (Agent 2)
- AI guide / catalog / profile yamls (Phase 3)

**Deliverable**: PR with championship results + per-stratum winner + Pareto chart + R1-entity-canon evaluation.

### Agent 2 — Transcription + Diarization championships (#929 + #930) + R1 grounding track (#940 Track 3)

**Goal**: same as Agent 1 but for audio stages, plus exploration of R1 for insight grounding (Track 3 of #940).

**MAY edit**:

- `scripts/eval/championship_transcription.py` (NEW)
- `scripts/eval/championship_diarization.py` (NEW)
- `scripts/eval/r1_grounding_eval.py` (NEW — Track 3)
- `data/eval/runs/championship_transcription_*/` (NEW)
- `data/eval/runs/championship_diarization_*/` (NEW)
- `data/eval/configs/championship/transcription_*.yaml`, `diarization_*.yaml` (NEW)
- `docs/guides/eval-reports/EVAL_CHAMPIONSHIP_TRANSCRIPTION.md`, `EVAL_CHAMPIONSHIP_DIARIZATION.md` (NEW)
- `docs/guides/eval-reports/EVAL_R1_GROUNDING_2026_XX.md` (NEW — Track 3)

**MUST NOT edit**:

- Summary/GI/KG eval (Agent 1)
- AI guide / catalog / profile yamls (Phase 3)

**Deliverable**: PR(s) with per-stage championship results + R1-grounding evaluation.

### Why this split

Agent 1 owns the "text → text" track (summary/GI/KG + entity-canon). Agent 2 owns the "audio → text" + "audio → speakers" tracks + grounding. Different model APIs, different metrics, different failure modes. Each agent owns one of #940's exploration tracks naturally.

R1 Track 4 (cleaning verification) + Track 5 (transcript decomposition) deferred — lower priority per the #940 issue body. Pick up in a future iteration.

---

## Phase 3 — Synthesis (solo)

Single agent. Pulls together all championship verdicts + writes the prod commit. Must be solo because the work is inherently holistic.

### Whichever agent has bandwidth — Hybrid routing + prod profile (#931 + #923)

**MAY edit**:

- `config/profiles/prod_dgx_full_with_fallback.yaml` (the actual commit)
- `docs/operations/DGX_MODEL_CATALOG.md` (consolidate everything from Phase 0.5 + Phase 2)
- `docs/guides/AI_PROVIDER_COMPARISON_GUIDE.md` (final scoreboard merge — v2 + v2.1 + tuned re-runs + championships)
- `docs/adr/ADR-XXX-dgx-prod-profile.md` (NEW — the decision record)
- `docs/guides/eval-reports/EVAL_CHAMPIONSHIP_HYBRID.md` (NEW — synthesis report)
- `docs/wip/AUTORESEARCH_NEXT_PHASE_DEPENDENCIES.md` (update to "done")
- `docs/wip/AUTORESEARCH_NEXT_PHASE_AGENT_PLAN.md` (this file — mark complete)

**Deliverable**: one PR that:

- Synthesizes #928 + #929 + #930 into a routing decision
- Commits `prod_dgx_full_with_fallback.yaml` with concrete model picks
- Writes the ADR
- Closes #923, #927, #931
- Updates the dependency map + agent plan to mark the programme as effectively done

---

## Coordination rules across all phases

1. **One PR per phase per agent.** Don't mix phases in the same PR.
2. **Agents never edit the same file in the same phase.** If a file needs touching by both, the synthesis agent in the next phase consolidates.
3. **Conflict resolution**: if Agent 1 needs something from Agent 2's tree (or vice versa), file a short coordination note in `docs/wip/AUTORESEARCH_AGENT_COORDINATION.md` rather than directly editing.
4. **Sync points at phase boundaries.** Phase 2 doesn't start until Phase 1 PRs land. Phase 3 doesn't start until all Phase 2 PRs land.
5. **Phase 0 (silver upgrade) is gating.** Nothing in Phase 0.5+ runs until #939 lands — every subsequent comparison consumes silver.
6. **Prod-curated tier (#933)** is a half-day micro-task that either agent picks up between active subtasks; it's not a phase of its own.

---

## Notes on file contention

The reason the plan splits this way: a handful of files are touched by every agent at the end of every phase. Without explicit ownership, both agents will race to update them, causing rework.

| File | Touched by | Resolution |
|---|---|---|
| `docs/operations/DGX_MODEL_CATALOG.md` | Phase 0.5 + Phase 2 + Phase 3 | Only Phase 3 synthesizer touches |
| `docs/guides/AI_PROVIDER_COMPARISON_GUIDE.md` | Phase 0.5 + Phase 2 + Phase 3 | Only Phase 3 synthesizer touches |
| `config/profiles/prod_*.yaml` | Phase 3 only | Only Phase 3 |
| `data/eval/references/silver/silver_sonnet46_smoke_v1/` | Read-only after Phase 0 | Frozen for historical comparison |
| `data/eval/references/silver/silver_opus47_smoke_v1/` | Phase 0 only | After that, read-only |
| `docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md` | Phase 1 Agent 2 + future | Phase 1 Agent 2 owns during Phase 1 |
| `docs/wip/AUTORESEARCH_NEXT_PHASE_DEPENDENCIES.md` | Phase 3 synthesis only | Update at end of programme |

Agents in Phase 2 record championship verdicts in their own report files (`EVAL_CHAMPIONSHIP_*.md`). The Phase 3 synthesizer reads those reports and consolidates into the high-traffic docs. Phase 0.5 agents record per-model retuning results in eval-report addenda — Phase 3 synthesizer pulls those into the catalog + AI guide.

---

## When the plan needs revision

- If #933 prod backup isn't supplied in time → Phase 1 still runs (finale tier ships without sanity check; #921 doesn't need it). Phase 2 + 3 do need it.
- If v3 (#921) is slower than finale (#932) — likely → Phase 2 can start with v2 smoke as the dataset, switch to v3 when ready.
- If Phase 0.5 retuning reveals a clear new champion → re-run #932 with the new top-tier candidate as a finale entry; don't wait for Phase 2 championships.
- If Opus silver upgrade (#939) reveals score compression (smaller deltas across models) → that's a feature, but the ranking may flip; reassess #923 model pick before any prod swap.
- If new candidate models surface mid-programme → run as a separate small task, fold into the next available championship.
- If a championship runs into a model-API change (e.g. Anthropic API breaking change) → block that championship, others continue.

---

## Related

- [Next-phase dependency map](AUTORESEARCH_NEXT_PHASE_DEPENDENCIES.md) — issue-level dependency graph
- [Autoresearch learnings for v3](AUTORESEARCH_LEARNINGS_FOR_V3.md) — failure-mode catalogue (becomes #921 spec)
- Open issues by phase:
  - **Phase 0**: [#939](https://github.com/chipi/podcast_scraper/issues/939) Opus silver
  - **Phase 0.5**: [#935](https://github.com/chipi/podcast_scraper/issues/935), [#936](https://github.com/chipi/podcast_scraper/issues/936), [#937](https://github.com/chipi/podcast_scraper/issues/937), [#938](https://github.com/chipi/podcast_scraper/issues/938) per-model prompt tuning
  - **Phase 1**: [#932](https://github.com/chipi/podcast_scraper/issues/932) finale, [#921](https://github.com/chipi/podcast_scraper/issues/921) v3 fixtures, [#940 Track 1](https://github.com/chipi/podcast_scraper/issues/940) R1-as-judge, [#933](https://github.com/chipi/podcast_scraper/issues/933) prod-curated
  - **Phase 2**: [#928](https://github.com/chipi/podcast_scraper/issues/928), [#929](https://github.com/chipi/podcast_scraper/issues/929), [#930](https://github.com/chipi/podcast_scraper/issues/930), [#940 Tracks 2-3](https://github.com/chipi/podcast_scraper/issues/940)
  - **Phase 3**: [#931](https://github.com/chipi/podcast_scraper/issues/931), [#923](https://github.com/chipi/podcast_scraper/issues/923)
