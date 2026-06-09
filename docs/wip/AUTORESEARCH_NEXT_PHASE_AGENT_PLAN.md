# Autoresearch next phase — 2-agent parallelization plan

**Generated:** 2026-06-09
**Scope:** the open work after the [v2.1 PR](#) ships. Companion to [`AUTORESEARCH_NEXT_PHASE_DEPENDENCIES.md`](AUTORESEARCH_NEXT_PHASE_DEPENDENCIES.md) — that doc maps the issue graph, this one maps the agent roles.

Designed for 2 agents working in parallel on the same branch (per operator's standing setup), with explicit file-ownership boundaries so the agents don't step on each other.

---

## TL;DR sequence

```text
Phase 1 (parallel, ~1-2 weeks):
  Agent 1 → #932 (finale tier code)
  Agent 2 → #921 (v3 fixtures)
  Either agent → #933 (prod-curated, when operator delivers backup)
  → 2 PRs merged

Phase 2 (parallel, ~1-2 weeks):
  Agent 1 → #928 (summary/GI/KG championship)
  Agent 2 → #929 + #930 (transcription + diarization championships)
  → 3 PRs merged

Phase 3 (solo, ~1 week):
  One agent → #931 + #923 + ADR + final synthesis
  → 1 PR merged
```

**Total**: 6 PRs across 3-5 weeks, 2 agents most of the time.

---

## PHASE 1 — Methodology + v3 foundation (parallel)

Both agents work simultaneously. Different file trees, zero contention.

### Agent 1 — Finale tier (#932)

**Goal**: ship the G-Eval LLM-judge tier so future autoresearch verdicts have a real dimensional score, not just ROUGE.

**MAY edit**:

- `src/podcast_scraper/evaluation/g_eval.py` (NEW)
- `src/podcast_scraper/evaluation/finale_runner.py` (NEW)
- `scripts/eval/finale_sweep.py` (NEW)
- `data/eval/configs/finale/<sweep_tag>.yaml` (NEW)
- `tests/unit/evaluation/test_g_eval.py` (NEW)
- `docs/guides/eval-reports/EVAL_FINALE_METHODOLOGY.md` (NEW — design doc)
- Minor edits to `src/podcast_scraper/evaluation/comparator.py` (touch the existing class)

**MUST NOT edit**:

- `tests/fixtures/v3/` (Agent 2's territory)
- `docs/operations/DGX_MODEL_CATALOG.md`, `docs/guides/AI_PROVIDER_COMPARISON_GUIDE.md`, `config/profiles/prod_*.yaml` (those land in Phase 3 synthesis)

**Deliverable**: one PR that

- Implements G-Eval 4-dimension scoring (faithfulness / coverage / coherence / fluency)
- Stratifies finalists per #932 spec (cloud / MBP / DGX Spark / hybrid)
- Wires Sonnet 4.6 primary + Gemini 2.5 Pro cross-check on top-2
- Runs end-to-end against the v2 sweep's existing outputs as smoke test
- Defers prod-curated sanity check to a follow-up (gated on #933)

**Acceptance**: G-Eval runs against v2 sweep outputs, judge-agreement ≥80% on top-2, Pareto chart generated.

### Agent 2 — v3 smoke fixtures (#921)

**Goal**: rebuild v2 smoke as v3 with all the failure modes from `AUTORESEARCH_LEARNINGS_FOR_V3.md` baked in.

**MAY edit**:

- `tests/fixtures/v3/` (NEW big tree — generator + ground truth + episodes)
- `scripts/build_v3_fixtures.py` or similar generator (NEW)
- `data/eval/datasets/curated_5feeds_smoke_v3/` (NEW)
- `docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md` (mark items as "landed in v3")
- `docs/guides/eval-reports/EVAL_FIXTURES_V3.md` (NEW)
- Tests under `tests/integration/eval/test_v3_fixtures.py`

**MUST NOT edit**:

- Anything under `src/podcast_scraper/evaluation/` (Agent 1's territory)
- Existing v2 fixtures (additive, don't break v2)
- Catalog / AI guide / profile yamls (Phase 3)

**Deliverable**: one PR (or series) that

- Implements every failure mode tag in `AUTORESEARCH_LEARNINGS_FOR_V3.md` as a knob in the generator
- Ships ~30-50 v3 episodes with ground truth labels
- Adds CI gate that runs autoresearch on v3 and asserts result-shape consistency

**Acceptance**: each failure-mode tag exercised by ≥1 episode; smoke v3 runs to completion against v2's existing model matrix without errors.

### Synchronization mid-phase

- When operator supplies prod backup → **#933 prod-curated set** curated (half-day task, either agent picks it up between active subtasks)
- Once #933 lands → Agent 1 wires the top-2 sanity check into the finale tier

---

## PHASE 2 — Championships (parallel)

Both agents run their championship sweeps + reports in parallel. Each touches its own eval domain.

### Agent 1 — Summary/GI/KG championship (#928)

**Goal**: run the cloud-vs-DGX summary championship using the finale tier from Phase 1.

**MAY edit**:

- `scripts/eval/championship_summary.py` (NEW)
- `data/eval/runs/championship_summary_*/` (NEW)
- `data/eval/configs/championship/summary_*.yaml` (NEW)
- `docs/guides/eval-reports/EVAL_CHAMPIONSHIP_SUMMARY.md` (NEW)
- Possibly minor edits to GI/KG eval helpers

**MUST NOT edit**:

- Transcription / diarization eval code (Agent 2)
- AI guide / catalog / profile yamls (Phase 3)

**Deliverable**: PR with championship results + per-stratum winner + Pareto chart.

### Agent 2 — Transcription + Diarization championships (#929 + #930)

**Goal**: same as Agent 1 but for the audio stages.

**MAY edit**:

- `scripts/eval/championship_transcription.py` (NEW)
- `scripts/eval/championship_diarization.py` (NEW)
- `data/eval/runs/championship_transcription_*/` (NEW)
- `data/eval/runs/championship_diarization_*/` (NEW)
- `data/eval/configs/championship/transcription_*.yaml`, `diarization_*.yaml` (NEW)
- `docs/guides/eval-reports/EVAL_CHAMPIONSHIP_TRANSCRIPTION.md`, `EVAL_CHAMPIONSHIP_DIARIZATION.md` (NEW)

**MUST NOT edit**:

- Summary/GI/KG eval (Agent 1)
- AI guide / catalog / profile yamls (Phase 3)

**Deliverable**: PR(s) with per-stage championship results.

### Why this split

Agent 1 owns the "text → text" championship (summary/GI/KG). Agent 2 owns the "audio → text" + "audio → speakers" championships. Different model APIs, different metrics, different failure modes. Clean conceptual boundary, zero file overlap.

---

## PHASE 3 — Synthesis (solo)

Single agent. Pulls together all 3 championship verdicts + writes the prod commit. Must be solo because the work is inherently holistic.

### Whichever agent has bandwidth — Hybrid routing + prod profile (#931 + #923)

**MAY edit**:

- `config/profiles/prod_dgx_full_with_fallback.yaml` (the actual commit)
- `docs/operations/DGX_MODEL_CATALOG.md` (consolidate everything)
- `docs/guides/AI_PROVIDER_COMPARISON_GUIDE.md` (final scoreboard merge)
- `docs/adr/ADR-XXX-dgx-prod-profile.md` (NEW — the decision record)
- `docs/guides/eval-reports/EVAL_CHAMPIONSHIP_HYBRID.md` (NEW — synthesis report)
- `docs/wip/AUTORESEARCH_NEXT_PHASE_DEPENDENCIES.md` (update to "done")

**Deliverable**: one PR that

- Synthesizes #928 + #929 + #930 into a routing decision
- Commits `prod_dgx_full_with_fallback.yaml` with concrete model picks
- Writes the ADR
- Closes #923, #927, #931
- Updates the dependency map to mark the autoresearch programme as effectively done

---

## Coordination rules across all phases

1. **One PR per phase per agent.** Don't mix phases in the same PR.
2. **Agents never edit the same file in the same phase.** If a file needs touching by both, the synthesis agent in the next phase consolidates.
3. **Conflict resolution**: if Agent 1 needs something from Agent 2's tree (or vice versa), file a short coordination note in `docs/wip/AUTORESEARCH_AGENT_COORDINATION.md` rather than directly editing.
4. **Sync points at phase boundaries.** Phase 2 doesn't start until Phase 1 PRs land. Phase 3 doesn't start until all Phase 2 PRs land.
5. **Prod-curated tier (#933)** is a half-day micro-task that either agent picks up between active subtasks; it's not a phase of its own.

---

## Notes on file contention

The reason the plan splits this way: a handful of files are touched by every agent at the end of every phase. Without explicit ownership, both agents will race to update them, causing rework.

| File | Touched by | Resolution |
|---|---|---|
| `docs/operations/DGX_MODEL_CATALOG.md` | Phase 2 + Phase 3 | Only Phase 3 synthesizer touches |
| `docs/guides/AI_PROVIDER_COMPARISON_GUIDE.md` | Phase 2 + Phase 3 | Only Phase 3 synthesizer touches |
| `config/profiles/prod_*.yaml` | Phase 3 only | Only Phase 3 |
| `docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md` | Phase 1 Agent 2 + future | Phase 1 Agent 2 owns during Phase 1 |
| `docs/wip/AUTORESEARCH_NEXT_PHASE_DEPENDENCIES.md` | Phase 3 synthesis only | Update at end of programme |

Agents in Phase 2 record championship verdicts in their own report files (`EVAL_CHAMPIONSHIP_*.md`). The Phase 3 synthesizer reads those reports and consolidates into the high-traffic docs.

---

## When the plan needs revision

- If #933 prod backup isn't supplied in time → Phase 1 still runs (finale tier ships without sanity check; #921 doesn't need it). Phase 2 + 3 do need it.
- If v3 (#921) is slower than finale (#932) — likely → Phase 2 can start with v2 smoke as the dataset, switch to v3 when ready.
- If new candidate models surface mid-programme (e.g. v2.2 sweep) → run as a separate small task, fold into the next available championship.
- If a championship runs into a model-API change (e.g. Anthropic API breaking change) → block that championship, others continue.

---

## Related

- [Next-phase dependency map](AUTORESEARCH_NEXT_PHASE_DEPENDENCIES.md) — issue-level dependency graph
- [Autoresearch learnings for v3](AUTORESEARCH_LEARNINGS_FOR_V3.md) — failure-mode catalogue (becomes #921 spec)
- Open issues: [#932](https://github.com/chipi/podcast_scraper/issues/932), [#933](https://github.com/chipi/podcast_scraper/issues/933), [#921](https://github.com/chipi/podcast_scraper/issues/921), [#928](https://github.com/chipi/podcast_scraper/issues/928), [#929](https://github.com/chipi/podcast_scraper/issues/929), [#930](https://github.com/chipi/podcast_scraper/issues/930), [#931](https://github.com/chipi/podcast_scraper/issues/931), [#923](https://github.com/chipi/podcast_scraper/issues/923)
