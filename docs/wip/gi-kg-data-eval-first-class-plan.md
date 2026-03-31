# Plan: GI and KG as first-class tasks in `data/eval`

**Purpose:** Actionable roadmap to extend the ML experiment framework (`data/eval`,
`scripts/eval/run_experiment.py`, `ExperimentConfig`) so **Grounded Insights (GIL / GI)**
and **Knowledge Graph (KG)** can be run, versioned, and scored like summarization and NER.

**Implemented (baseline):** `task: grounded_insights` / `task: knowledge_graph` with
`backend: eval_stub`, sample configs under `data/eval/configs/`, scorers and `score_run(task=...)`,
gold paths `references/gold/gil/` and `references/gold/kg/`. **Not yet implemented:** OpenAI
(or other) backends and coupled “regenerate summary then GIL/KG” in `run_experiment`.

**Audience:** Implementers touching evaluation, providers, and artifact schemas.

### Engineering stance (agreed)

- **Two separate capabilities** — Grounded insights (**GI / GIL**) and knowledge graph (**KG**)
  are **independent** in `data/eval`: distinct **`task`** values, distinct experiment configs,
  distinct runs under `data/eval/runs/`, distinct references and baselines. A run does **one**
  task (GI **or** KG, not both). Either capability can be dropped or paused later without
  collapsing the other into a combined pipeline.
- **No RFC** — This is an **anticipated extension** of the existing experiment system
  (same patterns as summarization and NER). Document changes in this WIP plan and normal
  guides; skip a standalone RFC unless something becomes breaking or cross-cutting beyond
  eval.
- **Extension-only** — Prefer **additive** changes (new tasks, scorers, reference paths,
  schemas). Touch summarization/NER code paths only when required for shared helpers
  (e.g. explicit `task` passed into `score_run()`).

**Related:**

- Prior gap analysis (conversation): eval today supports **summarization** + **ner_entities**;
  GI is **off** in eval (`generate_gi=False`); KG-as-graph is **not** in eval; NER supports
  **KG bootstrapping** via `entity_set` scoring only.
- Manual validation today: [manual-test-plan-gi-kg.md](manual-test-plan-gi-kg.md),
  `config/manual/*.yaml`, `config/acceptance/gi/`, `config/acceptance/kg/`.
- Scope calibration (shallow v1 vs depth): [gi-kg-shallow-v1-vs-full-depth.md](gi-kg-shallow-v1-vs-full-depth.md).
- Experiment layout: `data/eval/README.md` (repo root),
  [docs/guides/EXPERIMENT_GUIDE.md](../guides/EXPERIMENT_GUIDE.md).
- Normative artifacts: [docs/gi/gi.schema.json](../gi/gi.schema.json),
  [docs/kg/kg.schema.json](../kg/kg.schema.json).

---

## Before you start (pre-flight)

Do this **once** before Phase 1 coding so **GI eval** and **KG eval** (separate
capabilities) stay aligned with the existing system (templates, config rules, metrics shape,
scorer, promotion story).

### 1. Lock a few product decisions (short)

You can refine later, but pick defaults so implementers do not fork patterns:

- **Coupled vs frozen-summary (per task):** For `summary_bullets`, will the first **GI**
  and **KG** eval configs **regenerate** summary in-run or **read** frozen `summary_final`
  per episode? Decide **independently** if needed; no single combined GI+KG run.
- **Reference strategy for v1:** Gold-only smoke episodes vs silver promoted run vs hybrid.
- **Primary regression metrics:** One scalar (or small tuple) per task for “did we break
  the baseline?” — same role ROUGE/embeddings and NER F1 play today.

No RFC is required if you record choices here or in a short WIP addendum.

### 2. Read the canonical experiment contract (existing behavior)

Skim these so new code **extends** them instead of inventing parallel flows:

| Topic | Where |
| ----- | ----- |
| Workflow order (sources → dataset → materialize → baseline → experiments) | [EXPERIMENT_GUIDE.md](../guides/EXPERIMENT_GUIDE.md) |
| Run layout, immutability, roles of baselines vs silver/gold | `data/eval/README.md` (repo root) |
| Config file rules (`id` = filename, task/backends) | `data/eval/configs/README.md` (repo root) |
| `ExperimentConfig` schema and validators | `src/podcast_scraper/evaluation/experiment_config.py` |

### 3. Implementation alignment checklist (must match existing code)

These are **non-negotiable** for “first class” parity with summarization and NER:

1. **Prediction record shape (JSONL)** — Same envelope as today: `episode_id`,
   `dataset_id`, `output`, `fingerprint_ref`, `metadata` (hashes, paths, char counts,
   `processing_time_seconds`). Add GI/KG payloads under **`output`** using **distinct
   key** per task (`output.gil` **or** `output.kg`, never both in one row). Follow the same
   pattern as `output.summary_final` and `output.entities`.

2. **Task detection in scoring** — `score_run()` in `scorer.py` currently **infers**
   `task_type` from the **first** prediction’s `output` keys (not from YAML). When you
   add GI/KG you must either:
   -    extend that inference in a deterministic order, **or**
   - pass **`task` explicitly** from `run_experiment.py` into `score_run()` (recommended
     for clarity; each run has exactly one primary `output` artifact type).

3. **Reference resolution** — Extend `find_reference_path()` in `run_experiment.py` with
   the same style as `references/gold/ner_entities/` and
   `references/gold/summarization/` (e.g. `references/gold/gil/` and
   `references/gold/kg/`). Keep silver and baseline fallback behavior consistent.

4. **`metrics.json` schema** — Mirror existing layout: top-level `dataset_id`, `run_id`,
   `episode_count`, `intrinsic`, `vs_reference`, **`task`**, **`schema`**
   (`metrics_*_v1`). Add JSON Schema files under `data/eval/schemas/` and wire
   `validate_metrics_*` in `schema_validator.py` + the post-score block in
   `run_experiment.py` (same pattern as summarization and NER).

5. **Intrinsic metrics** — `compute_intrinsic_metrics()` is **summary-text-centric**
   (gates, length tokens). For NER, episodes without `summary_final` are mostly **skipped**,
   so gates/length can be zero or uninformative. For GI/KG, either **accept the same
   pattern initially** (performance block still works from `metadata`) **or** add a
   **task-aware branch** for meaningful intrinsic gates — but do not silently redefine
   “gate” semantics for summarization.

6. **Run artifacts** — Same directory contract as existing runs: `predictions.jsonl`,
   `baseline.json`, `fingerprint.json`, `metrics.json`, `metrics_report.md`, `README.md`
   from `data/eval/runs/RUN_README_TEMPLATE.md`, and config snapshot behavior already used
   for baselines/runs.

7. **Entrypoint** — Prefer **`make experiment-run CONFIG=data/eval/configs/<id>.yaml`**
   with the same optional `BASELINE=` / `REFERENCE=` / `SCORE_ONLY=` flags; only add new
   Make targets if there is a strong reason.

8. **Human-readable report** — `reporter.py` is mostly generic; extend it if `vs_reference`
   for GI/KG needs structured sections (same idea as NER vs summarization blocks, if any).

9. **Fingerprints** — Keep `generate_enhanced_fingerprint()` honest for new backends/tasks
   so `fingerprint.json` remains comparable run-over-run.

10. **Runtime vs eval data** — Do **not** import `data/eval/` from `src/podcast_scraper/`;
    reuse production builders (`gi.pipeline`, KG generation paths) from **scripts** or thin
    adapters called by `run_experiment.py` only.

### 4. Optional but recommended

- Trace one full **summarization** and one **NER** run locally (small dataset) and diff
  the produced files against [EXPERIMENT_GUIDE.md](../guides/EXPERIMENT_GUIDE.md)
  expectations.
- Grep `promote_run` / baseline promotion for assumptions on `task` or `schema` before
  relying on promotion for GI/KG.

---

## Goals and non-goals

### Goals

1. **Two first-class tasks** (implemented and operated **separately**):
   - **`grounded_insights` (GI)** — per-episode GIL payloads aligned with production
     `gi.json`, written to `predictions.jsonl`, scored vs GI-only references/baselines.
   - **`knowledge_graph` (KG)** — per-episode KG payloads aligned with production `kg.json`,
     same eval loop, scored vs KG-only references/baselines.
   - No combined “GI+KG” experiment type: if both matter, run **two** configs (two run IDs).
2. **Same invariants** as existing eval: fixed `dataset_id`, materialized transcripts,
   fingerprinting, `runs/` → optional promotion to `baselines/` / `references/`.
3. **Isolated evaluation** per task: transcript-first (and optional frozen-summary inputs
   **within that task’s config** when you add decoupled mode), without requiring full RSS
   ingestion for the eval dataset — mirror the summarization experiment pattern.

### Non-goals (initial phases)

- **Full-depth KG** (entity resolution, cross-episode joins, NL query IR) as a metric —
  defer until graph references and metrics are stable for shallow v1.
- **Replacing** acceptance configs under `config/acceptance/` — eval complements them;
  unify naming and thresholds over time.
- **Runtime coupling** — application code must **not** import `data/eval/` (existing
  invariant); promotion scripts may read eval outputs.

---

## Current state (baseline)

| Area | Summarization / NER | GI | KG (graph) |
| ---- | ------------------- | -- | ---------- |
| `ExperimentConfig.task` | Implemented in runner | Not a task | Not a task |
| `run_experiment` inference | Yes | `generate_gi=False` hardcoded | No path |
| `predictions.jsonl` shape | `summary_final` / `entities` | N/A | N/A |
| References | Silver + gold paths exist | None under `references/` | None |
| Scorers | ROUGE / embeddings / NER | None | None |

---

## Design principles

1. **Separation of concerns** — GI and KG share **machinery** (runner, JSONL envelope,
   `metrics.json` shape) but **never** a single run or a single baseline ID. Metrics schema
   IDs and reference trees are **per capability** (`metrics_gil_v1` vs `metrics_kg_v1`,
   `gold/gil/` vs `gold/kg/`).
2. **One episode → one prediction record** (extend the existing JSONL row schema with a
   single dominant `output` key per task, e.g. `gil` **or** `kg`, plus `metadata`).
3. **References are versioned directories** under `data/eval/references/`, analogous to
   `gold/ner_entities/<ref_id>/` and silver summarization runs — **separate** GI and KG trees.
4. **Reuse production builders** where possible: call into `gi.pipeline.build_artifact`
   and KG generation entry points used by `metadata_generation` with a **thin eval adapter**
   that supplies transcript text + `Config` (task-specific subset) — avoid duplicating prompt
   and provider wiring.
5. **Phase metrics** — start with **schema-valid + shallow v1 metrics** (counts, rates,
   simple graph overlap); add semantic metrics later if needed.
6. **Align with shallow v1** ([gi-kg-shallow-v1-vs-full-depth.md](gi-kg-shallow-v1-vs-full-depth.md)):
   evaluation targets should match what the product guarantees today, not backlog depth work.

---

## Phase 0 — Decisions (short)

Record answers in this doc (no RFC):

- **Reference strategy:** Silver-only (LLM judge / human light touch) vs gold (curated
  per-episode expected graphs) vs hybrid (gold for 3–5 smoke episodes, silver for bulk).
- **GI scoring contract:** Minimum bar = schema + % insights with ≥1 grounded quote
  verbatim in transcript? NLI/QA thresholds as separate knobs?
- **KG scoring contract:** Node/edge set overlap vs typed relation F1 vs embedding
  similarity of labels — pick one primary metric for regression gating.
- **Dependency on summary (per task):** For `summary_bullets` **GI** or **KG** runs, choose
  default: frozen summary sidecar vs regenerate summary inside that single-task run.
  Decisions can differ between GI and KG; keep modes as **separate config flags**, not one
  shared “GI+KG” switch.

---

## Phase 1 — Config and schema

1. **Extend `ExperimentConfig`** (`src/podcast_scraper/evaluation/experiment_config.py`):
   - Add task literals, e.g. `grounded_insights` and `knowledge_graph` (names TBD; keep
     consistent with docs and Makefile).
   - Add optional sections or `params` keys for GI/KG mirroring
     [CONFIGURATION.md](../api/CONFIGURATION.md) (insight source, max insights, KG
     extraction source, provider ids, thresholds).
2. **Validation rules:** Which backends are allowed per task (e.g. GI may require OpenAI
   or hybrid for `provider` path; stub allowed for smoke only — document explicitly).
3. **Sample configs** under `data/eval/configs/`:
   - `*_gi_openai_stub_smoke_v1.yaml` — fast, no API.
   - `*_gi_openai_provider_smoke_v1.yaml` — real provider, small dataset.
   - `*_kg_summary_bullets_smoke_v1.yaml` — frozen summary inputs (if Phase 0 chooses).
   - `*_kg_provider_smoke_v1.yaml` — provider extraction.

---

## Phase 2 — Reference artifacts and layout

1. **Directory convention** (mirror existing patterns):
   - `data/eval/references/gold/gil/<reference_id>/<episode_id>.json`
   - `data/eval/references/gold/kg/<reference_id>/<episode_id>.json`
   - Optional silver: `data/eval/references/silver/gil_<reference_id>/` (promoted run).
2. **Reference README + index** files (same governance as NER gold).
3. **Materialization:** Reuse `data/eval/materialized/<dataset_id>/` transcripts; no change
   unless GI/KG eval needs **frozen companion inputs** (e.g. `summary_ref.json` per
   episode) — if so, extend dataset meta or add `data/eval/materialized/.../extras/`.

---

## Phase 3 — Runner integration (`run_experiment.py`)

1. **Branch on `cfg.task`** — add **`grounded_insights`** and **`knowledge_graph`** as
   **separate** branches (alongside summarization and NER). Do not merge GI and KG into one
   branch or one prediction row that holds both artifacts.
2. **Summarization paths stay summary-only** — keep `generate_gi=False` (and no KG) on
   existing summarization experiment `Config` builds. GI task uses a **GI-focused** `Config`
   (e.g. `generate_gi=True`, KG off); KG task uses a **KG-focused** `Config` (GI off unless
   the product pipeline requires GI for that code path — if so, isolate in KG branch only).
3. **Inference adapter (per task):**
   - Build `Config` from experiment YAML + env (API keys).
   - **GI task:** production-equivalent GIL dict matching `gi.schema.json` →
     `predictions.jsonl` under e.g. `output.gil`.
   - **KG task:** production-equivalent KG dict matching `kg.schema.json` →
     `predictions.jsonl` under e.g. `output.kg`.
4. **Modes from Phase 0 (optional, per task):** e.g. regenerate summary inside a **KG-only**
   run vs frozen summary file — implemented as **params on that task’s YAML**, not as a
   shared multi-capability mode.
5. **`find_reference_path` / scoring:** Add `references/gold/gil/` and
   `references/gold/kg/` (and silver/baseline naming **per capability**).

---

## Phase 4 — Scoring and reports

1. **New modules** (suggested):
   - `src/podcast_scraper/evaluation/gi_scorer.py` — schema validation, grounding rate,
     quote-in-transcript checks, optional threshold stats from artifact.
   - `src/podcast_scraper/evaluation/kg_scorer.py` — schema validation, node/edge set
     metrics, optional label normalization.
2. **Wire into** existing metrics aggregation (`metrics.json`, `metrics_report.md`) the
   same way NER and summarization reports are built.
3. **`score_only` mode** must work for GI/KG once `predictions.jsonl` exists.

---

## Phase 5 — Tooling and governance

1. **Makefile** — keep **`make experiment-run CONFIG=...`** as the single entrypoint; use
   **separate YAML files** for GI vs KG (`id` matches filename). Add dedicated Make targets
   only if ergonomics demand it (optional).
2. **Docs:**
   - Update `data/eval/README.md` and `data/eval/configs/README.md` (repo root) with new tasks.
   - Add a subsection to [EXPERIMENT_GUIDE.md](../guides/EXPERIMENT_GUIDE.md) for GI/KG.
3. **Optional CI:** Non-secret smoke (stub / schema-only) on a tiny dataset; gate
   expensive provider runs on manual or scheduled jobs.

---

## Phase 6 — Consolidation (later)

- Align **metric names** with `docs/wip/METRICS_GAP_ANALYSIS.md` if still active.
- Consider **promote_run.py** / baseline promotion extensions for GI/KG.
- Cross-link **acceptance** thresholds with eval thresholds to avoid two divergent
  definitions of “pass.”

---

## Suggested work order (milestones)

| Milestone | Outcome |
| --------- | ------- |
| M1 | **Shared:** `ExperimentConfig` task literals + docs; optional explicit `task` passed into `score_run()` (helps all tasks). |
| M2–M3 | **GI only:** GI inference → JSONL; `gi_scorer` + `metrics_gil_v1`; gold under `gold/gil/`; GI baseline optional. |
| M4–M5 | **KG only:** KG inference → JSONL; `kg_scorer` + `metrics_kg_v1`; gold under `gold/kg/`; KG baseline optional. |
| M6 | **Per task (optional):** frozen-summary mode on **GI** YAML and/or **KG** YAML independently. |

KG work can start after M1 even if GI is incomplete (parallel tracks), as long as shared
runner hooks land in M1.

---

## Risks and mitigations

| Risk | Mitigation |
| ---- | ---------- |
| Eval becomes coupled to full workflow | Keep transcript-in / artifact-out adapter; optional frozen summary sidecar |
| Flaky LLM outputs break regression | Temperature 0, fixed prompts, use smoke datasets; separate “drift” from “break” |
| Large JSONL files | Store full artifact path or gzip sidecar policy; document in README |
| Duplicate Config fields | Shared helper: `experiment_config_to_runtime_config(cfg, task)` |

---

## Open questions

1. Should GI eval **require** NLI/QA providers for scoring, or only substring grounding
   checks on quotes?
2. For KG, is **topic/entity string equality** enough for v1, or do we need normalized IDs
   in references?
3. **NER in KG runs:** Default stance — **no**; KG eval is standalone like other tasks.
   Optional diagnostic joins (NER predictions + KG) are a later enhancement if needed.

---

**Status:** WIP plan — GI and KG are **independent** eval capabilities; adjust Phase 0
decisions here as you implement.
