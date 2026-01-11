# AI Quality & Experimentation: Phase 1 Implementation & Validation Plan

This document governs the first phase of the **AI Quality & Experimentation Platform (PRD-007)**. It provides a technical roadmap and strict validation gates for **RFC-016 (Phase 2)**, **RFC-015 (Phase 1)**, and **RFC-041 (Phase 0)**.

- **Objective**: Move from manual "eyeball" comparisons to a data-first product driven by objective, reproducible benchmarking.
- **Status**: 📋 WIP Plan
- **Primary Goal**: Establish the "Sacred Baseline" and the universal evaluation harness.

---

## 1. Validation Principles (Rules of the Game)

Before implementation begins, these principles must be adopted:

### Principle A — Validation ≠ Tests

Unit tests are necessary but not sufficient. Validation means answering: *"Can this step be misused, bypassed, or misunderstood—and does the system stop that?"*

### Principle B — Every Phase Has a Hard Invariant

Each phase introduces one non-negotiable architectural or product invariant. If the invariant can be violated (e.g., running an experiment without a baseline), the phase is not done.

### Principle C — Validation is Negative Testing

We validate by attempting to:

- Omit required fields.
- Compare incompatible artifacts (cross-dataset).
- Run things in the wrong order or with "fuzzy" parameters.

---

## 2. Step-by-Step Roadmap

---

### Step 1 — RFC-016 Phase 2 (The Reproducible Foundation)

**Highest ROI: Typed Params, Fingerprinting, and Preprocessing Profiles.**

#### 🏗️ Technical Scope

- **`src/podcast_scraper/providers/params.py`**: Define Pydantic models for `SummarizationParams`, `TranscriptionParams`, and `SpeakerDetectionParams`.
- **`ProviderFingerprint`**: Capture model names, versions, device, precision, and git state in every result.
- **`PreprocessingProfile`**: Registry of versioned cleaning logic (e.g., `cleaning_v3`).

#### 🔒 Core Invariant
>
> **"Every provider run produces a fully fingerprinted, typed, and reproducible artifact."**

#### ✅ Validation Gates

| Gate | How to Validate | Expected Failure |
| :--- | :--- | :--- |
| **Malformed Params** | Create a config with `chunk_size: "large"` or unknown fields. | ❌ Immediate failure at config load with clear type errors. |
| **Explicit Defaults** | Run provider with no params specified; inspect fingerprint. | ❌ FAIL if fingerprint contains `null` or "default" instead of concrete values used. |
| **Fingerprint Sensitivity** | Run same input 3x changing only model name, then device, then profile. | ❌ FAIL if two materially different runs produce identical fingerprint hashes. |
| **Preprocessing Visibility** | Run with `cleaning_v2` vs `cleaning_v3` and check metadata. | ❌ FAIL if the preprocessing profile is "invisible" in the output artifacts. |

---

### Step 2 — RFC-015 Phase 1 (The Universal Harness)

**Minimal Experiment Runner with Contract Enforcement.**

#### 🏗️ Technical Scope

- **`scripts/run_experiment.py`**: Minimal runner that only does ingestion, validation, and artifact emission.
- **Contract Enforcement**: Mandatory `dataset_id`, `baseline_id`, and `golden_required` fields.
- **Structured Artifacts**: deterministic directory structure (`results/<run_id>/`).

#### 🔒 Core Invariant
>
> **"An experiment cannot run unless it is objectively comparable."**

#### ✅ Validation Gates

| Gate | How to Validate | Expected Failure |
| :--- | :--- | :--- |
| **Contract Enforcement** | Try to run an experiment without `dataset_id` or `baseline_id`. | ❌ Run is blocked before execution with a contract violation error. |
| **Dataset Identity** | Compare experiment on `Dataset A` against a baseline on `Dataset B`. | ❌ Explicit "Invalid Comparison" error; runner refuses to emit scores. |
| **Structured Outputs** | Execute one run and inspect the directory. | ❌ FAIL if outputs are "just logs" or unstructured JSON without fingerprints. |
| **Determinism Check** | Run identical config twice and diff the artifacts. | ❌ FAIL if results drift randomly or fingerprints differ between identical runs. |

---

### Step 3 — RFC-041 Phase 0 (The Sacred Baseline)

**Freezing Datasets and Establishing First Baselines.**

#### 🏗️ Technical Scope

- **`benchmarks/datasets/*.json`**: Canonical definitions of `indicator_v1`, `shortwave_v1`, and `journal_v1`.
- **Baseline Materialization**: Running the current `main` branch to produce frozen artifacts in `benchmarks/baselines/`.
- **Baseline Changelog**: Documenting why a baseline was created and what it represents.

#### 🔒 Core Invariant
>
> **"Baselines are frozen, reviewable, and impossible to mutate accidentally."**

#### ✅ Validation Gates

| Gate | How to Validate | Expected Failure |
| :--- | :--- | :--- |
| **Single Source of Truth** | Try to run the runner using raw glob paths, bypassing dataset JSON IDs. | ❌ Rejected; runner must only accept valid Dataset IDs from the registry. |
| **Baseline Immutability** | Try to overwrite an existing baseline or re-run with the same ID. | ❌ Hard failure; the system requires a new versioned ID for any change. |
| **Explainability** | Pick a baseline artifact and ask: *"What commit and model produced this?"* | ❌ FAIL if you need tribal knowledge; metadata must contain all answers. |

---

## 3. Looking Ahead: Quality Gates & CI

#### 🔒 Core Invariant
>
> **"Known failure modes (speaker leaks, boilerplate) cannot reach main silently."**

#### ✅ Validation Gates

| Gate | How to Validate | Expected Failure |
| :--- | :--- | :--- |
| **Negative Regression** | Inject a bad summary (e.g., with "credits:" text) into an experiment run. | ❌ CI failure; clear "Boilerplate Leak" gate violation message. |
| **Severity Enforcement** | Trigger a "Minor" (warning) vs "Critical" (blocking) gate. | ❌ FAIL if all gates behave identically (e.g., all block or all warn). |
| **Actionable Feedback** | Read CI output as if you were an outside contributor. | ❌ FAIL if the output is just red text without "How to fix" context. |

---

## 4. The Ultimate Sanity Check

Ask this single question after each phase is "done":

> **"Can a careless but well-meaning engineer accidentally invalidate our results?"**

- **If YES** → The phase is not done. Add more validation logic.
- **If NO** → Move forward to the next step.

---

## 📈 Phase 1 Deliverables Checklist

### Step 1 (RFC-016 P2)

- [ ] `src/podcast_scraper/providers/params.py` exists with typed Pydantic models.
- [ ] Every provider result contains a `fingerprint.json`.
- [ ] `PreprocessingProfile` registry implemented in `src/podcast_scraper/preprocessing/`.

### Step 2 (RFC-015 P1)

- [ ] `scripts/run_experiment.py` enforces the Experiment Contract.
- [ ] Runner detects and blocks cross-dataset comparisons.
- [ ] Experiment results use a deterministic, versioned folder structure.

### Step 3 (RFC-041 P0)

- [ ] Dataset JSONs for 3 content regimes committed.
- [ ] First `benchmarks/baselines/` artifact frozen and reviewed.
- [ ] `benchmarks/CHANGELOG.md` initialized.
