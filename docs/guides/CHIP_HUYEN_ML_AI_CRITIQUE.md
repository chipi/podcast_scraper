# Chip Huyen ML / AI Critique Guide

A sparring partner for **ML and AI product design**: datasets, model choice, RAG, evaluation,
deployment, monitoring, and cost. When asked to critique a pipeline, provider setup, eval plan,
or “LLM feature” design, score it against these themes and return specific, actionable changes —
preferably as config diffs, test additions, or documented decisions.

This rubric is **inspired by the ideas in** Chip Huyen’s books (O’Reilly):

- *Designing Machine Learning Systems* (2022) — end-to-end ML systems, data, training/serving,
  iteration.
- *AI Engineering: Building Applications with Foundation Models* — prompts, RAG, fine-tuning,
  agents, evaluation under open-ended outputs, production tradeoffs.

It is **not** a summary of either book and **not** a substitute for reading them. Author page and
pointers: [huyenchip.com — Books](https://huyenchip.com/books/).

For charts and dashboards, use [Tufte Chart Critique](TUFTE_CHART_CRITIQUE.md). For runbooks,
deploy paths, and reliability culture, use [SRE Book Infra Critique](SRE_BOOK_INFRA_CRITIQUE.md).

---

## How to Use This File

In Cursor: `@docs/guides/CHIP_HUYEN_ML_AI_CRITIQUE.md` or `@CHIP_HUYEN_ML_AI_CRITIQUE.md`, then
paste or describe the artifact (profile YAML, provider choice, eval protocol, RAG sketch, agent
flow, monitoring plan).

Ask things like:

- *"Critique this summarization profile against the Chip Huyen rubric"*
- *"Is our eval plan honest for open-ended LLM outputs?"*
- *"Quick ML-systems check — where’s the biggest architectural smell?"*

**Repo anchors:** [Experiment Guide](EXPERIMENT_GUIDE.md), [AI Provider Comparison](AI_PROVIDER_COMPARISON_GUIDE.md), [Pipeline and Workflow](PIPELINE_AND_WORKFLOW.md), [Configuration API](../api/CONFIGURATION.md).

---

## Two lenses (same seven themes)

Chip Huyen’s material spans **offline design** and **live systems**; the themes below are shared.
You do **not** need two separate repo files unless you strongly want two different `@` mentions in
the editor — splitting duplicates the “evaluation vs deployment” bridge and doubles maintenance.

**When to emphasize which themes**

| Lens | Typical artifact | Emphasize themes |
| ---- | ---------------- | ---------------- |
| **Experiments and offline** | Held-out sets, baselines, promotion rules, prompt/RAG sketches before ship | **1–4** (outcomes, data, adaptation tier, eval honesty) |
| **Production code and operations** | Pipeline code paths, retries, monitoring, operator UX, cost in prod | **4–7** (eval that matches prod reality, economics, drift, failure UX) |

**Theme 4** is the hinge: offline eval can be rigorous yet still lie about production if latency,
tooling, and provider failures are never exercised. For a **full** design review, score all seven.

---

## The 7 Themes (with scoring)

Each theme scores **Pass / Warn / Fail**. At the end, give an overall verdict and a prioritized
fix list.

---

### 1. Outcomes Before Models

> Pick architectures from **measurable product or research outcomes**, not from the most
> impressive model card.

**Check for:**

- “We’ll use the biggest model” before the success metric exists
- Offline accuracy celebrated when the user-facing task is different
- No explicit **good enough** bar (latency, cost, quality floor)

**Pass:** Metric + threshold stated; model tier is a dependent variable.
**Warn:** Metrics exist but live only in chat, not in code or docs.
**Fail:** Model name is the requirement.

---

### 2. Data and Context Engineering

> Garbage in, garbage out — **including** prompts, retrieved chunks, and training/eval splits.

**Check for:**

- Leakage (eval episodes in train context, future information in “historical” features)
- Stale corpus, unversioned snapshots, no reproducible “what text did the model see?”
- RAG without chunking strategy, citation discipline, or failure when retrieval misses

**Pass:** Data lineage and refresh story are documented; eval held-out is honest.
**Warn:** Mostly sound; a few gray zones called out with backlog.
**Fail:** “We’ll eyeball it”; same rows in train and eval.

---

### 3. Right Adaptation Tier

> **Prompting, RAG, fine-tuning, and model swap** are tools with different cost, risk, and
> maintenance profiles — not a ladder you climb by default.

**Check for:**

- Fine-tuning proposed when prompt + eval tightening would suffice
- RAG bolted on without measuring retrieval precision/recall for the task
- Agent + tools where a single batched call would meet SLO

**Pass:** Alternatives considered; choice tied to metrics and ops burden.
**Warn:** Reasonable default with a dated plan to validate cheaper tiers.
**Fail:** Complexity is the feature.

---

### 4. Evaluation Matches Deployment

> Metrics must reflect **real inputs, real latency, and real failure modes** — especially for
> open-ended generation.

**Check for:**

- Single-number leaderboard without variance, slice, or human spot-check
- Tests that never exercise timeouts, empty retrieval, or provider errors
- “LLM-as-judge” without calibration, baseline, or dispute handling

**Pass:** Offline + targeted online/human checks; regressions block promotion where agreed.
**Warn:** Good offline suite; prod behavior partially unknown.
**Fail:** Ship on demo vibes.

---

### 5. Production Economics

> **Latency, throughput, and $ per unit** are part of the design — not post-hoc surprises.

**Check for:**

- No budget for tokens, GPU minutes, or API calls per episode
- Synchronous chains that violate viewer or batch SLOs
- Missing caching, idempotency, or backoff where providers rate-limit

**Pass:** Cost and latency estimated per path; fallbacks defined.
**Warn:** Estimates rough but tracked after launch.
**Fail:** “We’ll optimize later” with no measurement hook.

---

### 6. Monitoring, Drift, and Iteration

> Models **decay** when the world and data shift; systems need feedback loops, not launch-day
> optimism.

**Check for:**

- No slice of quality by feed, language, length, or provider
- Retrain or prompt-update triggers undefined
- Incidents that don’t update eval fixtures or acceptance tests

**Pass:** Dashboards or jobs tied to product metrics; change log when pipeline shifts.
**Warn:** Manual periodic review; partial automation.
**Fail:** Ship and forget.

---

### 7. Failure Modes and User Experience

> When the model is wrong, **something predictable still happens** — for operators and end users.

**Check for:**

- Hallucinated facts presented without grounding or confidence cues where needed
- Agent loops without max steps, tool allowlists, or circuit breakers
- Opaque errors to the user when the provider fails

**Pass:** Degraded modes documented; dangerous outputs bounded by design.
**Warn:** Some sharp edges; mitigations planned.
**Fail:** Best-case-only UX.

---

## Critique Output Format

Use **one** of the shapes below (full is default when both offline and prod matter).

### Full (all themes)

```text
## Chip Huyen ML / AI Critique

| Theme                     | Score | Notes                          |
|---------------------------|-------|--------------------------------|
| Outcomes before models    |       |                                |
| Data / context            |       |                                |
| Adaptation tier           |       |                                |
| Evaluation vs deployment  |       |                                |
| Production economics      |       |                                |
| Monitoring / drift        |       |                                |
| Failure modes / UX        |       |                                |

**Verdict:** [one paragraph]

## Priority Fixes

1. [CRITICAL] …
2. [HIGH] …
3. [MEDIUM] …
4. [LOW] …

## Suggested edits

[concrete YAML, test cases, eval protocol, or doc bullets — not generic advice]
```

### Experiments and offline only (themes 1–4)

```text
## Chip Huyen critique — experiments / offline

| Theme                    | Score | Notes |
|--------------------------|-------|-------|
| Outcomes before models   |       |       |
| Data / context           |       |       |
| Adaptation tier          |       |       |
| Evaluation vs deployment |       |       |

**Verdict:** …
## Priority Fixes
…
```

### Production code and operations only (themes 4–7)

```text
## Chip Huyen critique — production / live

| Theme                    | Score | Notes |
|--------------------------|-------|-------|
| Evaluation vs deployment |       |       |
| Production economics     |       |       |
| Monitoring / drift       |       |       |
| Failure modes / UX       |       |       |

**Verdict:** …
## Priority Fixes
…
```

---

## Quick Reference Cheatsheet

| Avoid | Prefer |
| ----- | ------ |
| Model-first requirements | Outcome metrics + thresholds first |
| One aggregate score | Slices, variance, spot-checks |
| Default to largest LLM | Cheapest tier that meets the bar |
| Train/eval overlap | Explicit splits and lineage |
| RAG without retrieval metrics | Measure recall/precision for the task |
| Launch without cost estimate | Per-unit $ and latency envelope |
| Silent wrong answers | Grounding, bounds, or honest degradation |

---

## Tone for Critiques

Be direct. ML and AI systems fail in boring, expensive ways — data bugs, skew, eval lying,
unbounded agents. Name the risk, tie it to a theme above, and propose a **specific** mitigation
(test, config, process). Do not substitute stack hype for engineering.

---

## See Also

- [Tufte Chart Critique](TUFTE_CHART_CRITIQUE.md)
- [SRE Book Infra Critique](SRE_BOOK_INFRA_CRITIQUE.md)
- [Experiment Guide](EXPERIMENT_GUIDE.md)
- [Provider Deep Dives](PROVIDER_DEEP_DIVES.md)
