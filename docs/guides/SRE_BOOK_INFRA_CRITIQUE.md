# SRE Book Infra Critique Guide

A sparring partner for **reliability, operations, and infra design**. When asked to critique a runbook,
a deploy path, monitoring, or on-call practice, score it against these themes and return specific,
actionable changes — preferably as diffs, checklist edits, or workflow YAML.

This rubric is **inspired by the ideas in** Google’s *Site Reliability Engineering* (O’Reilly /
Google; free online). It is **not** a summary of the book and **not** a substitute for reading it.
Canonical text and deeper treatment: [Site Reliability Engineering — table of contents](https://sre.google/sre-book/table-of-contents/).

For chart and dashboard critique, use [Tufte Chart Critique](TUFTE_CHART_CRITIQUE.md) instead.

---

## How to Use This File

In Cursor: `@docs/guides/SRE_BOOK_INFRA_CRITIQUE.md` or `@SRE_BOOK_INFRA_CRITIQUE.md`, then paste or
describe the artifact (runbook section, workflow, alert rule, Terraform module, incident notes).

Ask things like:

- *"Critique this deploy sequence against the SRE infra rubric"*
- *"Does our alerting match symptom-based paging?"*
- *"Quick SRE check — what's the biggest gap in this runbook?"*

**Repo anchors** (this checkout’s narrative, not part of the book): [Hosting and infrastructure](../architecture/HOSTING_AND_INFRASTRUCTURE.md), [Prod runbook](PROD_RUNBOOK.md), [CI workflows](../ci/WORKFLOWS.md), [DR drill runbook](DR_DRILL_RUNBOOK.md).

---

## The 7 Themes (with scoring)

Each theme scores **Pass / Warn / Fail**. At the end, give an overall verdict and a prioritized
fix list.

---

### 1. SLIs Tied to User Journeys

> Measure what **users** experience (availability, latency, correctness of a critical path), not
> only what is easy to scrape inside the box.

**Check for:**

- Health checks that prove “process up” but not “user can complete the task”
- Metrics with no link to a concrete user or operator story
- “Green” infra while the product path is broken

**Pass:** SLIs are named, measurable, and map to a journey (e.g. “viewer loads corpus”, “API
returns digest for episode X”).
**Warn:** Good internal metrics; user journey is implied but not written down.
**Fail:** Only CPU/disk/ping; no notion of what “good” means for a human.

---

### 2. SLOs That Can Be Judged

> An SLO is a **target + window + measurement method**. If three people can’t agree whether you
> met it, it is not an SLO yet.

**Check for:**

- “We aim for high availability” without numbers or period
- No aggregation or exclusion policy (what counts as “bad”?)
- SLO documents that never appear in incident or release discussion

**Pass:** Written SLO (or interim objective), measurement defined, reviewed when things change.
**Warn:** Informal targets; measurement partly manual.
**Fail:** No written bar; success is vibes.

---

### 3. Error Budget as Policy

> Reliability is a **product decision**: shipping and hardening trade off. The error budget makes
> that trade explicit.

**Check for:**

- Deploy cadence and risk unrelated to recent stability
- “Always add more gates” or “always ship faster” with no link to observed burn
- Post-incident actions that ignore whether budget was exhausted

**Pass:** Team can say how much risk appetite remains and how that affects change policy this week.
**Warn:** Budget concept understood but not written or not used in decisions.
**Fail:** Every incident is a surprise; no shared sense of acceptable failure rate.

---

### 4. Toil Budget

> **Toil** is manual, repetitive, interrupt-driven work that scales linearly with service growth.
> It should shrink as you automate and simplify.

**Check for:**

- Runbook steps that are copy-paste across hosts with no script or workflow
- On-call tasks that are identical every week with no ticket to eliminate root cause
- “We’ll document it” without a path to automation or deletion

**Pass:** Toil is listed, time-boxed, and has owners and elimination work.
**Warn:** Some toil accepted temporarily with dated plan.
**Fail:** Heroic manual ops normalized; bus factor one.

---

### 5. Monitoring and Alerting: Symptoms, Action, Ownership

> Pages should be **urgent, important, actionable, and owned**. Prefer symptoms users see over
> causes only engineers infer.

**Check for:**

- Alerts that fire when nothing needs doing until business hours
- Alerts with no runbook link or no clear first action
- Dashboards nobody looks at during incidents

**Pass:** Alert → symptom → runbook step → owner; paging is rare and trusted.
**Warn:** Some noisy alerts; cleanup backlog exists.
**Fail:** alert fatigue; on-call dreads the phone.

---

### 6. Change Management Matches Risk

> **Most outages follow change.** Progressive delivery, gates, and fast rollback beat “big bang
> confidence.”

**Check for:**

- Single step that flips prod with no prior gate on representative workload
- No documented rollback or rollback untested
- CI green but no path that exercises prod-like compose (see [ADR-082](../adr/ADR-082-gitops-app-deploy-via-stack-test-and-gha.md), [ADR-085](../adr/ADR-085-ephemeral-stack-test-integration-gate.md))

**Pass:** Change is batched, gated, observable, reversible; blast radius is understood.
**Warn:** Good habits on paper; occasional exceptions without review.
**Fail:** “SSH and hope”; secrets in chat; prod-only debugging as default.

---

### 7. Incident Learning Loop

> Incidents are **data**. Blameless postmortems produce **bounded** action items with owners and
> dates — not infinite process.

**Check for:**

- Repeat incidents with no tracking of “why it happened again”
- Postmortems that name people instead of systems and pressures
- Action items that are vague (“be more careful”)

**Pass:** Timeline, root causes, contributing factors, actions with owners; follow-up verified.
**Warn:** Lightweight retro for small events; major ones get full treatment.
**Fail:** shame culture or no written record.

---

## Critique Output Format

When critiquing an artifact, return this structure:

```text
## SRE Infra Critique

| Theme                 | Score | Notes                                      |
|-----------------------|-------|--------------------------------------------|
| SLIs / user journeys  |       |                                            |
| SLOs / measurability  |       |                                            |
| Error budget / policy |       |                                            |
| Toil                  |       |                                            |
| Monitoring / alerts   |       |                                            |
| Change management     |       |                                            |
| Incident learning     |       |                                            |

**Verdict:** [one paragraph]

## Priority Fixes

1. [CRITICAL] …
2. [HIGH] …
3. [MEDIUM] …
4. [LOW] …

## Suggested edits

[concrete runbook bullets, workflow snippet, or checklist — not generic advice]
```

---

## Quick Reference Cheatsheet

| Avoid | Prefer |
| ----- | ------ |
| “Highly available” | SLO numbers + measurement window |
| Paging on every anomaly | Paging on user-visible symptom + clear owner |
| Infinite manual checklist | Script, workflow, or delete the step |
| Big-bang deploy | Gates, progressive steps, tested rollback |
| “Someone should fix that” | Named owner + date + verification |
| Internal metric only | SLI tied to a user or operator story |
| Postmortem without actions | Few, tracked, completed actions |

---

## Tone for Critiques

Be direct. Good SRE writing is blunt about risk and tradeoffs. Do not replace “this fails the
error-budget idea” with “we might consider eventually improving reliability culture.” Name the gap,
cite which theme it violates, and propose a **specific** change (text, YAML, script, or policy).

If the system is safe but operationally immature, say so: reliability is **both** uptime and the
**sustainable** ability to change and learn.

---

## See Also

- [Chip Huyen ML / AI Critique](CHIP_HUYEN_ML_AI_CRITIQUE.md) — data, eval, foundation-model
  product design, production economics
- [Tufte Chart Critique](TUFTE_CHART_CRITIQUE.md) — visualization and dashboards
- [Hosting and infrastructure](../architecture/HOSTING_AND_INFRASTRUCTURE.md) — how this repo’s
  always-on stack is meant to fit together
- [Prod runbook](PROD_RUNBOOK.md) — operator procedures
- [CI workflows](../ci/WORKFLOWS.md) — what automation actually runs
