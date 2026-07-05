# RFC-086: Operator Data-Access & Tooling Architecture

- **Status**: Draft
- **Authors**: Marko
- **Companion PRD**: `PRD-035-operator-control-room.md`
- **Layer**: internal / Pro — not OSS.
- **Depends on**:
  - `RFC-081-personal-knowledge-layer-reconciliation.md` — substrate (decay fn, reconcile
    join, changelog, L2 derivation) reused read-only
  - `RFC-072-canonical-identity-layer.md` — canonical refs for impact preview
- **Relates to**:
  - `RFC-082` (reserved) — the *maintained* reverse index for the live tier; this RFC's
    reverse lookups are read-only/on-demand and do not require it

> **Proposed numbering** — `RFC-086` / `PRD-035` are placeholders; verify and renumber.

---

## Summary

A **read-mostly operator layer** over the Knowledge Retention substrate, plus three new
operator capabilities — **parameter simulation**, an **isolated synthetic sandbox**, and
**CIL impact preview** — all behind a gated, audited `/api/ops/*` surface. The operator layer
**never becomes a parallel source of truth**: it reuses RFC-081's decay function, reconcile
join, changelog, and L2 derivation, exposing them at a *cross-user* / *any-user* access scope
that the user-scoped endpoints do not provide.

---

## Motivation

The v1 substrate is correct-by-test but opaque-in-operation (PRD-035 §Background). The
operator needs to (a) observe the machine across users, (b) simulate decay params before
committing, (c) test reconcile/decay/R1 without real data, and (d) preview CIL blast radius.
These are **different access patterns** from the user-facing layer — aggregate / any-user /
isolated — which is why they get their own thin layer rather than being bolted onto the
per-user endpoints.

---

## Principles

1. **Read-mostly, reuse the substrate.** Operator reads call the *same* derivation/reconcile
   code paths as users, at operator scope. No reimplementation, no second truth.
2. **No operator writes to user ground truth.** The operator does not hand-edit anchors or
   L2. Corrections flow through existing pipelines (CIL, enrichers).
3. **Isolation for synthetic data.** Sandbox anchors/changelog are namespaced and can never
   reach real users or the real changelog.
4. **Gate + audit cross-user / PII reads.** Any read that crosses user boundaries or exposes
   a user's L2 is operator-role-gated and logged.
5. **On-demand over maintained where possible in v1.** Reverse lookups are computed on demand;
   the maintained reverse index is deferred to the live tier (RFC-082).

---

## Operator data access

### Cross-user / any-user read projections

RFC-081's `/l2/*` and `/reconcile` are **caller-scoped** (the current user). The operator
needs the same computations for an **arbitrary** user or across users. Rather than widen the
user endpoints (privacy risk), expose operator-scoped mirrors under `/api/ops/*` that take an
explicit `user_id` (or sandbox scope) and run the identical derivation, behind the operator
gate.

### Reverse lookup: `node_ref → anchors → users`

Needed by **CIL impact preview** and **inspect**. In v1 this is an **on-demand** query over
the `anchors` table (`WHERE node_ref ∈ targets`), not a maintained index. It is read-only and
bounded by how many anchors reference a node. The *maintained* reverse index (`node_ref →
active users`) is a different object owned by RFC-082 (live fan-out) — do not build it here.

### λ/θ simulation (pure, no commit)

`simulate(params, scope)` applies a **candidate** decay function to an existing anchor set
without persisting:

```
simulate(λ', θ', scope=user|cohort|sandbox):
    for each anchor in scope:
        s' = base · exp(-λ'·Δt) · f(reinforce_count)     # candidate params, not stored
    working_set' = { node : aggregate s' > θ' }
    return { curve(λ'), working_set', delta_vs_current, active_set_size }
```

Reuses RFC-081's decay formula with override params. Promotion to real config is a **separate
explicit operator action**, not a side effect of simulate.

### Synthetic sandbox (isolation)

A sandbox lets the operator inject fake anchors + changelog entries and run reconcile / decay
/ simulate / R1 probes with **no real-world effect**. Isolation options (decide in build):

- **Scoped namespace** — sandbox rows carry a `sandbox_id`; all operator computations accept
  a scope of `user_id` *or* `sandbox_id`; the real changelog/anchors are never read or written
  in sandbox scope. (Preferred — minimal schema, reuses the same code paths.)
- Alt: a parallel schema/prefix.

Hard rule: **a sandbox run never appends to the real changelog and never mutates real
anchors/watermarks.** Sandbox reconcile uses a sandbox changelog + sandbox watermark only.

### Anchor-resolution trace

For one anchor, return the GIL provenance path the async resolution job (PLAN M1) walked to
discover canonical targets, with per-step confidence — so mis-resolutions are visible
(faithfulness is the critical quality dimension).

### CIL impact preview

`impact(ref, op=merge|split, args)` = reverse-lookup the affected `node_ref`(s) → count
anchors + distinct users, and produce a **diff** of post-op canonical refs (which anchors
would re-point where). Read-only preview; the actual CIL op runs through its existing path,
now gated by this preview.

---

## Operator API surface (all gated + audited)

| Method | Path | Purpose | FR |
|---|---|---|---|
| `GET` | `/api/ops/users/{id}/l2/graph` | inspect-as-user L2 overlay | G2 |
| `GET` | `/api/ops/changelog` | changelog feed; filters `node`,`type`,`since` | D1–D3 |
| `GET` | `/api/ops/anchors/search` | search anchor/L2 layer across users | S1 |
| `GET` | `/api/ops/node/{ref}/references` | reverse lookup: anchors+users referencing node | C1 |
| `GET` | `/api/ops/cil/{ref}/impact` | CIL merge/split blast-radius + diff | C1–C2 |
| `POST` | `/api/ops/decay/simulate` | simulate `(λ,θ)` vs real/sandbox anchor set | L1–L2 |
| `GET` | `/api/ops/anchors/{id}/resolution` | provenance resolution trace | AR1 |
| `POST` | `/api/ops/reconcile/inspect` | run join; matched-entries-and-why + cost; R1 probe | RI1–RI2 |
| `POST` | `/api/ops/sandbox` | create isolated sandbox (anchors+changelog) | RI3 |
| `GET` | `/api/corpus/coverage` | corpus coverage/density (unblocks flagged dep) | DB2 |

Metrics (FR-DB1) read the existing Langfuse/PostHog stores; no new endpoint.

---

## Access control & audit

- **Operator role** required for every `/api/ops/*` route; default-deny.
- **Audit log** for any cross-user or per-user-L2 read (who, which user, when, why).
- **Redaction** option for inspect-as-user where full content is unnecessary.
- **OSS exclusion:** this layer is compiled/served only in the internal/Pro build; it must
  not exist in the OSS distribution. Production-derived data stays private (GDPR + moat).

---

## Phasing

1. **P0 (with v1 milestones)** — operator read projections + `/api/ops/changelog` feed +
   `/api/ops/reconcile/inspect` (incl. R1 probe) + sandbox (`sandbox_id` scoping) + Graph
   L1/L2 overlay binding. These debug M3/M5/M6.
2. **P1** — `decay/simulate`; `anchors/{id}/resolution`; `node/{ref}/references` +
   `cil/{ref}/impact` (before first real CIL merge); Dashboard metrics + `/api/corpus/coverage`.
3. **P2** — `anchors/search`.

---

## Open questions

1. **Sandbox isolation mechanism** — `sandbox_id` scoping vs parallel schema. (Lean
   `sandbox_id`: reuses code paths, minimal schema, easy teardown.)
2. **Reverse lookup cost** — on-demand `WHERE node_ref ∈ targets` is fine until anchor volume
   on hot nodes grows; define the threshold at which a cached/maintained index (borrowing from
   RFC-082) becomes warranted.
3. **Audit model** — store location + retention for inspect-as-user audit, consistent with the
   private-data posture.
4. **inspect-as-user redaction default** — full L2 vs structure-only by default.

---

## Security note

`inspect-as-user` and cross-user reads are the highest-risk capability in the whole Knowledge
Retention design: they expose exactly the per-user L2 that is both GDPR-relevant and the
product's moat. Gate hard, audit always, redact by default where possible, and never let this
surface or its data into the OSS layer or any public artifact.
