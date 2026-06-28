# PRD-035: Operator Control Room (Knowledge Retention Viewer / Admin)

- **Status**: Draft
- **Authors**: Marko
- **Target Release**: alongside Knowledge Retention v1 (internal)
- **Layer**: **internal / Pro — NOT OSS.** Inspects per-user L2 and cross-user state;
  GDPR-sensitive and moat-bearing. Must not ship in the OSS layer.
- **Parent PRD**: `PRD-034-knowledge-retention-layer.md`
- **Architecture**: `RFC-086-operator-data-access-and-tooling.md`
- **Substrate**: `RFC-081-personal-knowledge-layer-reconciliation.md`
- **Host**: the existing `gi-kg-viewer` (Vue 3 + Cytoscape + Tailwind), surfaces: Graph,
  Library, Digest, Search, Explore, Dashboard.

> **Proposed numbering** — `PRD-035` / `RFC-086` are placeholders; verify against the repo
> tree and renumber before merge.

---

## Summary

Knowledge Retention introduces a **shared machine that no single user ever sees**: the
user-agnostic changelog, the decay engine, the reconcile join, anchor resolution, and CIL
impact. The **player app is the user's surface** (one person's L2); the **viewer must become
the operator's control room over L0 and that machine**. You cannot build, debug, or tune
what you cannot see — today this machinery is invisible.

This PRD specifies the operator surfaces, mostly as **extensions to existing viewer
surfaces** plus a small number of **new operator-only panels**. It deliberately does **not**
rebuild the consumer surfaces in the viewer.

---

## The two-surface principle (read this first)

| | Player app (PRD-034) | Operator viewer (this PRD) |
|---|---|---|
| Audience | one user | operator |
| Renders | **that user's L2** (derived) | **L0 + the shared machine** (+ any user's L2 on demand, gated) |
| Job | recall / maintain / grow | build · debug · tune · safeguard |
| Layer | OSS user product | internal / Pro |

Keep them separate codebases/surfaces. The operator viewer renders the substrate and the
machine; it is not a second consumer product. Where it shows a user's L2, that is an
**inspection** capability, gated and audited — not a feature.

---

## Background

The v1 build (PLAN M0–M8) stands up: anchors + async resolution, salience decay, the
changelog (extension), the reconcile join, and L2 derivation. Each of these is a moving part
whose correctness is asserted by tests but whose *behaviour over real data* is currently a
black box. Three things specifically have no operator visibility today:

1. **Is the machine working?** Is the changelog actually emitting? Does reconcile match the
   right anchors at bounded cost? Are anchors resolving to the right canonical targets?
2. **Is it tuned?** `λ`/`θ` are behavioural and must be **fit from data, not guessed** — but
   there is nowhere to simulate a candidate setting against real anchor sets before
   committing.
3. **Is it safe?** Anchors target canonical refs, so a bad CIL merge/split silently corrupts
   **every user's L2 and all cross-show reconciliation**. There is no impact preview.

The operator control room exists to make all three legible.

---

## Goals

1. Make the shared machine **observable**: changelog emission, reconcile behaviour, anchor
   resolution, decay/working-set.
2. Make `λ`/`θ` **tunable from evidence** via simulation before commit.
3. Make CIL changes **safe** via anchor-impact preview.
4. Provide a **dogfood / inspection** path (inspect-as-user) under strict access + audit.
5. Reuse existing viewer surfaces wherever possible; add new panels only for genuinely new
   operator objects.

---

## Non-Goals

- **Not** a re-implementation of the consumer surfaces (Map/Dossier/etc. as a user product).
- **Not** an OSS surface — this is internal/Pro tooling.
- **Not** a write path into user L1/L2 as ground truth (operator edits to CIL/enrichers flow
  through their existing pipelines; the operator does not hand-edit users' anchors).
- **Not** the live-reconcile reverse index (RFC-082) — operator reverse lookups are
  read-only and may be computed on demand in v1.

---

## Operator surfaces

Organised as **extend** (rides an existing viewer surface) vs **new** (a new operator
object). Each lists operator value, what it de-risks, requirements, and the operator
endpoint it binds to (defined in RFC-086). Priority in §Priority.

### Extend — Graph: L1/L2 overlay + inspect-as-user *(P0)*

**Value / de-risks.** Highest leverage — rides existing Cytoscape rendering. Visually
verifies the **induced-subgraph + decay weighting** and the **invariant** (wipe →
recompute → identical); the dogfood surface for user-zero.

**FR-G1** — On an L0 graph, toggle overlay layers: anchor markers, **node size = effective
salience**, cross-show halo, and the **changelog-driven conflict/staleness marker channel**
(inert in v1, wired per **R1** so contradiction lights up later with no change).
**FR-G2** — A **user selector** overlays a chosen user's working set (their active L2) on
L0. Gated + audited (see §Access).
**FR-G3** — `θ` preview slider: show how the working set grows/shrinks as the threshold
moves (does not commit config).
*Binds:* `GET /api/ops/users/:id/l2/graph`.

### Extend — Digest: operator changelog feed *(P0)*

**Value / de-risks.** The changelog is the reconcile substrate; this confirms **emission is
firing** (PLAN M3) and makes corpus evolution legible. Digest is already change-oriented.

**FR-D1** — Stream changelog entries `{ seq, node_ref, change_type, source, detected_at }`,
filterable by node / change-type / time window.
**FR-D2** — Per-node "evolution" drill: all changelog entries for one node over time.
**FR-D3** — Emission health: rate of entries, gaps, last-emitted-per-enricher.
*Binds:* `GET /api/ops/changelog`.

### Extend — Search: search the anchor / L2 layer *(P2)*

**FR-S1** — Search across the anchor layer (not just L0): "which users anchored entity X",
"anchors with unresolved/low-confidence targets", by node/type/salience.
*Binds:* `GET /api/ops/anchors/search`.

### Extend — Dashboard: M8 metrics + corpus coverage *(P1)*

**Value / de-risks.** Makes **cold-start** legible and provides the data to **fit `λ`/`θ`**.

**FR-DB1** — Metrics home: time-to-first-aha, anchors-per-session, **acknowledge vs dismiss
rates** (resurface + what-changed), deposit→resolve latency, Map opens + density.
**FR-DB2** — **Corpus coverage / density** view (unblocks the flagged
`GET /api/corpus/coverage` dependency; prerequisite for temporal/coverage-aware features).
*Binds:* existing metrics store (Langfuse/PostHog) + `GET /api/corpus/coverage`.

### New — Reconcile inspector + synthetic sandbox *(P0)*

**Value / de-risks.** The debug surface for M3/M6; verifies the **cost property** and **R1**;
the sandbox lets you test reconcile/decay/R1 with **no real data**.

**FR-RI1** — Pick a user (or sandbox) + watermark + types; run the join; show **which
changelog entries matched which anchors and why**, plus the per-user cost (active-set size).
**FR-RI2** — **R1 probe:** inject a synthetic unknown `change_type` and confirm it flows
through unfiltered end-to-end.
**FR-RI3** — **Synthetic sandbox:** inject fake anchors + changelog entries into an isolated
namespace; run reconcile/decay/simulate against them; never touches real users or the real
changelog (isolation per RFC-086).
*Binds:* `POST /api/ops/reconcile/inspect`, `POST /api/ops/sandbox`.

### New — λ/θ simulator *(P1)*

**Value / de-risks.** Directly serves the **`λ`/`θ` open question** and the biggest tuning
risk; decay params are behavioural and must be fit, not guessed.

**FR-L1** — Plot decay curves for candidate `λ` (half-life) and overlay `θ`.
**FR-L2** — **Simulate** a candidate `(λ, θ)` against a **real anchor set** (a user or
cohort) → resulting working set, what enters/leaves vs current, reconcile active-set size.
No commit; a separate explicit action promotes a setting to config.
*Binds:* `POST /api/ops/decay/simulate`.

### New — Anchor-resolution inspector *(P1)*

**Value / de-risks.** Faithfulness of resolution is the **critical quality dimension**; you
must *see* bad resolutions, not trust them.

**FR-AR1** — For one anchor, show the **GIL provenance path** the resolution walked to
discover its canonical targets, with confidence, so mis-resolutions are visible.
*Binds:* `GET /api/ops/anchors/:id/resolution`.

### New — CIL impact preview *(P1, safety)*

**Value / de-risks.** A bad merge/split corrupts every user's L2 + cross-show reconcile.
Build the guard in from the start.

**FR-C1** — Before any CIL merge/split in the admin, show **"N anchors across M users
reference this"** and a **diff preview** of the post-change canonical refs.
**FR-C2** — Block/confirm gate proportional to blast radius.
*Binds:* `GET /api/ops/node/:ref/references`, `GET /api/ops/cil/:ref/impact`.

---

## Priority & sequencing

| Add | Surface | De-risks | Priority | Rides v1 PLAN milestone |
|---|---|---|---|---|
| L1/L2 overlay + inspect-as-user | Graph (extend) | invariant, decay, dogfood | **P0** | alongside M5 (read surfaces) |
| Changelog feed | Digest (extend) | M3 emission black-box | **P0** | alongside M3 |
| Reconcile inspector + sandbox | new | M3/M6, R1, cost | **P0** | alongside M3/M6 |
| λ/θ simulator | new | the λ/θ open question | **P1** | after M2 (decay) + M8 data |
| Anchor-resolution inspector | Graph/Explore | faithfulness/grounding | **P1** | alongside M1 (resolution) |
| CIL impact preview | entity mgmt | catastrophic-merge safety | **P1** | before first real CIL merge |
| M8 metrics + coverage | Dashboard (extend) | cold-start legibility | **P1** | alongside M8 |
| Anchor/L2 search | Search (extend) | ops convenience | **P2** | post-v1 |

The P0 items **should be built alongside the named v1 milestones, not after** — they are how
those milestones are debugged. Treat them as part of the v1 build loop.

---

## Access, privacy & safety

- **inspect-as-user (FR-G2) is the sensitive capability.** It exposes per-user L2 →
  GDPR-relevant and moat-bearing. Gate behind an operator role; **audit every access**;
  support redaction; **never expose in OSS**; never leak production-derived data into public
  artifacts (consistent with the private-eval/PostHog-self-host posture).
- **Synthetic sandbox is strictly isolated** — fake data never reaches real users or the
  real changelog (RFC-086 isolation).
- **Operator API is namespaced + gated** (`/api/ops/*`, operator-only).
- **CIL impact gate** is a safety control, not a convenience — it stands between an admin
  action and corruption of all users' L2.

---

## Success Criteria

1. An operator can watch a deposit → resolution → changelog → reconcile flow **end-to-end**
   in the viewer and confirm each stage fired correctly.
2. The **invariant** is visually verifiable: overlay a user's L2, force recompute, confirm
   identical.
3. A candidate `(λ, θ)` can be **simulated against a real anchor set** and its working-set
   effect inspected **before** any config change.
4. The **R1 probe** demonstrates an injected unknown change-type rendering unfiltered.
5. A CIL merge shows **anchor blast radius across users** before it is applied.
6. None of this tooling is present in the OSS layer; inspect-as-user is gated + audited.

---

## Spec map & dependencies

- **RFC-086** — operator data-access layer (cross-user read projections, reverse lookup,
  simulation, sandbox isolation, access control, the `/api/ops/*` surface).
- **RFC-081** — the substrate being observed (decay fn, reconcile join, changelog, L2
  derivation reused read-only).
- **RFC-072 (CIL)** — impact preview reads canonical refs + the reverse lookup.
- **RFC-082 (reserved)** — the *maintained* reverse index for the live tier; operator
  reverse lookups are read-only/on-demand and do **not** require it in v1.
- **`gi-kg-viewer` + RFC-077** — host + graph modes for the overlay.
- **No new UXS** in v1 — operator surface anatomy is carried here; spawn a UXS only if a
  panel's interaction grows complex (e.g. the simulator).

---

## References

- `PRD-034-knowledge-retention-layer.md`
- `RFC-086-operator-data-access-and-tooling.md`
- `RFC-081-personal-knowledge-layer-reconciliation.md`
- `RFC-072-canonical-identity-layer.md`
- `RFC-077-graph-visualization-modes.md`
- `PLAN-knowledge-retention-v1-build.md`
