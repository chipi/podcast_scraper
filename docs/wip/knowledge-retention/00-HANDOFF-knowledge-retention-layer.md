# Knowledge Retention Layer — Handoff Package (Index)

- **Status**: Draft handoff for implementation planning
- **Audience**: a code-adjacent agent (Claude Code / Cursor) taking v1 into planning + build
- **Owner**: Marko

> **Numbering caveat (read first).** `PRD-034`, `RFC-081`, and all reserved downstream
> numbers (`RFC-082`+) are **proposed placeholders** — they were not verified against the
> live `docs/prd/`, `docs/rfc/`, `docs/uxs/` trees. Confirm the next free numbers and
> renumber before merge. The UXS and PLAN docs are name-based pending number assignment.

---

## 0. What this package is

The **Knowledge Retention Layer** turns the unit of value from the *episode* to a
**persistent personal knowledge state (L2)** that episodes write into. This package is the
complete v1 spec set: the model, the architecture, the v1 UX, and a build plan. v1 is an
**internal value-chain dogfood**, not a public release — its job is to close the loop
end-to-end on the cheapest reconciliation cell and harden the contracts every later tier
reuses.

---

## 1. Reading order

0. **VISION — Intelligence-Powered Information Consumption** — the north star: what we want,
   why, and the non-negotiables. Read first; settle this before building.
1. **This doc** — orientation, vocabulary, decided-vs-open, scope, conventions.
2. **PRD-034** §Model + §The five operations — the *why* and the product model.
3. **RFC-081** — the substrate: L1/L2/Anchor data model, the reconciliation inversion, the
   two-dial `Reconciler` abstraction, watermark + decay detailed design (with diagrams).
4. **PRD-034** §UI Features + §How each surface creates value + §Spec map — the surfaces
   and which downstream docs they spawn.
5. **UXS — Knowledge Layer v1 surfaces** — build-level UX for the v1 surfaces.
6. **PLAN — Knowledge Retention v1 build** — milestones, acceptance criteria, test
   strategy, risks. Start execution here.
7. **PRD-035 — Operator Control Room** + **RFC-086 — Operator data-access & tooling** —
   the viewer/admin control room over the shared machine (internal/Pro; built *alongside* the
   P0 v1 milestones, not after).
8. **PRD-036 — Simulation & Validation Harness** + **RFC-087 — Simulation harness
   architecture** — how value is validated **without real users** (first-class, permanent
   infra); read before trusting any value claim or `λ`/`θ` setting.
9. **PRD-037 — Corpus Scout** + **RFC-088 — Corpus Scout architecture** — the standalone tool
   that *operationalises* the corpus-composition precondition: diagnose → brief → prospect RSS
   feeds → advise what to ingest. Reuses the RFC-087 diagnostic.

---

## 2. Document inventory

| Doc | Type | Status | Answers |
|---|---|---|---|
| `00-VISION…` | Vision | Draft | the north star: what we want, why, non-negotiables |
| `00-HANDOFF…` (this) | Index | Draft | orientation, scope, vocabulary, conventions |
| `PRD-034-knowledge-retention-layer` | PRD | Draft | what/why; model; operations; surfaces; value; spec map |
| `RFC-081-personal-knowledge-layer-reconciliation` | RFC | Draft | substrate architecture; Anchor schema; reconciliation; watermark + decay |
| `UXS-knowledge-layer-v1-surfaces` | UXS | Draft | v1 surface anatomy, states, interactions, data binding |
| `PLAN-knowledge-retention-v1-build` | Plan | Draft | milestones, sequencing, acceptance, tests, risks |
| `PRD-035-operator-control-room` | PRD | Draft | operator/admin viewer surfaces over the shared machine (internal/Pro) |
| `RFC-086-operator-data-access-and-tooling` | RFC | Draft | operator data-access layer, simulation, sandbox, CIL impact, access control |
| `PRD-036-simulation-validation-harness` | PRD | Draft | validate value without real users; personas, metrics, validation ladder (internal) |
| `RFC-087-simulation-validation-harness` | RFC | Draft | virtual-clock harness, persona engine, synthetic corpus, ground-truth metrics |
| `PRD-037-corpus-scout` | PRD | Draft | standalone corpus-curation/feed-prospecting tool; diagnose → brief → advise (internal/Pro) |
| `RFC-088-corpus-scout` | RFC | Draft | gap→brief→prospect pipeline, cheap-proxy scoring, content-measured divergence, calibration loop |

---

## 3. Canonical vocabulary (single source — do not redefine elsewhere)

- **L0 — Corpus.** KG entities/claims/edges, GIL insight nodes, episodes. Shared, evolving.
- **L1 — Salience overlay.** Per-user set of **Anchors**. Typed pointers + provenance +
  salience. **No L0 content is copied here.**
- **L2 — Personal subgraph.** The induced, salience-weighted neighbourhood of L0 selected
  by L1. **Derived on read; never stored as ground truth.** Reconstructable from
  `L1 ⋈ current L0`.
- **Anchor.** The save unit: `{ targets: typed refs into L0, provenance, salience,
  explicitness, note?, captured_at }`. One-tap to create; resolved async by walking GIL/KG
  provenance.
- **Salience.** A per-anchor weight that **decays** (`λ`, half-life) and is **reinforced**
  on re-encounter. `θ` is the inclusion threshold for the *active* set.
- **Working set.** `{ nodes with effective_salience > θ }` — the live slice of L1 that
  induces L2. *This is the user's felt knowledge state.*
- **Changelog.** Shared, append-only, user-agnostic log of node-affecting corpus writes:
  `{ seq, node_ref, change_type, source, detected_at }`. Computed once per node.
- **Watermark.** Per-user cursor into changelog `seq`; records how far the user has been
  shown. v1 = per-user; per-node later.
- **The five operations.** Deposit · Connect · Resurface · Reconcile · Advise.
- **The two dials.** *Trigger tier* (on-demand → batch → live) × *change-type richness*
  (extension → contradiction → evolution → corroboration). v1 = on-demand × extension.
- **`ReconcileEvent`.** Generic output of `reconcile()`; carries `change_type` as data.
  **Consumers must not branch on change-type or tier** (contract **R1**).

---

## 4. Scope boundary

**In v1 (build this):**
- L1 Anchor store + async resolution; salience **decay live from day one**.
- L2 derived-view endpoints (`/l2/graph`, `/l2/node`) — derive-on-read.
- Corpus **changelog (extension entries only)**.
- `OnDemandReconciler` + `extension`; **per-user** watermark.
- Surfaces: **The Map, dossier, salience library, what-changed, resurface card, save-time
  connect, L2-relevance badge** (per the UXS).
- Consumer contract **R1** enforced across all surfaces.
- Instrumentation for time-to-first-aha + acknowledgement rates.

**Explicitly NOT v1 (do not build; reserved/deferred):**
- Batch digest / live event-triggered reconciliation (Dial A upper tiers).
- contradiction / evolution / corroboration change-types (Dial B beyond extension).
- Per-node watermark, reverse index, RFC-065 fan-out.
- Auto-threads, frontier prompts, active-recall, "feeds-your-map" ordering.
- Inferred/ambient total capture beyond the limited high-confidence seed.
- Any content storage in L1; any stored/materialised L2 as ground truth.
- Social / shared subgraphs.

---

## 5. Decided vs open

**Locked (do not re-litigate in v1):**
- Unit of value = persistent L2, not the episode.
- L1 stores **typed pointers + salience only**, never content.
- L2 is **derived on read**, never authoritative-stored.
- Salience **decays**; L1 is not append-only.
- v1 = **on-demand × extension**; **per-user** watermark.
- **R1** generic-event rendering contract.
- Reconciliation **inversion** (shared changelog + lazy per-user join), not push fan-out.
- **Corpus value is gated on shape, not size:** high **overlap** (entity recurrence ≥2–3
  per tracked entity) **and** high **divergence** (stance/temporal/source/claim dispersion).
  Avoid the high-overlap/low-divergence **echo chamber** — it starves the differentiated value. Verify via the
  RFC-087 value-potential diagnostic before building. (PRD-034 §Corpus composition.)

**Open (needs a decision during/after v1 — see RFC-081 §Open questions):**
1. Initial `λ` / half-life and `θ` values (seed conservative-long; fit from data).
2. Metric that triggers per-node watermark migration.
3. Inferred-deposit policy (defaulted **off** in v1).
4. Changelog retention window (controls hole-fallback frequency).
5. Whether to cache derived L2 in v1 (correctness vs `gi-kg-viewer` render cost).

---

## 6. Downstream specs register (reserved, **do not implement in v1**)

Per the just-in-time rule: a surface earns its own doc only when it adds interaction
complexity (UXS) or new backend algorithm (RFC). These are **reserved and scoped**, not
written:

| Reserved | Type | Scope | When |
|---|---|---|---|
| `RFC-082` | RFC | Live / event-triggered reconcile: reverse index `node_ref → users` + RFC-065 fan-out | when live tier is warranted |
| `RFC-083` | RFC | L2 advise ranking: salience-weighted touch score + frontier topology | fast-follow |
| `RFC-084` | RFC | Auto-threads: clustering over the anchor set | fast-follow |
| `RFC-085` | RFC | Active-recall: grounded question-gen + double opt-in | later, gated |
| `RFC-078/079 +` | addendum | `/connect` neighborhood endpoint (save-time connect) | **v1 — small addition, not a new RFC** |

(`/connect` is the one v1 backend dependency that lives outside this package — its contract
is given in the UXS for now; formalise it into the hybrid-search RFC.)

---

## 7. Conventions for the receiving agent

- **Docs culture:** RFC for architecture, PRD for product, UXS for UX, ADR for contained
  decisions; update `DECISIONS.md` / `LEARNINGS.md` / `CLAUDE.md` as you go.
- **OSS/Pro boundary:** these are OSS-layer docs — **no commercial info** (tiers, pricing,
  competitive positioning). Keep it out of code comments and docs too.
- **Grounding contract:** Anchor → entity/claim resolution walks GIL provenance;
  **faithfulness of resolution is the critical quality dimension** over cleverness.
- **Enrichers:** always non-blocking, always `derived: true`. The async Anchor-resolution
  job and changelog emission follow this rule — never block deposit or the core pipeline.
- **Single source of truth:** schema lives in RFC-081; vocabulary lives here (§3). Do not
  duplicate — reference.
- **Output:** default to repo `docs/**` paths on a branch; PR with the verified numbers.
- **Stack (for the UX/build agent):** FastAPI (Python) backend; Vue 3 + Vite + Cytoscape.js
  + Chart.js + Tailwind frontend; LanceDB for neighborhood/vector; DuckDB/JSONL where it
  already fits.

---

## 8. Definition of done (v1)

v1 is done when all of these hold (full criteria in PRD-034 §Success + PLAN acceptance):

1. A user with ≥ threshold Anchors can open **The Map** and recognise their retained model.
2. **L1 stores zero L0 content**; wiping any derived L2 view and recomputing yields the
   identical view (invariant test).
3. **Reconcile** returns correct `extension` events for anchored nodes with correct
   provenance, at per-user cost bounded by the **active** anchor count.
4. **Salience decay** curates: stale anchors fade out of L2 views without user pruning;
   reinforced anchors persist.
5. Every surface renders `ReconcileEvent` **tier-/type-agnostically** (R1) — an injected
   unknown future change-type still renders.
6. Instrumentation emits time-to-first-aha and acknowledgement rates for later `λ`/`θ` fit.
7. **Value plausibility is validated via the simulation harness** (RFC-087) before real users:
   computed connect/reconcile precision, a cold-start curve, bounded working-set, and a
   flywheel A/B verdict — and the corpus value-potential diagnostic (recurrence + dispersion) has been run over the real beta corpus.
