# PLAN — Knowledge Retention v1 Build

- **Status**: Draft (execution plan)
- **Layer**: OSS (v1 build)
- **Parent**: `PRD-034`, `RFC-081`, `UXS-knowledge-layer-v1-surfaces`
- **Goal**: close the value chain end-to-end on the cheapest cell (**on-demand ×
  extension**), internal dogfood, with all v1 contracts hardened so the dials turn later
  with no core rewrite.

> This is a planning artifact: milestones are dependency-ordered with acceptance criteria
> and test hooks. The receiving agent can decompose each milestone into issues/tasks. Each
> milestone names the FR/RFC section it satisfies.

---

## 0. Sequencing logic

Backend substrate before surfaces; the **R1 generic-event contract** and **salience decay**
land early because everything depends on them. Reconcile is cheap in v1 (extension =
graph-delta), so it can land before the read surfaces and be dogfooded via API. Critical
path: **M0 → M1 → M2 → M3** (substrate) then **M4 → M5/M6/M7** (surfaces, parallelisable).

```
M0 schema ─▶ M1 deposit ─▶ M2 L2 derive + decay ─▶ M3 changelog + reconcile core
                                   │                        │
                                   └──────────┬─────────────┘
                                              ▼
                          M4 frontend substrate (R1, salience encoding)
                                              ▼
                  ┌───────────────┬───────────────┬───────────────┐
                M5 read         M6 maintain      M7 connect+advise
              (Map/Dossier/lib) (what-changed/   (save-connect,
                                 resurface)        L2 badge)
                                              ▼
                                  M8 instrumentation (cross-cutting)
```

---

## M0 — Schema & migrations
*Satisfies: RFC-081 §Data model.*

- L1 `anchors` table: `id, user_id, targets[] (typed refs), provenance{episode,t_offset,
  surface}, salience{base,last_reinforced_at,reinforce_count}, explicitness, note?,
  captured_at`. Secondary indexes: `targets[].ref`, computed `effective_salience`.
- `changelog` table: `seq (monotonic), node_ref, change_type, source{episode,claim},
  detected_at`. Append-only. Index on `(node_ref, seq)`.
- `reconcile_watermark`: `user_id, last_reconciled_seq`. (Per-user in v1; design the column
  so a per-node table can be added without migration of this one.)

**Acceptance.** Tables created; `effective_salience` computable server-side; changelog
append is O(1) and monotonic. **Tests:** migration up/down; monotonic seq under concurrent
append.

---

## M1 — Deposit path
*Satisfies: PRD FR1.1–1.5; UXS C5.*

- `POST /anchors` — optimistic create from one tap; returns immediately.
- Async **resolution job** (non-blocking, `derived:true`): walks GIL/KG provenance to
  populate `targets` beyond the directly-saved node. Follows the enricher rule (never blocks
  deposit).
- `DELETE /anchors/:id`; `POST /anchors/:id/reinforce`.
- Episode-favorite → episode-scoped Anchor; optional high-confidence inferred sub-anchors
  from salient GIL nodes, flagged `explicitness=inferred`.

**Acceptance.** Save confirms < 200ms before resolution; resolved targets are canonical
(RFC-072). Reinforce updates salience. **Tests:** deposit→resolve backfills targets;
reinforce raises effective salience and resets clock; inferred vs explicit flagged.

---

## M2 — L2 derivation + salience decay
*Satisfies: RFC-081 §L2 derivation, §Decay; PRD FR (Map/library data).*

- Salience decay function `effective_salience = base·exp(-λ·Δt)·f(reinforce_count)`.
  **Conservative long half-life seed** (param-config, not hardcoded). `θ` config too.
- **Injectable clock:** decay (and spaced-resurface scheduling, and changelog/anchor
  timestamps) read time from an injected clock, **never `datetime.now()`** — load-bearing for
  the simulation harness (RFC-087); without it the trajectory metrics are untestable.
- `GET /l2/graph?focus?&min_salience?` — induced, salience-weighted subgraph over **current
  L0**, derive-on-read. `GET /l2/node/:ref` — dossier payload.
- Active-set = nodes with aggregate effective_salience > θ; below-θ dormant (excluded from
  Map, included in library).

**Acceptance.** L2 is reconstructable: wipe any cache → recompute identical (**invariant
test**). Active-set membership tracks decay. **Clock is injectable** (fixed virtual time →
deterministic decay). **Tests:** invariant recompute; decay
monotonic without reinforcement; θ boundary inclusion/exclusion; node aggregates multiple
anchors correctly; decay deterministic under an injected clock.

---

## M3 — Changelog emission + reconcile core
*Satisfies: RFC-081 §Reconciliation, §Reconciler protocol; PRD FR4.1–4.2, FR4.5.*

- **Changelog emission (extension only):** a graph-delta hook on enrichment — when a new
  GIL insight / KG edge references a node, append `change_type=extension`. Shared,
  user-agnostic, once per node. (No conflict ML in v1.)
- **`reconcile()` pure core** = the lazy join: `changelog WHERE seq>watermark ∧
  node_ref∈active_anchors ∧ type∈requested`; emit `ReconcileEvent[]`; advance watermark.
- `Reconciler` protocol with **`OnDemandReconciler`** as the only v1 driver.
- `GET /reconcile?types=extension` → `{ events, watermark, fallback? }`. Hole-fallback when
  `watermark < changelog.min_seq` → signal full-recompute.
- **Virtual-time aware:** changelog `detected_at` and reconcile run at the injected clock;
  `reconcile()` and changelog are callable at **sandbox scope** (RFC-086) so the harness
  (RFC-087) drives them without touching real data.

**Acceptance.** Per-user reconcile cost scales with **active** anchor count, not corpus
size. Events carry correct provenance back-links. **Tests:** join selects only
anchored+newer; watermark advances; hole-fallback triggers correctly; cost test (fixed
anchors, growing corpus → flat per-user work); **R1 backend:** an injected unknown
`change_type` flows through unfiltered (incl. an injected synthetic `contradiction`).

---

## M4 — Frontend substrate
*Satisfies: UXS C1–C5.*

- Generic `ReconcileEvent` renderer + change-type **style lookup** (unknown → neutral).
  **No component branches on type to render.**
- Salience visual encoding primitives (size/order/heat) reused across surfaces.
- Derive-on-read data layer with skeleton/progressive reveal; optimistic deposit affordance.

**Acceptance.** A storybook/fixture with an unknown change-type renders without code change.
**Tests:** component test injecting `change_type="__future__"` → renders via default style.

---

## M5 — Read surfaces (Map · Dossier · Library)
*Satisfies: PRD Map/Dossier/Library; UXS §1–3.*

- **Map:** Cytoscape fcose; salience-sized nodes; type colour; cross-show halo; conflict
  marker channel wired (inert in v1); RFC-077 modes; focus/search; perf cap + lazy expand.
- **Dossier:** header + encounters timeline (jump-to-Player) + grounded insights + claims/
  events (C1); reinforce/dismiss/note.
- **Library:** salience-ranked list incl. below-θ; reinforce/dismiss in place.

**Acceptance.** Map renders a recognisable working set at target density within perf budget;
Dossier grounds every insight to a transcript span; library reflects decay ordering.
**Tests:** Map perf at N nodes; Dossier degraded-state (no GIL); library ordering matches
server salience.

---

## M6 — Maintain surfaces (What-changed · Resurface)
*Satisfies: PRD FR4.x, FR3.1–3.2; UXS §4–5.*

- **What-changed:** grouped-by-node, type sections via C1 (Extended populated; Contradicted/
  Shifted present-but-empty), grounded back-links, per-item acknowledge (wired), hole-
  fallback state.
- **Resurface card:** entity-triggered (reconcile join) + spaced fallback; acknowledge →
  reinforce; dismiss → cooldown.

**Acceptance.** Opening what-changed runs reconcile and shows correct extensions with
back-links; acknowledging reinforces. Resurface fires on a near-anchor corpus write.
**Tests:** end-to-end deposit→corpus-write→reconcile surfaces the event; acknowledge
reinforces; empty Contradicted section does not special-case.

---

## M7 — Connect + advise (v1 slice)
*Satisfies: PRD FR2.x, FR5.1; UXS §6–7.*

- **Save-time connect:** `GET /api/connect` (shared-entity ∪ vector-near over user's
  anchors); inline strip with reasons; cold-start state. *(Formalise endpoint into
  RFC-078/079.)*
- **L2-relevance badge:** `GET /advise/episodes` per-episode touch count + (inert) conflict
  slot; "why" popover. **No re-sort in v1** (ordering stays chronological; ranking = RFC-083
  later).

**Acceptance.** Connect returns relevant anchors with reasons at deposit, non-blocking;
badge shows accurate touch counts. **Tests:** connect precision on a seeded set; badge count
matches L2 touch; cold-start renders honest empty state.

---

## M8 — Instrumentation (cross-cutting)
*Satisfies: PRD §Cold-start; RFC-081 §Decay (data to fit λ/θ).*

- Metrics: **time-to-first-aha**, anchors-per-session, resurface/what-changed **acknowledge
  vs dismiss rates** (the data to later fit `λ`/`θ` and the per-node-watermark trigger),
  Map open + density, deposit→resolve latency.
- Wire via existing Langfuse (LLM traces on resolution/question-gen later) + PostHog
  (product events). Keep production-derived data private (GDPR + competitive sensitivity).

**Acceptance.** Dashboards emit the above; acknowledgement data is queryable for parameter
fitting. **Tests:** event emission on each tracked interaction.

---

## Test strategy (summary)

- **Invariant:** wipe derived L2 → recompute identical (M2). The defining correctness test.
- **Reconcile correctness + cost:** right events, right provenance, per-user cost flat under
  corpus growth (M3).
- **Decay:** monotonic decay, reinforcement reset, θ boundary (M2).
- **R1 contract:** unknown change-type renders end-to-end (M3 backend, M4 frontend).
- **Grounding:** every surfaced insight resolves to a transcript span (M5/M6).
- **Degrade/cold-start:** absent enrichment omits fields; density-gated surfaces show honest
  empty states (M5–M7).

---

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| **Cold-start** — grow surfaces thin until density | "so what?" first impression | Lead value on Mode-2 + Dossier; generous high-confidence seeding; honest empty states; track time-to-first-aha (M8) |
| **`λ`/`θ` miscalibration** | deletes live interests or rebuilds graveyard | Conservative long half-life seed; param-config; fit from M8 acknowledgement data, not guess |
| **Changelog cost / growth** | storage + hole-fallbacks | Extension = cheap graph-delta; retention window tuned to active reconcile gap (open Q) |
| **Cytoscape perf on dense L2** | Map sluggish | Top-N salience cap + lazy expand; worker layout; below-θ excluded |
| **Async resolution lag** | Anchor targets briefly thin | Optimistic deposit (C5); "resolving…" affordance; never block |
| **R1 violated by a shortcut** | dials cost UI rework later | Enforce via the unknown-type render test in CI (M4) |
| **Grounding/faithfulness drift** | trust loss (critical dimension) | Every insight links to span; faithfulness checks over cleverness |

---

## Definition of done

Maps to handoff §8 items 1–6 / PRD §Success: recognisable Map; zero-content L1 + invariant
recompute; correct bounded-cost reconcile; decay curates; R1 holds for an unknown type;
instrumentation feeds λ/θ fitting. (Handoff §8 item 7 — value validation without real users —
is a cross-track deliverable of PRD-036 / RFC-087, not this PLAN.) When these pass on the
internal dogfood, **turn the dials** (Dial B contradiction via enricher registry; Dial A
batch driver) with no core rewrite.
