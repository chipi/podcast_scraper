# UXS — Knowledge Layer v1 Surfaces

- **Status**: Draft
- **Assign number**: next free in `docs/uxs/` (placeholder name pending)
- **Parent PRD**: `PRD-034-knowledge-retention-layer.md`
- **Substrate**: `RFC-081-personal-knowledge-layer-reconciliation.md`
- **Layer**: OSS (user UX)
- **Stack**: Vue 3 + Vite, Cytoscape.js (fcose), Chart.js, Tailwind. Host: Player (PRD-030)
  + Catalog (PRD-029).

> Scope: the **v1** surfaces only — The Map, Dossier, Salience library, What-changed,
> Resurface card, Save-time connect, L2-relevance badge. Fast-follow/later surfaces
> (auto-threads, frontier, active-recall, feeds-map ordering) are out of scope here.

---

## 0. Cross-cutting contracts (apply to every surface)

**C1 — R1 generic event rendering.** Any surface that renders a corpus change consumes a
generic `ReconcileEvent { anchor_id, node_ref, change_type, source, detected_at, seq }`.
**No component branches on `change_type` to decide *whether* to render.** Styling by type
is a lookup (`extension → neutral`, `contradiction → warning`, unknown → neutral default).
An unknown future type must still render. This is the load-bearing contract — it is what
makes turning the dials zero-UI-rework.

**C2 — Salience visual encoding (consistent everywhere).** Salience maps to one visual
scale reused across surfaces: **size** (graph nodes), **order** (lists), **heat/opacity**
(fading). Fading (< reinforce-recent) dims; below-θ items are visually demoted, never shown
as errors.

**C3 — Derive-on-read.** L2 views call `/l2/*`; never assume a stored copy. Show a light
skeleton while deriving; progressive reveal as the graph hydrates (don't block the whole
view).

**C4 — Degrade gracefully.** Absent enrichment = omit the field, never a broken/empty
panel (consistent with PRD-029 FR4.1). Cold-start = honest "first save about X" states, not
barren panels.

**C5 — Deposit is instant; enrichment follows.** A save confirms immediately (optimistic);
async Anchor resolution backfills targets. UI shows a subtle "resolving…" affordance on the
new Anchor, never a blocking spinner.

---

## 1. The Map — "Your mind on the corpus" *(v1, headline)*

**Purpose.** Externalise the working set as a navigable graph; the proof that listening
compounded into an asset.

**Entry & nav.** Top-level "Mind"/"Map" tab. Deep-linkable to a focused node
(`/map?focus=person:CIL:…`). Back-stack: Map ⇄ Dossier ⇄ Player.

**Anatomy.**
- Cytoscape canvas (fcose). Nodes = anchored L0 nodes + salience-weighted k-hop neighbours;
  edges = real KG relations. Node **size = effective salience** (C2); node **colour =
  entity type** (`person`/`org`/`topic`/insight/claim); cluster by region.
- Cross-show signal: a halo/ring count showing how many shows feed the node ("fed by 5
  shows" reads differently than 1).
- **Conflict/staleness markers (data-driven, C1):** if a node has open reconcile events,
  show a marker; tap → opens that node's what-changed context. (In v1 only `extension`
  exists; the marker channel must already be wired so contradiction lights up later with no
  change.)
- Controls: RFC-077 modes (edge-weight, confidence surface, radial focus); search-to-focus;
  density slider (θ preview — show more/less of the working set).

**States.** Loading (skeleton graph) · sparse/cold-start (< N nodes → prompt "save insights
while you listen; your map grows here") · dense (perf budget below) · single-focus (radial).

**Interactions.** Tap node → Dossier. Long-press/▸ → quick actions (reinforce, dismiss,
open in Player at first encounter). Pan/zoom; pin focus.

**Data binding.** `GET /l2/graph?focus?&min_salience?` → `{ nodes:[{ref,type,salience,
shows,events_open}], edges:[{a,b,rel,weight}] }`.

**Perf.** Cap initial render to top-salience N nodes (e.g. 150) + neighbours; lazy-expand on
focus. Cytoscape fcose with worker layout if available. Below-θ nodes excluded from default
render (they're dormant).

---

## 2. Dossier (Anchor trace) *(v1)*

**Purpose.** Everything *you* know about one node, grounded and current. The inverse of a
dead bookmark.

**Entry & nav.** From Map node tap, from any entity chip, from what-changed item. URL
`/node/:ref`.

**Anatomy.**
- Header: canonical name, type, **current salience** (with rising/steady/fading affordance),
  #shows, #anchors, first-heard / last-reinforced.
- **Your encounters timeline:** each moment the user anchored/hit this node, with show +
  timestamp + **jump-to-Player** at offset.
- **Anchored insights:** the GIL insight nodes behind this entity, each grounded (expand →
  transcript span). Faithfulness first (grounding contract).
- **Claims & changes:** claims at/near the node; any open `ReconcileEvent`s rendered via C1
  (v1: extensions — "new episode discussed this").
- Actions: reinforce, dismiss/forget, add note.

**States.** Rich (full) · degraded (no GIL → encounters + jump only, no insight panel) ·
resolving (Anchor targets still backfilling → "resolving connections…").

**Data binding.** `GET /l2/node/:ref` → `{ canonical, salience, shows, anchors:[{episode,
t_offset, insight_ref?}], insights:[{ref, grounded_span}], claims:[…], events:[ReconcileEvent] }`.

---

## 3. Salience-ranked library *(v1)*

**Purpose.** The metabolism made actionable; the un-graveyard. The honest "what's top of
mind vs fading" list.

**Anatomy.** A single ranked list of retained nodes sorted by **current decayed salience**
(C2). Each row: name, type chip, salience heat bar, trend glyph (rising/steady/fading),
last-reinforced. Row actions: **reinforce** (one tap), **dismiss**, open Dossier.
Optional filter by type. Section divider where items cross below recent-reinforcement (the
"fading" zone) — but do **not** hide below-θ items here; this is the surface where dormancy
is *visible and reversible*.

**States.** Populated · cold-start (few rows → "your library fills as you anchor") · all-
fading (gentle nudge, never alarm).

**Data binding.** `GET /anchors?sort=salience&min_salience=0` (note: library shows below-θ
too, unlike the Map) → ranked anchors with effective salience computed server-side.

---

## 4. What-changed *(v1 on-demand; differentiated surface)*

**Purpose.** Keep the model true with zero maintenance effort — the differentiated capability, in UI form.

**Entry & nav.** "What changed" entry (badge with unseen count) on Map and home. Pull, not
push, in v1 (user opens it → `reconcile()` runs).

**Anatomy.**
- Grouped by **affected node**, then by change-type **section** rendered via C1
  (v1: *Extended*; *Contradicted*/*Shifted* sections exist in the component but are empty
  until Dial B — they must not be special-cased away).
- Each item: the affected Anchor/node, the change-type chip (styled by lookup, C1), a
  one-line description, and a **grounded back-link** to the triggering episode/claim in L0
  (jump-to-Player).
- Per-item **acknowledge** (marks handled). v1 advances the **per-user** watermark on view;
  per-item acknowledge is wired in the UI now (no-op beyond reinforcement) so the per-node
  watermark migration is zero-UI-change later.
- Acknowledging reinforces the affected anchor's salience.

**States.** Has-changes · none ("nothing new since you last looked") · **hole-fallback**
(watermark behind changelog compaction → "we refreshed your model; some interim changes may
not be itemised" — see RFC-081).

**Data binding.** `GET /reconcile?types=extension` → `{ events:[ReconcileEvent], watermark,
fallback?:bool }`. Component renders `events` grouped; never filters by type to decide
existence.

---

## 5. Resurface card — "from your knowledge" *(v1)*

**Purpose.** Reactivate a prior anchor at the moment of natural relevance.

**Entry & nav.** Inline, opt-in, surfaced in two triggers: (a) **entity-triggered** — a
newly processed L0 node sits near an existing Anchor (same join as reconcile); (b)
**spaced** — schedule fallback. Appears in the Player post-episode and/or home feed.

**Anatomy.** Old anchored insight (grounded) + the new connection that triggered it ("3
weeks ago you anchored this about X; today's episode extends it") + actions:
**acknowledge** (→ reinforce salience, C2) / **dismiss** (→ let it fade) / open Dossier.

**States.** Triggered · none · dismissed (suppress that anchor's resurface for a cooldown).

**Data binding.** Reuses the reconcile join (`extension` near anchors) + a spaced selector
over `/anchors`. Acknowledge → `POST /anchors/:id/reinforce`.

---

## 6. Save-time connect — "this connects to…" *(v1)*

**Purpose.** Elaborative encoding at capture — make the save worth more and densify the
graph.

**Entry & nav.** Fires inline at **deposit** (after a save in the Player), non-blocking.

**Anatomy.** A compact strip: up to *N* (e.g. 3) existing Anchors that share entities with,
or are vector-near, the new save — each with a **reason** ("both about NATO expansion").
Tap → Dossier/jump. Optional "same thread?" affordance (seeds a future auto-thread; v1
records the hint only). Cold-start: if nothing connects, show "first save about X" (C4),
not an empty box.

**Data binding (the one v1 backend dependency outside this package).**
`GET /api/connect?anchor_id=:id&limit=3` → `{ connections:[{ anchor_id, node_ref, reason,
score }] }`. Backend = shared-entity overlap (KG) ∪ vector-near (LanceDB) over the user's
own Anchor set; reason derived from the strongest shared entity. **Formalise this endpoint
into RFC-078/079; spec'd here for v1 build.**

---

## 7. L2-relevance badge on catalog cards *(v1)*

**Purpose.** Recommendation as a projection of L2 — surfaced where the user chooses what to
play.

**Entry & nav.** On episode cards in Catalog (PRD-029) and any episode list.

**Anatomy.** A small badge: count of tracked (above-θ) nodes the unconsumed episode touches,
and — when Dial B lands — whether any touch conflicts with something anchored ("touches 4
you track · 1 conflicts"). v1: extension/touch count only; the conflict slot is wired (C1)
but inert until contradiction exists. Tap → "why" popover naming which of the user's nodes
it hits.

**Data binding.** `GET /advise/episodes` returns per-episode `{ l2_touch_count,
conflict_count, hit_nodes:[ref] }`. v1 ordering stays chronological (PRD-029); the
**ranking** ("feeds your map" sort) is deferred to RFC-083 — the badge ships, the re-sort
does not.

---

## 8. Navigation model (summary)

```
Player ──save──▶ (deposit, optimistic) ──▶ Save-time connect strip
  │
  ├─▶ Catalog card ──L2 badge tap──▶ "why" popover ──▶ Dossier
  │
Map (working set) ⇄ Dossier ⇄ Player(@offset)
  │
What-changed (badge) ──item──▶ grounded back-link ──▶ Player(@offset)
Resurface card (inline) ──ack──▶ reinforce ;  ──open──▶ Dossier
Salience library ──row──▶ Dossier ;  ──reinforce/dismiss──▶ (in place)
```

Back navigation always available; Map↔Dossier↔Player preserve focus/scroll.

---

## 9. Accessibility & empty-state notes

- Salience encoding must not be colour-only (size + label + glyph) for the Map and library.
- Every density-gated surface has a defined cold-start copy (C4); never render a blank
  panel.
- Conflict/contradiction styling (later) must pair colour with an icon + text label.
