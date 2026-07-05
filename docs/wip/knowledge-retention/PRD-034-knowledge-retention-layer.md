# PRD-034: Knowledge Retention Layer

- **Status**: Draft
- **Authors**: Marko
- **Layer**: OSS (user product)
- **Target Release**: v3.0 (layer + v1 surfaces), tiers continue past v3.0
- **Parent PRD**: `docs/prd/PRD-026-podcast-platform.md`
- **Related RFCs**:
  - `docs/rfc/RFC-081-personal-knowledge-layer-reconciliation.md` — architecture, Anchor schema, reconciliation tiers (companion to this PRD)
  - `docs/rfc/RFC-072-canonical-identity-layer.md` — shared `person:`/`org:`/`topic:` vocabulary anchors point into
  - `docs/rfc/RFC-065-agent-observable-instrumentation.md` — corpus-write event spine reconciliation depends on
  - `docs/rfc/RFC-073-enrichment-pipeline.md` — enrichers that emit corpus-change signal
- **Related PRDs**:
  - `docs/prd/PRD-030-platform-player.md` — primary host surface; deposit happens during/after playback
  - `docs/prd/PRD-029-platform-catalog.md` — advise/recommend surfaces ride on catalog cards

> **Proposed numbering** — `PRD-034` / `RFC-081` are placeholders; confirm against the
> repo doc tree and renumber before merge. Numbers were not verified against
> `docs/prd/` and `docs/rfc/` at draft time.

---

## Summary

Today, listening is a flow that streams past the user and is gone. Save buttons do not
change that — they put a few items in a side-pocket while the flow stays fundamentally
ephemeral. After 10, 50, 100 episodes, almost nothing is retained, and the saved-items
list becomes a graveyard.

The **Knowledge Retention Layer** changes the **unit of value** from the *episode* to a
**persistent, per-user knowledge state** that episodes *write into*. Consumption stops
being a flow you watch go by and becomes a series of deposits into something that
compounds. The state is not a pile of saved text; it is a thin overlay of typed,
weighted *pointers* into the platform's canonical knowledge graph (KG) and Grounded
Insight Layer (GIL), kept alive and current by the same corpus-level intelligence the
platform already runs.

This single reframe collapses two goals that otherwise compete on the roadmap —
**remembering** what mattered and **tracking** how it evolves — into two operations on
one substrate. Recall is *reading* the state; tracking is *keeping it current*.

---

## Background

We have shipped **save-favorite-episode** and **save-insight**. They prove intent but
expose the real problem: **capture is the easy half and is almost never the bottleneck.**
People save constantly and never return; the forgetting curve does its work.

The hard, valuable half is **reactivation and maintenance**. And for a media-intelligence
product specifically, the thing worth retaining is not trivia — it is an accurate,
evolving model of how narratives, actors, and positions sit and shift. That is less a
*memory* problem than a *staleness* problem. A flashcard tool helps you hold a fixed
fact; this layer should help someone keep a mental *model* current.

A save here is therefore **a signal of salience, not a stored copy of text**. The user
supplies the pointer ("this mattered to me, here"); the KG and GIL supply and maintain
the structure behind it. That division of labour is the whole design.

---

## The Model

This section is intentionally narrative so the model reads cold for anyone picking up the
PRD. (Commercial framing — tiering, positioning, competitive comparison — is deliberately
excluded per our OSS-PRD rule and lives in separate commercial docs.)

### The unit of value moves from episode to personal knowledge state

Every consumption product, including ones with save buttons, treats the **episode** as
the unit. It streams, you consume it, the save decorates an ephemeral flow. This layer
makes the unit a **persistent knowledge state** that episodes deposit into. Each episode
becomes an *input that makes the state more valuable*, rather than a thing that is itself
the value and then is gone.

### Three ways knowledge dies — the layer must beat all three

A model that creates lasting value has to defeat each distinct failure mode:

1. **Decay** — you forget it. Fought by **resurfacing** (spaced + entity-triggered
   recall). This is the only death that generic "second brain" tools fight, and it is the
   the baseline capability generic tools already provide.
2. **Staleness** — the world moved; what you retained is now wrong or incomplete. Fought
   by **reconciliation**: when the corpus contradicts or extends something you retained,
   your state is flagged and updated. This is the differentiated axis and is only possible
   because retained items are *typed pointers into a live graph*, not strings.
3. **Isolation** — it never integrated; there was no structure to hang it on, so it
   cannot be retrieved or built upon. Fought by **connection**: each deposit links into
   the personal subgraph at capture time.

Most "second brain" products are ~90% death #1. Our differentiation is that we can
credibly attack #2 and #3 because the corpus-level intelligence already exists — we are
scoping it to one person's retained slice.

### The layers

- **L0 — Corpus** *(exists)*: episodes, KG entities/claims/edges, GIL insight nodes.
  Canonical, shared across all users, always evolving.
- **L1 — Salience overlay** *(new; per-user; thin)*: a set of typed **Anchors**. Each
  Anchor is `{ target: typed ref into L0, provenance: where the user hit it, salience:
  weight/why, captured_at }`. **It stores no content** — only "I, here, cared about *this
  node*."
- **L2 — Personal subgraph** *(derived; never stored as ground truth)*: the induced,
  salience-weighted neighbourhood of L0 selected by L1. This is the user's knowledge
  state, and because it is a **live view recomputed against current L0**, it is **never
  stale by construction**. The only thing that goes stale is the user's *awareness* of
  it — which is exactly what resurface and reconcile repair.

Because L1 holds pointers rather than copies, "keep it alive and expand it" is not a sync
problem we have to solve — it is free: a pointer always resolves to the current state of
the corpus.

### The Anchor (the save unit)

A save creates an **Anchor**: one tap for the user (frictionless, intentionally a *weak*
explicit signal), enriched *afterward by the system*, which walks GIL/KG provenance to
discover which canonical entities and claims the anchored insight references. **The user
saves a signal; the substrate unpacks it into structure.** The user never types the
structure. Free-text notes are allowed, but only as optional annotation hanging off a
typed Anchor — never as the Anchor itself.

### The five operations — all projections of L0 through L1

| Operation | What it is | Death it fights |
|---|---|---|
| **Deposit** | Create an Anchor (intentional, or high-confidence inferred) | — (entry point) |
| **Connect** | At deposit, surface nearby Anchors / shared entities (vector + KG) | Isolation |
| **Resurface** | Walk L1 on schedule + when L0 writes near an Anchor | Decay |
| **Reconcile** | Flag when a new corpus claim contradicts/extends an anchored target | Staleness |
| **Advise / Listen** | Rank unconsumed episodes by how densely they touch high-salience L2 | — (compounding) |

### The flywheel (this is the actual new model — the loop, not the features)

```
deposit → subgraph densifies → recommendation steers next listen toward the subgraph
   → that listen produces anchors that connect into existing structure
   → reconciliation keeps it true → denser subgraph → sharper recommendations → …
```

"Remember vs track" dissolves here: recall is reading L2, tracking is reconciling L2,
recommendation is L2 steering intake.

### The central tension to hold

**"Don't let knowledge go" pulls toward capturing everything. "Lasting value" pulls
toward only the signal lasting.** If deposit is automatic and total, we recreate the
graveyard at the system level — a personal state so noisy that reconciliation fires
constantly and resurfacing surfaces junk. **The state's value and its smallness are the
same property.** Two design consequences, both load-bearing:

- Deposit is an intentional act (or a high-confidence inferred one), never ambient total
  logging.
- **Salience itself decays.** L1 is not append-only; an Anchor's weight fades over time
  and is reinforced by re-encounter. The personal state has a *metabolism*, which
  auto-curates without ever asking the user to prune. (Decay function: see RFC-081.)

---

## Goals

1. Make the **personal knowledge state (L2)** the primary durable artifact of using the
   platform — readable, current, and growing across episodes.
2. Capture salience as **typed Anchors into L0**, not stored text, so the state can be
   kept live, connected, and reconciled.
3. Deliver the **five operations** as user-facing capability: deposit, connect,
   resurface, reconcile, advise.
4. Keep reconciliation **simple and scalable to a massive user base** via a tiered,
   abstracted design (RFC-081), shipping the cheapest tier first.
5. Ship **L2-native UI surfaces** that create value the moment density exists — not
   generic bookmark UI.
6. Degrade gracefully: every surface works (in reduced form) when enrichment/conflict
   data is absent.

---

## Non-Goals (v1-scoped; phrased so as not to pre-close later options)

- **Not a content store.** L1 holds typed pointers + salience only; copies of transcript,
  insight, or summary text never live in the personal layer. *(This is the architecture,
  stated as a boundary.)*
- **Not ambient total capture in v1.** Deposit is intentional or high-confidence inferred.
  Implicit-anchor heuristics (full listen-through, replays) are explicitly *deferred*, not
  ruled out, and must respect the smallness property if introduced.
- **Not a generic / collaborative recommender.** Advice is a projection of the user's own
  L2, not behavioural / collaborative filtering.
- **Not free-text-first.** Notes are optional annotation on a typed Anchor.
- **Not social / shared subgraphs in v1.** Personal only, to keep the model coherent;
  shared/comparative subgraphs are a possible later direction.
- **Not real-time interrupt alerting in v1.** Reconciliation ships pull-shaped
  (on-demand / digest); live event-triggered alerting is a later tier (RFC-081).

---

## User Stories

**As a listener who has heard 80 episodes**, I want a single view of what I've actually
retained — the entities, claims, and insights I marked as mattering — so the listening
adds up to something instead of evaporating.

**As someone tracking a topic across shows**, I want to be told when something I retained
gets contradicted or extended by a later episode, so my understanding stays current
without me re-listening to everything.

**As someone about to pick what to play**, I want recommendations that feed the model I'm
building — episodes that touch what I track, especially where they conflict with or extend
it — rather than generic "popular" picks.

**At the moment I save an insight**, I want to see what it connects to in what I already
know, so the new thing lands on existing structure instead of floating alone.

**Weeks later**, I want the platform to bring back something I cared about at the right
moment — when a new episode touches it — so it's reactivated instead of forgotten.

---

## User-Facing Requirements

Organized by operation. "v1" marks the initial release scope; unmarked items are
fast-follow.

### Deposit (Anchor creation)

**FR1.1** *(v1)* — From the Player, a one-tap save on an insight, claim, or entity
creates an **Anchor** referencing the corresponding L0 node(s). The user is not asked to
classify or describe anything at save time.

**FR1.2** *(v1)* — Saving a whole episode as "favorite" creates an episode-scoped Anchor;
the system may, post-hoc, derive finer-grained Anchors from the episode's salient GIL
nodes (high-confidence inferred deposit), clearly distinguished from explicit ones.

**FR1.3** *(v1)* — The system resolves each Anchor to its canonical targets
(`person:`/`org:`/`topic:`/claim/insight) by walking GIL/KG provenance, asynchronously
and non-blocking. The user sees a save confirmed immediately; enrichment of the Anchor
follows.

**FR1.4** — An Anchor may carry an optional free-text note. The note is annotation; it
never replaces the typed target.

**FR1.5** — A user can remove an Anchor. (Decay handles silent de-prioritisation; removal
is the explicit override.)

**FR1.6** *(v1)* — **User data rights.** A user can **export** their full L1 (all Anchors, as
typed pointers + provenance + salience) and **erase** it (account-level delete). Because L1
holds pointers and L2 is derived, **erasure is clean by construction**: dropping a user's L1
rows removes all their personal state, and the derived L2 simply ceases to exist — there is no
copied content to hunt down. This is a direct benefit of the content-free design (GDPR
erasure/portability fall out of the architecture, not a bolted-on subsystem).

### Connect (elaborative encoding at capture)

**FR2.1** *(v1)* — On deposit, surface up to *N* existing Anchors that share entities with,
or are vector-near, the new Anchor: "this connects to things you already saved."

**FR2.2** — Surface the canonical entities/claims the new Anchor touches that are *already
in L2*, so the user sees the new item landing on existing structure.

### Resurface (fight decay)

**FR3.1** *(v1)* — A "from your knowledge" surface brings back Anchors on a schedule
(spaced) and, more strongly, when a newly processed L0 node sits near an existing Anchor
(entity-triggered): "3 weeks ago you anchored this about X; today's episode extends it."

**FR3.2** — Acknowledging a resurfaced Anchor reinforces its salience (resets/raises the
decay weight). Dismissing it lets it continue to fade.

**FR3.3** *(opt-in, gated)* — Active-recall prompts generated from an anchored insight
(question form). This is effortful and homework-flavoured; it must be behind explicit
opt-in and never default-on.

### Reconcile (fight staleness — the differentiated operation)

**FR4.1** *(v1, on-demand tier)* — A "what changed" view computes, on request, which of
the user's anchored targets have had corpus changes since they last looked: new content
referencing the node (**extension**), and — as conflict-detection enrichers come online —
**contradiction** and **position shift**.

**FR4.2** — Each reconciliation item links back to the triggering episode/claim in L0 and
to the Anchor it affects, with the change-type labelled.

**FR4.3** *(later tier)* — A periodic **digest** ("what changed in your model") delivered
on a cadence, without the user having to ask.

**FR4.4** *(later tier)* — Near-real-time reconciliation alerts (event-triggered). Gated
behind the event spine; see RFC-081 tiering.

**FR4.5** *(v1, load-bearing)* — **All surfaces consume reconcile output tier- and
type-agnostically.** Every view that renders a corpus change (the Map, "what changed", the
resurface card, the catalog L2-relevance badge) renders a generic `ReconcileEvent`
regardless of *which trigger tier produced it* (on-demand / batch / live) or *which
change-type it carries* (extension / contradiction / evolution / corroboration). No UI may
branch on `change_type == "extension"` or assume the on-demand path. Consequence: turning
up Dial A (live) or Dial B (contradiction) later is **zero UI rework** — new tiers and new
change-types simply begin appearing in surfaces that already render them. This contract is
the reason v1 closes the value chain before the dials are turned: v1 proves the rendering/
UX contract that every higher tier reuses.

> Reconciliation is abstracted behind a `Reconciler` strategy with two orthogonal dials —
> **trigger tier** (on-demand → batch digest → live) and **change-type richness**
> (extension → +contradiction → +evolution). v1 = on-demand + extension-only. See RFC-081.

### Advise / Listen (steer intake — compounding)

**FR5.1** *(v1)* — On catalog/episode cards (PRD-029), show an **L2-relevance badge**: how
many tracked nodes an unconsumed episode touches, and whether any touch is a contradiction
of something anchored ("touches 4 nodes you track · 1 conflicts").

**FR5.2** — A "feeds your map" ordering ranks unconsumed episodes by salience-weighted L2
touch density.

**FR5.3** — **Frontier prompts**: surface episodes that bridge two heavily-tracked-but-
unconnected regions of L2 ("you track X and Y but never the link between them; this covers
it").

---

## UI Features (the value-creating surfaces on top of L2)

These are the concrete surfaces. Each is L2-native — it only makes sense because L2
exists. Grouped by the operation it serves; v1 marked.

### Reading L2 (recall)

- **The Map — "Your mind on the corpus"** *(v1, headline artifact)*. A visual personal
  subgraph: anchored entities/claims/insights, sized by current (decayed) salience,
  clustered by region. Built on the existing `gi-kg-viewer` stack (Cytoscape). This is the
  "after 100 episodes, here's what you absorbed" view. Reuses RFC-077 graph modes (edge
  weight, confidence surface, radial focus).
- **Anchor trace / dossier** *(v1)*. Tap an anchored node → every episode and moment where
  the user hit it, the GIL insights behind it, the provenance chain. The inverse of a dead
  bookmark: a living dossier on one node, always resolved against current L0.
- **Salience-ranked library** *(v1)*. The user's retained nodes sorted by current decayed
  salience — what's hot in their mind vs fading. Doubles as the human-readable face of the
  decay metabolism.

### Keeping it fresh (resurface)

- **Resurfacing card — "from your knowledge"** *(v1)*. Inline, opt-in, spaced + entity-
  triggered (FR3.1). Acknowledgement reinforces salience.
- **Active-recall prompt** *(gated opt-in)*. Question generated from an anchored insight.

### Keeping it true (reconcile — differentiated)

- **"What changed" view / digest** *(v1 on-demand; digest later)*. Per anchored entity:
  extended / contradicted / shifted since last looked. The differentiated capability in
  user-facing form.
- **Staleness & conflict badges on the Map** *(fast-follow)*. A node you anchored now
  carries a contradiction flag — visual staleness directly on the subgraph.

### Connecting (connect)

- **Save-time "this connects to…"** *(v1)*. At deposit, nearby Anchors / shared entities
  (FR2.1). Elaborative encoding at the exact moment of capture.
- **Auto-threads** *(fast-follow)*. Emergent collections grouping Anchors by entity/theme
  across shows, surfaced without manual foldering ("you've been tracking X across 8
  shows").

### Steering intake (advise)

- **L2-relevance badge on catalog cards** *(v1)* (FR5.1).
- **"Feeds your map" ordering** *(fast-follow)* (FR5.2).
- **Frontier prompts** *(later)* (FR5.3).

---

## How each surface creates value

The surfaces above are not a flat feature list; each does one of **three jobs** on L2, and
value is created differently in each. This section makes the *why* explicit so the
inventory is not mistaken for a backlog of equals.

**Mode 1 — Read L2 (make retained knowledge legible).** A person's retained knowledge is
normally illegible — vibes and half-memories. These surfaces turn it into a tangible,
inspectable object, which is how "lasting value" becomes *felt* rather than asserted.
- *The Map* externalises the invisible working set as a navigable graph; its value is the
  visible **proof that consumption compounded into an asset** (the user watches it
  densify). This is the headline "that's my mind" moment.
- *Dossier* converts a dead bookmark into a **grounded, always-current dossier** on one
  entity — recall depth on demand, reconstructable to source.
- *Salience library* makes the **decay metabolism actionable** — a self-curating list, the
  structural antidote to the graveyard.

**Mode 2 — Keep L2 true (the differentiated value).** Retained knowledge mostly
dies by going **stale or wrong**, not by being forgotten — the death that static, content-copy
tools structurally cannot address. *What-changed* flags when the corpus extends, contradicts, or shifts something the
user anchored, keeping their model true **with zero maintenance effort**. This is only
possible because anchors are typed pointers into a live graph; it also carries the
objectivization mission at individual scale (it actively surfaces contradicting evidence
about things the user has internalised — an anti-filter-bubble). The product's "wow" should
be concentrated here.

**Mode 3 — Grow L2 (connect + advise → the flywheel).** Value here compounds — each action
makes the *next* listen worth more.
- *Save-time connect* is **elaborative encoding at capture, automated** — the highest-
  leverage learning intervention, forcing integration when attention is highest, so the
  save is worth more *and* the graph densifies.
- *Auto-threads* are **cross-show synthesis scoped to the user's own interests** — the
  platform's core differentiator turned personal; structure the user discovers rather than
  authors.
- *L2 badge / feeds-map* is **recommendation as a projection of L2, not collaborative
  filtering** — every listen steered to densify or *challenge* the model; this is the
  flywheel's drive.
- *Frontier prompts* are **discovery driven by subgraph topology** — two dense clusters
  with no edge between them is a literal knowledge gap; directed growth, not random
  exploration.
- *Resurfacing card* fights decay, but the differentiated version is **entity-triggered,
  not spaced** — reactivation at the moment of natural relevance, far stickier than a
  calendar ping, and its acknowledgement feeds reinforcement (tunes `λ`).

### Value map

| Surface | Value mechanism (one line) | Death fought | Flywheel role |
|---|---|---|---|
| The Map | Makes the retained working set visible as a compounding asset | — | proof / read |
| Dossier | Grounded recall-on-demand per entity, always current | decay | read |
| Salience library | Metabolism made actionable; the un-graveyard | decay | read / curate |
| What-changed | Keeps the model *true* with zero effort — **the differentiated capability** | **staleness** | maintain |
| Save-time connect | Elaborative encoding at capture; the save worth more | isolation | grow |
| Auto-threads | Personal cross-show synthesis, unauthored | isolation | grow |
| L2 badge / feeds-map | Recommendation as a projection of *your* model | — | **grow (flywheel)** |
| Frontier prompts | Directed growth via subgraph topology | isolation | grow |
| Resurfacing card | Entity-triggered reactivation at natural relevance | decay | read + grow |

### Cold-start: value is not uniform across a user's lifetime

Most *grow* surfaces (connect, auto-threads, badge, frontier) and the headline Map are
**density-gated** — thin or empty below a threshold of anchors. This is the reconciliation
cold-start problem made concrete. The surfaces that create value **immediately, from anchor
#1** are: the **deposit feeling good** (instant confirm + async enrichment), the
**dossier** (one entity is already useful), and **what-changed / resurface on extension**
(works as soon as one anchored node gets new corpus). Early-life value therefore rides on
**Mode 2 + dossier**; the Map/advise "wow" arrives at density.

Design consequences: (a) instrument **time-to-first-aha** as a primary metric; (b) seed the
first several anchors generously (high-confidence inferred deposits) without breaching the
smallness rule; (c) for density-gated surfaces, render an honest "first save about X" empty
state, never a barren panel.

---

## API Requirements (sketch — see RFC-081 for shapes)

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/anchors` | Create an Anchor (target ref(s) + provenance + optional note) |
| `GET` | `/api/anchors` | List the user's Anchors; params: `since`, `min_salience` |
| `DELETE` | `/api/anchors/{id}` | Remove an Anchor |
| `POST` | `/api/anchors/{id}/reinforce` | Reinforce salience (resurface acknowledge) |
| `GET` | `/api/l2/graph` | The derived personal subgraph (induced + weighted view over L0) |
| `GET` | `/api/l2/node/{ref}` | Anchor trace/dossier for one node |
| `GET` | `/api/reconcile` | "What changed" — join of user Anchors vs corpus changelog since watermark; params: `since`, `change_types` |
| `GET` | `/api/advise/episodes` | Unconsumed episodes ranked by L2 touch density; per-item L2-relevance |

`/api/l2/graph` and `/api/l2/node` are **always derived live** from L1 + current L0; the
personal layer is never read from a stored copy.

---

## Phasing

> **v1 is an internal value-chain dogfood, not a public release.** Its purpose is to write
> up the full model, build the UI surfaces, develop the use cases, and **close the value
> chain end-to-end** on the cheapest reconciliation cell (on-demand × extension). It is the
> proving ground for the rendering/UX contract (FR4.5) that every higher tier reuses. Once
> the chain is closed internally, we move **immediately** to turning the dials — no core
> rewrite, because the abstraction (RFC-081) already accounts for both.

- **v1 (internal dogfood):** L1 Anchor store + async resolution; L2 derived-view endpoints;
  **The Map**, Anchor trace, salience library; save-time Connect; Resurfacing card;
  Reconcile **on-demand + extension-only**; L2-relevance badge on catalog cards. Salience
  decay live from day one (it is what keeps v1 from becoming a graveyard). All surfaces
  consume `ReconcileEvent` tier-/type-agnostically (FR4.5).
- **Dial B — change-type richness (turn immediately after v1):** `contradiction` in
  reconcile (as conflict enrichers land), then `evolution` / `corroboration`. Each is an
  **enricher emitting into the changelog** (enricher-RFC / v2.7-registry work) — *not* a
  new reconciliation RFC. The join and the UI are unchanged; new change-types simply appear
  in surfaces that already render them. Also lands: staleness/conflict badges on the Map,
  auto-threads, "feeds your map" ordering, active recall.
- **Dial A — trigger tier (turn in parallel):** **on-demand → batch digest** is a new
  *driver* around the identical `reconcile()` join — implementation, not a new RFC (ADR at
  most). **→ live event-triggered** is the one tier that earns its own focused RFC (future
  RFC-082): it adds the reverse index `node_ref → users` and the RFC-065 event-spine
  fan-out that RFC-081 deliberately deferred. Gated on RFC-065 + v2.7 enricher registry.
- **Later:** frontier prompts; implicit deposit heuristics; possible shared/comparative
  subgraphs.

**What earns a new document, so the trajectory is unambiguous:** lower Dial-A tiers
(on-demand→batch) = implementation; the live tier = future RFC-082; every Dial-B change-
type = enricher-registry work. The reconciliation design (RFC-081) is **not** re-opened to
walk up the matrix.

---

## Success Criteria

1. A user with ≥ *threshold* Anchors can open **The Map** and recognise it as an accurate
   picture of what they retained — the single durable artifact of their listening.
2. The personal layer stores **zero L0 content**; every L2 view is reconstructable from
   L1 + current L0. (Verifiable by wiping the derived view and recomputing.)
3. **Reconcile** returns, on demand, correct extension events for anchored nodes, with
   correct provenance back to L0 — at a cost per user bounded by the user's *active*
   Anchor count, not corpus size.
4. Salience decay measurably curates: stale Anchors fade out of L2 views without user
   pruning, and reinforced Anchors persist.
5. The L2-relevance badge changes what users choose to play (measurable lift in plays of
   high-L2-touch episodes vs baseline ordering).

---

## Corpus composition (a precondition the model depends on)

The value this model creates is a function of **corpus shape**, not corpus size. It is a
design input, not an afterthought — a large corpus of the wrong shape produces a competent
bookmark tool and none of the differentiated value. Two constraints, which fail
independently:

- **Overlap (the firing floor).** Entities must **recur** across episodes within a user's
  plausible consumption path (floor ≈ ≥2–3 episodes per tracked entity). Without recurrence,
  every anchor is an island and connect / resurface / advise / reconcile all starve regardless
  of volume. Clustering the corpus is what manufactures this.
- **Divergence (the fuel for the differentiated value).** Within those recurrences, sources must **disagree, shift, or
  differ** — stance dispersion (opposing views on the same entity), temporal spread (the same
  entity over time), source/show spread (the same entity across distinct shows), and claim-level
  revisit (the same proposition with differing verdicts). The differentiated operations —
  contradiction-reconcile, evolution, cross-show synthesis, objectivization — live **entirely**
  in this dimension.

The failure mode to design against is **high-overlap / low-divergence** — an *echo chamber*
that makes connect and extension-reconcile work while the differentiated value starves, and which is invisible
unless dispersion is measured. The target is **high-overlap / high-divergence clusters seeded
with adversarial / divergent sources**, plus some **cross-cluster bridge entities** for
frontier/Map value. Concentrated-and-divergent beats broad-and-aligned; do **not** "just add
more shows."

This is verified, not assumed: run the **corpus value-potential diagnostic** (RFC-087 —
recurrence *and* dispersion, per cluster) over the real beta corpus **before** building. A
cluster that clears recurrence but fails dispersion is fixed by **seeding divergent sources
into it**, not by adding content. (Note: `last-50-per-show` favours freshness/reconcile-arrival
but caps long-arc evolution; consider deeper temporal backfill for a few key entities.)

> **Substrate quality floor:** all of the above assumes content **amenable to extraction**
> (analytical, discursive, entity- and claim-rich). Heavily narrative/comedy/music content
> yields thin GIL/KG and silently starves the whole stack regardless of overlap/divergence.

---

## Dependencies

- **Corpus composition** (above) — high-overlap **and** high-divergence clusters; the model's
  value is gated on it. Verify via the RFC-087 diagnostic before building.
- **RFC-072 Canonical Identity Layer** — Anchors must reference shared canonical refs, or
  reconciliation across shows is impossible.
- **GIL grounding contract** — Anchor → entity/claim resolution walks GIL provenance;
  faithfulness of that resolution is the critical quality dimension.
- **RFC-073 / RFC-074 enrichers** — emit the corpus-change signal (extension now;
  contradiction/evolution as they mature).
- **RFC-065 event spine + v2.7 enricher registry** — prerequisites for the batch/live
  reconciliation tiers (not for v1 on-demand).
- **`gi-kg-viewer` + RFC-077 graph modes** — rendering substrate for The Map.

---

## Spec map (which surface spawns which downstream doc, and when)

This PRD is the umbrella. `RFC-081` is the substrate every surface rides. Detailed
per-surface specs are spawned **just-in-time per build phase** — a surface earns its own
doc only when it adds either (a) interaction complexity → a **UXS**, or (b) new backend
algorithm beyond L2/reconcile → a **supporting RFC/ADR**. Otherwise the FRs here plus
RFC-081 *are* the spec. Do not pre-write the later specs.

| Surface | v1? | Carried by | New doc needed |
|---|---|---|---|
| The Map | v1 | PRD FR + **UXS** | **UXS** (layout, salience viz, RFC-077 modes, drill-to-dossier) |
| Dossier | v1 | PRD FR + UXS | same UXS as Map; backend `/l2/node` already in RFC-081 |
| Salience library | v1 | **PRD FR only** | none |
| What-changed | v1 | PRD FR + **UXS** | **UXS** (triage, acknowledge, badge propagation); backend = RFC-081 reconcile |
| Staleness badges | f-follow | PRD FR | none (rides reconcile + Map UXS) |
| Resurfacing card | v1 | PRD FR + UXS | trigger = RFC-081 join; card UX folds into the v1 UXS |
| Save-time connect | v1 | PRD FR + UXS | `/connect` neighborhood endpoint **added to RFC-078/079** (hybrid search), not a new RFC; UX in v1 UXS |
| Auto-threads | f-follow | PRD FR + **RFC** | **RFC** — clustering over the anchor set |
| L2 badge / feeds-map | v1 / f-follow | PRD FR + **RFC** | **RFC — L2 advise ranking** (salience-weighted touch score); `/advise/episodes` endpoint exists, scoring unspecced |
| Frontier prompts | later | PRD FR + RFC | bundle into the advise-ranking RFC (topology gap detection) |
| Active-recall | gated/later | PRD FR + **RFC** | **RFC** — grounded question-gen + double opt-in |

**Net new specs (not one per surface — four total, two for v1):**

1. **UXS — Knowledge Layer surfaces (v1):** Map + dossier + what-changed + resurface card +
   salience library. *(now)*
2. **RFC-078/079 addition:** `/connect` neighborhood query. *(now, small)*
3. **RFC — L2 advise ranking** (feeds-map score + frontier topology). *(fast-follow)*
4. **RFC — auto-threads clustering** and **RFC — active-recall generation**. *(later, gated)*

Everything else is fully carried by this PRD's FRs + RFC-081.

---

## References

- `docs/prd/PRD-026-podcast-platform.md`
- `docs/prd/PRD-029-platform-catalog.md`
- `docs/prd/PRD-030-platform-player.md`
- `docs/rfc/RFC-081-personal-knowledge-layer-reconciliation.md`
- `docs/rfc/RFC-072-canonical-identity-layer.md`
- `docs/rfc/RFC-065-agent-observable-instrumentation.md`
- `docs/rfc/RFC-073-enrichment-pipeline.md`
- `docs/rfc/RFC-077-graph-visualization-modes.md`
