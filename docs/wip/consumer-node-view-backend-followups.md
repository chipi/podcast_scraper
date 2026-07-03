# Consumer node-view — backend follow-ups (feat/consumer-remember)

Two follow-ups surfaced while shipping the review-round node-view work on
`feat/consumer-remember` (person/topic/podcast panels, Across-shows, back-nav,
"Appears in" shows, timeline mention drill). Both were shipped as **viewer-side
approximations**; the precise versions need backend work and are captured here.

Status: **Backlog** (viewer approximations already shipped; these are the
accurate versions). Operator files GH issues.

---

## 1. Precise per-show host/guest — pipeline `person → hosts → show` edge

**Shipped approximation** (`3809300a`): the person panel infers "Host" per show
from *episode coverage* — a host recurs across most of a show's episodes, a guest
appears in a few. It only claims "Host" when coverage is confident (≥50% and ≥2
of the show's episodes) and never asserts "Guest".

**Why it's only an approximation:** the corpus is a *sample*, so a real host who
appears in few sampled episodes reads as low-coverage (e.g. Katie Martin is an
Unhedged host but only 1 of 10 sampled Unhedged episodes has her → no Host
claim). The person node also carries a single coarse `role` (host/guest/
mentioned) with **no per-show link**.

**Precise fix (pipeline):**
- Emit a typed `person —hosts→ podcast` (and/or `person —guests_on→ podcast`)
  edge during GI/KG extraction, sourced from feed metadata (RSS `<itunes:author>`
  / author fields) and/or a per-episode speaker-role signal, not transcript
  coverage.
- Surface it via a relational query (e.g. `/api/relational/shows?person=…`
  returning `{show, role}`), and have `PersonLandingView` read the real role
  instead of the coverage heuristic (drop `showTotalEpisodes` inference).
- Viewer change is then small: replace `isHost` (coverage) with the endpoint's
  role, keep the same chip UI.

---

## 2. Out-of-slice insight rendering — insight-detail endpoint

**Shipped behaviour** (`566a9f69`): timeline mentions are clickable and drill
into the insight's node view (`subject.focusGraphNode(insightId)`), with Back.
This resolves for insights **in the loaded graph slice** (the common case for a
focused topic's own mentions).

**The gap:** the topic Timeline is corpus-wide (CIL), so many mentions are
insights **outside** the viewer's loaded graph slice. NodeDetail's out-of-slice
gate (`inferredKindFromId`) covers person/topic/org/podcast — which all have
server-backed detail — but **not insight**, so an out-of-slice mention click
lands on the empty "Node" rail.

**Why it can't be a viewer-only fix:** an insight's detail is its **text +
supporting quotes + connections**, which live in the graph node. For an
out-of-slice insight, that node isn't in the artifact, and the relational routes
(`entities_in`, `related_insights`, `insights_about`) return *related* nodes for
an insight, never the insight's **own** text/quotes.

**Fix (backend endpoint + small viewer view):**
- Add `GET /api/relational/insight-detail?insight=<id>` (or `/api/cil/insights/{id}`)
  that resolves the insight on the **full server graph** (`_graph_or_none`, which
  already spans the whole corpus, not the slice) and returns its `text`,
  supporting `SUPPORTS`/quote nodes, `ABOUT` topics, and `MENTIONS` entities.
- Extend `inferredKindFromId` + the `<aside>` gate to `Insight`, and add a small
  `InsightNodeView` (text + quotes + related-insights via the existing
  `related_insights` endpoint) that renders from the endpoint — same pattern as
  the person/topic out-of-slice views.
- Then the mention drill works corpus-wide, with Back.

The same endpoint would also let an out-of-slice **episode**'s insights render;
episodes themselves already open via `focusEpisode` (Library panel) regardless of
slice, so no episode-specific work is needed.

### Also blocked on this endpoint: contradiction "what exactly" (N7)

The person Signals **Contradictions** section (`NodeEnrichmentSection`) names the
counterpart + topic, both click-through (person → focusPerson, topic →
focusTopic → the topic's Key voices, which shows both takes with text). The
operator asked to also show *what exactly* was said, inline. The
`nli_contradiction` envelope carries `insight_a_id` + `insight_b_id` +
`contradiction_score` but **not** the insight texts, and no existing endpoint
resolves an arbitrary insight id → text corpus-wide:

- `who-said?topic=` returns per-person insights for a topic but caps at `k=20`
  per person, so the specific contradicting insight isn't guaranteed to be in
  the window — a viewer-only resolution would silently miss rows.
- The `/brief` `topics` map only covers the **focused** person's own insights,
  never the counterpart's.

The same `insight-detail?insight=<id>` endpoint from follow-up #2 resolves both
sides reliably. Viewer change is then small: read `insight_a_id`/`insight_b_id`
(already in the envelope), resolve each to text, render the two statements under
the contradiction row. Until then the topic click-through is the "what exactly"
path.
