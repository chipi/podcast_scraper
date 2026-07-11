# Consumer node-view — backend follow-ups (feat/consumer-remember)

Three follow-ups surfaced while shipping the review-round node-view work on
`feat/consumer-remember` (person/topic/podcast panels, Across-shows, back-nav,
"Appears in" shows, timeline mention drill). Each was shipped as a **viewer-side
approximation**; the precise versions need backend work and are captured here
(#1 open, #2 done, #3 open — #3 rides on #1's missing edge).

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

## 2. Out-of-slice insight rendering — insight-detail endpoint — ✅ DONE

**Resolved.** `GET /api/relational/insight-detail?insight=<id>` (`rq.insight_detail`)
returns the insight's text + type + grounded flag, its `SUPPORTED_BY` quotes, its
`ABOUT` topics, and its `MENTIONS` entities from the full corpus graph.
`NodeDetail.inferredKindFromId` + the `<aside>` gate now cover `Insight`, and a new
`InsightNodeView` renders from the endpoint (topics/entities click-through; the
resolved text is emitted up so the header shows the claim, not the hash). The
timeline-mention drill now works corpus-wide, with Back. (A running API started
before this route must be restarted to serve it.)

Original context below (kept for the record).

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

### Contradiction "what exactly" (N7) — SHIPPED (not via this endpoint)

The person Signals **Contradictions** section (`NodeEnrichmentSection`) named the
counterpart + topic but not the opposing statements. The fix turned out to be at
the **producer**, not a new endpoint: the `nli_contradiction` enricher already
reads both insight texts to feed the NLI scorer, but only persisted the two
insight *ids* — dropping the texts it had in hand.

`nli_contradiction` v1.1.0 now persists `insight_a_text` / `insight_b_text` per
record; `NodeEnrichmentSection` reads them (oriented to the focused person) and
renders the two claims under each row. No id → text endpoint needed, no
`who-said` k=20 truncation risk. Prod-v2's envelope was regenerated
(`--only nli_contradiction --with-ml`, 660 records, all with both texts).

Note the out-of-slice insight endpoint (#2) is still wanted for the *timeline
mention drill* — that's a different consumer and remains open.

---

## 3. Per-show role "Host of" + back-catalogue drop — reuses #1's `person → hosts → show` edge

**Shipped approximation** (`EntityCardBody.vue`, `// Per-show role (#3 follow-up)`):
the person panel surfaces the shows a person **hosts** in a "Host of" section up top
and drops those shows' back-catalogue from the episode list below (a daily-show host
shouldn't list 500 of their own episodes; show their appearances on *other* shows
instead). Host-vs-guest per show is inferred viewer-side from a `hostFeedIds`
coverage filter — the same episode-coverage heuristic as #1.

**Why it's only an approximation:** identical root cause to #1 — there is no typed
per-show role from the pipeline, so the "Host of" set and the back-catalogue drop
both ride on coverage inference and misfire on sample-sparse hosts.

**Precise fix:** none beyond #1 — once the `person → hosts → show` edge + relational
role query land, `hostShows` reads the real role and the back-catalogue drop keys off
it. Tracked here so the viewer approximation isn't mistaken for the accurate version.
