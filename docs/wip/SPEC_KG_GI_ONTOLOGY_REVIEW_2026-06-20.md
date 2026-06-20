# KG + GI ontology review — first draft (2026-06-20)

**Status**: WIP draft, iterative. Not a PRD/RFC yet — this is the
"thinking out loud" doc we chat over until we agree on (a) what's
actually broken, (b) what we want to be true, (c) how to evolve.

**Trigger**: Your read after #1035/#116/#113 — "our KG extraction is
not good." NER pre-pass closed the entity gap, but topic recall caps
in the 65–77% band even on the strongest model, and you instinctively
feel the layer isn't pulling its weight yet. This doc is to figure out
whether the issue is the *extraction* (we're not asking the LLM the
right questions), the *ontology* (we don't have the right slots to
put the answers in), or the *goal* (we never decided what the layer
is FOR).

I'll lay out what I see, mark **`?`** for things I need your call on,
and **`!`** for places I have a strong opinion. Push back freely.

---

## 0. Why this review now

| Signal | Reading |
|---|---|
| KG topic coverage caps 65-77% even with v5 prompt + NER hints | Either we ask wrong, or silver expects things we don't model |
| KG schema explicitly bans Insight semantics ("separation from GIL") | Two siblings that don't talk to each other — by design, but is the design right? |
| GI Entity is "deferred to v1.1" and v1.1 never happened | We have entities in KG, "deferred" in GI, and the silver expects them in both — mismatch |
| RFC-091 (2026-06-03) rejected KG proximity for retrieval | Decision said retrieval value lives in **cross-layer** Insight→MENTIONS→Entity edges (in GI) — but those don't currently get emitted |
| Ontology docs say things like "Topic — abstract subject discussed" | Both KG and GI use the same one-line definition for Topic. That's a smell |

**My read**: We have two reasonable ontologies (KG = browse/discover,
GI = trust/evidence) that were specified independently and never had
their *interfaces* designed. The result is two impoverished graphs that
each carry half of what we'd actually want for a unified corpus surface.

**?** Does that match your gut, or do you think the problem is more
narrowly inside KG?

---

## 1. What are we trying to build (the goals question)

Before we touch the ontology, I want to write down what we think the
KG + GI layers ARE FOR. Because every gap discussion downstream depends
on this. Here's my current model — please correct.

### KG goal (as I read PRD-019 + RFC-055 + the code)

> Episode-anchored map of *what an episode is about* (topics) and *who
> is mentioned in it* (entities), browsable + linkable across the
> corpus.

Surfaces it should feed:
- Viewer "graph" tab — clickable Topic / Entity chips
- Topic clustering (`search/topic_clusters.py`) — Pareto threshold 0.75
- Cross-episode discovery: "show me other episodes that mention Maya"
- Cross-feed corpus view (added in #658 via `Episode.feed_id`)

Surfaces it does NOT serve (today):
- Q&A / retrieval (RFC-091 explicitly rejected this)
- Truthfulness verification (that's GI)
- Episode-internal narrative (the ABOUT edge from Insight → Topic lives in GI's schema, not KG's)

### GI goal (as I read PRD-017 + RFC-049 + the ontology doc)

> Grounded, evidence-backed claims (Insights) extracted from episode
> content, each linked to verbatim Quotes from the transcript, with a
> hard **grounding contract**: `grounded=true` iff ≥1 `SUPPORTED_BY`
> edge to a verbatim Quote.

Surfaces it feeds:
- `gi explore` CLI
- Future RAG / claim-verification UX
- Trust dashboards ("X% of insights grounded")

The grounding contract is the part that's **load-bearing for trust**.
Without it, an Insight is just an LLM paraphrase. With it, the Insight
is traceable back to the speaker's actual words.

**?** Is GI's grounding contract still load-bearing for you? Or has the
project drifted toward "the whole thing is summarization + KG and GI
is a side feature"? Because where you sit on that changes everything
downstream.

### Where the two should meet (but don't yet)

The RFC-091 decision pointed at this explicitly:
> the KG's retrieval value comes from **meaning-bearing relational
> edges in the cross-layer graph**: `Person→Insight` (`SPOKEN_BY`),
> `Insight→MENTIONS→Entity`, `Podcast→HAS_EPISODE→Episode`

That cross-layer graph isn't a thing today. GI defines an Insight →
MENTIONS → Entity edge (line 91 of GI ontology), but Entity is
"deferred" so the edge has nothing to point at. KG defines an Entity
type but no Insight type. Both define Topic but with the same
hand-wavy one-liner.

**!** I think this is the real KG quality issue. We're not extracting
"badly"; we're extracting into a too-thin shape that throws away most
of what we know.

---

## 2. Current ontology — what's actually shipped

(Skipping aspirational stuff. This is what `build_artifact` produces +
what the schemas validate.)

### KG (`kg.json`, schema v1.2)

```
Nodes:    Episode | Topic | Entity(kind=person|org)
Edges:    MENTIONS (Topic|Entity → Episode)
          RELATED_TO (defined but never emitted)
```

Properties worth noting:
- `Topic.label` + `Topic.slug` (required); `Topic.description` optional
- `Entity.name` + `Entity.kind`; `Entity.role` ∈ {host, guest, mentioned}; `Entity.description` optional
- `Episode.podcast_id`, `Episode.title`, `Episode.publish_date`, `Episode.feed_id` (cross-feed; #658)

**What's missing in practice**: any structural relationship between
Topic and Entity. We know "Maya talked about braking technique" but
the graph only says "Maya → mentions → episode" and "braking technique
→ mentions → episode". The cross-product is lost.

### GI (`gi.json`, schema v2.0)

```
Nodes:    Podcast | Episode | Person | Topic | Insight | Quote
          (Entity defined but deferred to v1.1 — never landed)
Edges:    HAS_EPISODE       Podcast → Episode
          SPOKE_IN          Person → Episode
          HAS_INSIGHT       Episode → Insight
          SUPPORTED_BY      Insight → Quote    (the grounding contract)
          SPOKEN_BY         Quote → Person
          ABOUT             Insight → Topic    (when enriched)
          MENTIONS          Insight → Entity   (#874 — but Entity is deferred)
          RELATED_TO        Topic ↔ Topic      (optional)
```

GI is the richer ontology by far. The grounding contract makes it
*trustworthy*. The Insight type is the closest thing we have to "what
this episode is actually claiming."

**!** Things I notice immediately:
- `Insight.MENTIONS.Entity` is the bridge that #874 / RFC-091 said
  matters most, and we ship neither the Entity node in GI nor any
  attempt at the edge. That's a big hole.
- `ABOUT (Insight → Topic)` exists in the schema but ontology doc says
  "not produced automatically by the default pipeline yet"
- `Person` in GI vs `Entity(kind=person)` in KG — same real-world
  thing, two storage slots, no canonical bridge. The dataclass shape
  is even different.

### Silver structure (what we eval against)

Sampled `silver_opus47_kg_dev_v1::p01_e01`:
- 11 silver topics, each with a 1-sentence description
- 3 silver entities (Maya, Liam, Singletrack Sessions) — no `entity_kind`
- Silver shape is **flat**: `output.topics = [...]`, `output.entities = [...]` — NOT the nested KG-artifact shape we ship

Sampled `silver_opus47_gi_dev_v1::p01_e01`:
- 8 silver insights, each as `{text: "..."}` — no quotes, no grounding,
  no topics, no entities
- Shape: `output.insights = [...]`

**Observations**:
1. The silver is *flat schema, no grounding* — it's the LLM's
   best-effort knowledge extraction, not the trustworthy contract our
   GI schema expects
2. Topics in the KG silver have descriptions; topics in the GI silver
   don't exist at all (decoupled from insights)
3. Entities in KG silver lack `entity_kind` — the silver doesn't know
   person vs org (we infer it downstream)

**?** Is the silver methodology aligned with what we *want* to measure,
or is it just measuring whatever a good LLM happens to produce? If the
former, we should evolve the silver to test the richer ontology. If the
latter, we should pick.

---

## 3. Extraction → ontology mapping (today)

```
Transcript
    ↓ (summary stage — LLM)
Summary + bullets
    ↓ (#1034 deleted the bullets → KG path)
Transcript again (provider source)
    ↓ (NER pre-pass adds PERSON+ORG hints)
    ↓ (LLM call with v5 prompt)
{topics: [...], entities: [...]}     ← LLM output
    ↓ (kg/pipeline.py::build_artifact)
KG artifact: Episode + Topic nodes + Entity nodes + MENTIONS edges
    ↓ (kg/filters.py: normalize topic labels, repair entity_kind, consolidate names)
Final kg.json
```

For GI:
```
Transcript
    ↓ (gi.build_artifact)
GI artifact: Episode + Insight nodes + Quote nodes + SUPPORTED_BY edges
    ↓ (grounding evidence stack: QA + NLI)
gi.json with grounded=true/false per insight
```

**The mapping has these specific information losses**:

| What the LLM knows | What we keep |
|---|---|
| Maya talks about braking on the descent at minute 12 | "Maya is mentioned in this episode" + "braking technique is mentioned in this episode" — no link |
| The insight "brake earlier and smoother" is something Liam said | Insight emitted; speaker attribution lost (unless diarization aligns) |
| Braking technique and tire pressure are both "ride quality" topics | No topic hierarchy / category |
| Singletrack Sessions is the show, Maya is the host, Liam is the guest | KG `Entity.role` knows host/guest from speaker pipeline, but the show entity has role=mentioned, not "show" |
| The silver insight "good trail drainage preserves both rideability and soil structure" maps to silver topic "trail drainage and durability" | Neither side knows; ABOUT edge could express it but it's not emitted |

**!** I claim: the topic coverage cap (65–77%) isn't really a model
quality problem. It's a *taxonomy* problem — we're asking the LLM to
produce labels that match the silver's exact phrasing on a small list,
and the silver itself isn't producing a canonical taxonomy (each
episode's 10 topics are picked fresh by the LLM, ad hoc). Even a
perfect extractor would land at ~70% because the labels are inherently
underspecified.

The fix isn't a better prompt. It's giving topics structure.

---

## 4. Gaps — concrete list

### G1. Topic ontology is too flat

Topic only has `label`, `slug`, `description`. There's no:
- **Category / domain** ("ride craft" vs "gear" vs "trail building")
- **Parent topic** ("braking technique" is a child of "ride craft")
- **Topic-to-topic relationships** (RELATED_TO is reserved but unused)
- **Topic-to-entity links** (no edge type)

**?** Do you want a flat topic taxonomy (what we have, just better
labels) or a hierarchical one (categories + sub-topics)? Hierarchical
is more work but enables cross-episode "show me all braking
discussions" via category, not literal label match.

### G2. KG ↔ GI bridge is missing

The cross-layer graph that RFC-091 said matters for retrieval doesn't
exist. Specifically:
- No way to ask "which insights mention Maya" (Insight + Entity live
  in different artifacts, no shared id space for Entity)
- No way to ask "what does this episode say about braking technique"
  (Insight.ABOUT.Topic isn't emitted)
- No way to ask "what insights involve the show host vs the guest"
  (Person/Entity split + Insight not attributed)

**!** The cheapest, highest-leverage move is probably: **ship the
Insight.MENTIONS.Entity edge in GI**, reusing KG's already-extracted
entities. That's the #874 commitment that RFC-091 leaned on.

### G3. Entity vs Person duplication

KG has `Entity(kind=person)`. GI has `Person`. Same real-world thing,
two ontology slots, no migration story. Today:
- Diarized speakers → GI `Person` nodes via SPOKEN_BY
- Show hosts/guests → KG `Entity(kind=person, role=host|guest)`
- LLM-extracted person mentions → KG `Entity(kind=person, role=mentioned)`

A guest who speaks gets both a KG Entity *and* a GI Person, with no
edge between them.

**?** Should there be ONE canonical Person concept (the GI one) that
KG references, or do you actually want them separate (KG's Entity =
"a name appearing in this episode regardless of role" and GI's
Person = "a diarized speaker")?

### G4. Topic vs Insight tension

GI's Insight is roughly "what this episode is *saying* about a topic."
KG's Topic is "what this episode is *about*." These overlap heavily.
The silver:
- KG topic: "braking technique for speed"
- GI insight: "Riding speed improvements come more from braking earlier
  and smoother than from taking bigger risks"

The insight is the *claim* attached to the topic. The schemas treat
them as orthogonal (Insight.ABOUT.Topic edge defined but not emitted).

**?** Is an Insight always about exactly one Topic? Multiple? Should
the relationship be modeled at all, or are they separate axes?

### G5. The "show" / "podcast" boundary is fuzzy

Today:
- KG models `Episode.podcast_id` and `Episode.feed_id` (string fields)
- GI ontology defines a `Podcast` node + HAS_EPISODE edge — but the
  current GI pipeline doesn't emit it (per the ontology doc's shipping
  note, only Episode/Insight/Quote/SUPPORTED_BY are produced)
- KG has no `Podcast` node at all

When the LLM extracts "Singletrack Sessions" as an org, that's actually
the show name — it ought to be the same identity as the Podcast node.
Today it gets a generic `Entity(kind=org, role=mentioned)` in KG and is
absent from GI.

**?** Do you want a first-class `Podcast` / `Show` node, with the same
identity used in KG and GI? Or is `feed_id` sufficient?

### G6. Quote → Topic / Quote → Entity is missing

GI grounds insights with quotes. But the quotes themselves don't link
to topics or entities. That means a Quote about "braking technique"
spoken by "Liam" doesn't carry that info — only the Insight does.

**?** Does Quote want to be a first-class addressable thing (so you
can search "show me quotes about braking") or stay as Insight's
evidence-only annotation?

### G7. Schema versioning is uncoordinated

- KG schema is at 1.2
- GI schema is at 2.0
- They version independently, with no doc explaining what changed
  *across* them

**!** Not load-bearing, but smells. If we evolve the cross-layer
edges, we need a coordinated version bump.

### G8. Silver methodology may need to evolve too

The current silvers (Opus 4.7 for KG, dev_v1) test what we ship now.
If we evolve the ontology, the silvers also need to evolve — otherwise
we'd be measuring the new pipeline against the old shape. That's a
not-cheap follow-up if we touch ontology.

---

## 5. Forward sketches (not committed)

Some directions a real evolution could go. Picking between them is your
call once we agree on the gap list.

### Direction A — "minimal bridge" (low risk)

- Ship `Insight.MENTIONS.Entity` edge in GI by re-using KG entities
  via shared id space (`person:{slug}` already canonical per RFC-072)
- Ship `Insight.ABOUT.Topic` edge in GI by re-using KG topics
- Keep schemas otherwise unchanged
- Adds the cross-layer retrieval RFC-091 wanted, without ontology surgery

**Effort**: small. **Quality lift**: significant for retrieval / Q&A.

### Direction B — "unified ontology" (bigger)

- One shared schema (`corpus.schema.json`?) covering both layers
- One Topic concept, one Entity concept, one Person concept (no Entity
  vs Person split)
- Insight + Quote are nodes that live alongside Topic + Entity in the
  same graph, with the grounding contract preserved
- KG and GI become *views* over the same graph, not separate artifacts

**Effort**: large (migration, scorer, viewer, search). **Payoff**: the
ontology actually says what's in the corpus.

### Direction C — "domain-aware topics" (orthogonal)

- Add Topic taxonomy: categories (e.g. "ride craft", "gear",
  "navigation"), each Topic gets a `category` property
- Adds RELATED_TO edges as is-a / sibling-of relations
- Could land independently of A or B
- Need a domain dictionary (manually curated initially? auto-generated
  from corpus-wide topic clusters?)

**Effort**: medium. **Payoff**: better browse + retrieval. Doesn't fix
the grounding/Insight story.

### Direction D — "evaluate harder before evolving" (skeptical)

- Maybe the topic coverage cap is fine and we're solving a
  non-problem
- Pull the actual viewer / corpus-search UX and see if the missing
  edges are felt by users
- Could decide nothing needs to change

**Effort**: zero. **Payoff**: don't over-engineer.

---

## 6. Questions I want your call on

In rough priority (the answer to each unlocks the next):

1. **Goal alignment** — Is GI's grounding contract still load-bearing
   for the project, or has the center of gravity moved toward "KG +
   summary, GI is gravy"?

2. **User-facing surface** — What's the primary place a human (you,
   reader, future user) sees this graph? Viewer graph tab? CLI search?
   Future RAG? The answer changes which gaps matter.

3. **Cross-layer bridge** — Are you OK with KG and GI sharing an id
   space and emitting cross-layer edges (Direction A minimum)?

4. **Entity vs Person** — One concept or two?

5. **Topic taxonomy** — Flat (today) or hierarchical (Direction C)?

6. **Show / Podcast as first-class** — Yes or no?

7. **Evaluation methodology** — If we evolve the ontology, do we need
   new silvers? Are you OK with breaking the #1033/#1035 baselines?

---

## 7. What I think we should do next (if you agree)

Two small concrete things to do before any actual coding:

1. **You answer questions 1–4 above.** That bounds the scope.
2. **I do a "round-2" version of this doc** that proposes a specific
   ontology shape based on those answers, with concrete schema diff
   sketches and a migration plan.

Then we decide whether to ship that as an RFC (formal), an ADR
(decision), or just commit the ontology v2 docs + plan the code in
parallel.

This doc is small enough to iterate over — feel free to scribble
inline, push back on framing, add new gaps I missed.

---

## Appendix — files referenced

| File | Purpose |
|---|---|
| `docs/architecture/kg/ontology.md` | Current KG ontology spec (v1.2, frozen) |
| `docs/architecture/kg/kg.schema.json` | KG validator |
| `docs/architecture/gi/ontology.md` | Current GI ontology spec (v2.0) |
| `docs/architecture/gi/gi.schema.json` | GI validator |
| `docs/rfc/RFC-055-knowledge-graph-layer-core.md` | KG design rationale |
| `docs/rfc/RFC-049-grounded-insight-layer-core.md` | GI design rationale |
| `docs/rfc/RFC-091-retrieval-decision.md` | Why KG proximity was rejected for retrieval |
| `docs/rfc/RFC-072-entity-id-migration.md` | Person/Entity id migration |
| `src/podcast_scraper/kg/pipeline.py` | KG extraction → artifact mapping |
| `src/podcast_scraper/gi/pipeline.py` | GI extraction → artifact mapping |
| `data/eval/references/silver/silver_opus47_kg_dev_v1/predictions.jsonl` | KG silver (flat shape) |
| `data/eval/references/silver/silver_opus47_gi_dev_v1/predictions.jsonl` | GI silver (insights only) |
