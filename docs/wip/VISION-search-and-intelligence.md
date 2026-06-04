# Vision: Search as Intelligence Substrate

*Grounding document for PRD-032, PRD-033, RFC-078, RFC-079*

---

## Where this started

The starting point for this work was a study of code intelligence tools — lean-ctx, ctxo, and the broader context-engineering space emerging around AI agents. These tools solve a specific problem: agents navigating large corpora of interrelated artifacts with no structural understanding of what they're reading. They developed a clear answer — property graphs, hybrid retrieval, intent-aware context assembly — and proved it works at scale.

The observation that matters: **a podcast corpus and a code repository are structurally identical at the retrieval level.** Both are graphs of typed entities with typed edges. Both have the same challenge — given an intent, surface the right subgraph without drowning the consumer. The techniques transfer directly. You just rename the nodes.

That observation is what kicked off this entire body of work.

---

## The deeper mission

Podcast Scraper exists to help people **see through narratives and deception** — to objectivize information by exposing how narratives are constructed, where they conflict, how positions evolve, who is saying what and why.

This is a harder problem than it sounds. It requires:

- Not just retrieving what was said, but understanding *why it matters* that it was said
- Not just finding mentions of a topic, but mapping the full terrain of how a topic is discussed across shows, over time, by different voices
- Not just surfacing claims, but grounding them in evidence — specific quotes, specific moments, traceable provenance

Podcast is the first domain to prove this model. It's a controlled corpus (known shows, known speakers, known topics), which makes it the right place to build and validate the objectivization enrichers before expanding to broader media. The sequencing is deliberate: podcast product first, then media intelligence product. Get the intelligence layer right on a constrained corpus, then open it up.

---

## What the current stack can and cannot do

The current platform — GIL, KG, FAISS, enrichment layer, MCP tools — is sophisticated. The grounding invariants in GIL are a genuine architectural innovation. The canonical identity layer (RFC-072) solves a hard entity disambiguation problem cleanly. The enrichment pipeline (RFC-073) is well-designed.

But the retrieval layer is a ceiling. Single-signal FAISS over insight nodes only means:

- Named entities (person names, show names, specific terminology) are systematically under-retrieved
- Raw transcript evidence is invisible — if it didn't make it into a GIL insight, it doesn't exist to the search layer
- The knowledge graph — with all its typed edges, entity relationships, relational structure — contributes nothing to ranking
- Every query is treated identically regardless of intent
- Agents receiving MCP tool output get unstructured data dumps, not shaped intelligence

The platform stores and enriches podcast intelligence well. It cannot yet *deliver* it with the precision and structure that the objectivization mission requires.

---

## What this work delivers

PRD-032 and its implementing RFCs (RFC-078, RFC-079) are not a search feature. They are a **retrieval infrastructure upgrade** that changes what the platform fundamentally is.

Specifically:

**Two-tier indexing** means the corpus is now fully searchable — raw transcript segments and synthesized GIL insights both contribute to retrieval. A user looking for an exact quote finds the segment. A user looking for a synthesized position finds the insight. A compound result delivers both when they refer to the same moment. The corpus is no longer partially dark.

**Hybrid retrieval (BM25 + vector + KG proximity)** means retrieval quality is no longer bounded by the weakest signal. Named entities surface correctly. Semantic meaning is captured. Relational context from the KG — the graph structure you've built — finally contributes to what gets returned. The knowledge graph stops being a display layer and becomes a retrieval substrate.

**Intent-aware query routing** means the platform understands what you're asking. A person lookup, a cross-show synthesis question, a raw evidence search, a temporal position-tracking query — each gets a different retrieval strategy. The platform adapts to the question rather than applying one strategy to everything.

**LITM-aware MCP context packs** mean agents receive intelligence, not data. The `corpus_briefing_pack` tool delivers a structured, compressed, evidence-grounded context document — canonical entity definition at the top, strongest insight, detected contradictions, supporting raw evidence, coverage summary. This is the difference between a file server and a briefing analyst.

**PRD-033** then maps these capabilities across every viewer surface — Library, Digest, Detail panels, Graph, Dashboard. The retrieval upgrade propagates: topic bands in Digest rank by evidential strength, Person Landing panels are grounded by retrieval queries, Graph nodes reflect insight density, Dashboard cards carry drillable evidence. The platform coherence improves because every surface draws from the same intelligent retrieval layer.

---

## The moat

Code intelligence tools like lean-ctx and turbopuffer provide the retrieval infrastructure. They are domain-agnostic — they work on any codebase, any corpus. That's their strength and their ceiling.

Podcast Scraper's moat is exactly what those tools don't have: **a domain ontology**.

- A curated taxonomy of show clusters and topic hierarchies specific to the podcast intelligence domain
- Rhetorical move detection and position tracking enrichers that understand how podcast discourse works
- Contradiction detection that understands the difference between a speaker changing their mind and two speakers genuinely disagreeing
- A provenance chain (quote → speaker → episode → show → enricher version → confidence) that makes every claim attributable and auditable

The retrieval infrastructure (BM25, RRF, KG proximity) is table stakes by 2026. Any team can build it. What no generic tool can replicate is the domain layer on top — the objectivization enrichers, the media intelligence ontology, the grounding contract that makes every insight traceable to its source.

Build the retrieval foundation fast. Go deep on the domain layer where no one else can follow.

---

## How the PRDs and RFCs connect to this

| Document | Role in the vision |
|---|---|
| RFC-078 | Builds the retrieval foundation — two-tier index, BM25+vector+RRF, SearchBackend abstraction, LanceDB |
| RFC-079 | Adds the domain-specific third signal (KG proximity), the intelligence delivery layer (LITM context packs), and the ML router that improves with corpus query data |
| PRD-032 | Product-level spec for the retrieval capability — what it delivers, to whom, measured how |
| PRD-033 | How the capability propagates across surfaces — Library, Digest, Detail, Graph, Dashboard all change |

Together they move the platform from **"a tool that stores and enriches podcast content"** to **"an intelligence substrate that agents and humans can query with intent, receive grounded evidence from, and trust the provenance of."**

That is what the objectivization mission requires at the infrastructure level. The enrichers that detect narratives, track positions, and surface deception run on top of this. Without a retrieval layer that can deliver their outputs precisely and credibly, those enrichers produce intelligence that nobody can find or trust.

---

## What comes next

The search foundation and surface propagation (PRD-032, PRD-033) are v2.7 and podcast product work. The next architectural layer — which this work explicitly reserves slots for — is:

**Contradiction as a first-class KG edge type.** Multiple surfaces (Digest bands, Graph, Topic Entity View, Dashboard, LITM context packs) have placeholder slots waiting for this. It's the most depended-upon missing piece and the most direct architectural expression of the objectivization mission. A contradiction that's a typed, queryable, traversable KG edge is not just a detected artifact — it's navigable intelligence.

**Corpus impact surface (RFC-080).** The blast-radius analogue: for any topic or person, compute the full surface before synthesis. Coverage breadth, contradiction density, temporal span, position volatility. This is how agents and users plan before they query.

**Objectivization enrichers.** Once retrieval is solid and contradictions are first-class edges, the enrichers that detect rhetorical moves, track position evolution, and surface narrative construction can run against a retrieval substrate that can deliver their outputs accurately. This is where the domain moat deepens.

The sequencing is right: infrastructure first, domain intelligence on top.
