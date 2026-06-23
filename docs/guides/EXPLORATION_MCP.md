# Exploring the corpus through MCP

The exploration MCP (RFC-095, `src/podcast_scraper/mcp/`) lets an agent — Claude or any MCP
client — *explore the knowledge graph your pipeline builds*: people, topics, positions,
insights, and how they connect **across shows**. It's read-only over a built corpus.

## Run it

```bash
python -m podcast_scraper mcp --corpus /path/to/corpus    # stdio (the agent-client transport)
```

## The one rule: resolve, then traverse by id

Names are ambiguous; ids are not. So:

1. **`resolve_entity("Kara Swisher")`** → a canonical id (`person:kara-swisher`, with a
   confidence score). Do this **first** — every other tool takes ids, not names.
2. Then **address by id** and traverse the graph.

## The value: connectivity

The corpus isn't a pile of episodes — it's a *graph*. A person **states** insights; insights
are **about** topics and **mention** entities; topics span **shows**. The tools let an agent
walk those edges. The keystone is one call that hands back the whole local neighborhood:

### `entity_neighborhood(id)` — understand any entity in one call

```text
entity_neighborhood("person:kara-swisher") →
  { ok, kind: "person", subject: {id, label},
    data: { stated: [...],        # insights she stated
            mentioned_in: [...],  # insights about her
            topics: [...],        # what she engages (AI ethics, regulation, ...)
            co_speakers: [...],   # who else speaks on her topics (the social graph)
            shows: [...] },
    note: "" }                    # why empty/sparse, when it is
```

For a **topic** it returns the entities, the speakers, and the **cross-show synthesis** (one
insight per distinct show). For an **org**, what mentions it; for a **podcast**, its episodes.
One call replaces 4–5 chained ones — and it doesn't dead-end on topics or co-people.

**Every tool returns the same envelope — `{ok, data, note}`.** Check `ok` (a tool that errors
comes back `ok: false` with the reason in `note`, never a crash), read the payload from `data`,
and use `note` to learn *why* a result is empty ("no insights — likely an unnamed speaker")
instead of guessing per-tool shapes or confusing *no data* with *feature off*. The connectivity
tools additionally carry `kind` + `subject` at the top level.

## Use cases (what an agent can now answer fluidly)

| Question | How |
| --- | --- |
| *"Tell me everything about Kara Swisher."* | `resolve_entity` → **`entity_neighborhood`** (one call) |
| *"What topics does she engage?"* | `person_topics(person_id)` (or the `topics` facet above) |
| *"How has her position on AI regulation evolved?"* | `resolve_entity` ×2 → `position_arc(person_id, topic_id)` |
| *"Who else is in this conversation?"* | `co_occurring_entities(person_id)` — people on shared topics |
| *"How are two people related?"* | `bridge(person_a, person_b)` — shared topics + do they co-occur |
| *"What themes sit next to this one?"* | `related_topics(topic_id)` — topics that share insights |
| *"What do different shows say about AI safety?"* | `resolve_entity` → `cross_show_synthesis(topic_id)` (the corpus differentiator) |
| *"Who said what about a topic?"* | `who_said_about_topic(topic_id)` — grouped by speaker |
| *"Find the exact quotes / the claims around X."* | `search_corpus(query, tier="segment"` or `"insight")` → `related_insights` |
| *"Where do I start?"* | `top_people` / `list_feeds` / `list_episodes` → pick an id and explore |

## The shape of a session

A typical agent loop: **`resolve_entity`** (name → id) → **`entity_neighborhood`** (the lay of
the land) → drill in with a focused tool (`position_arc`, `cross_show_synthesis`,
`search_corpus`) → fan out via **`co_occurring_entities`** to the next voice. Address by id,
let the graph carry you across shows — that's the intelligence the corpus exists to provide.

## Also in the viewer (same brain)

These traversals aren't MCP-only. The viewer's `/api/relational/*` routes wrap the **same**
`relational_queries` layer, so the human surfaces share the agent's logic: the **Person**
view shows a person's *topics* + *co-speakers* ("in the same conversation"), the **Topic**
view shows *related topics*, and `who-said` / `cross-show` ground the Topic view. One layer,
two front-ends — an agent over MCP and a human in the viewer see the same connectivity.

## Notes

- **Speaker attribution is diarization-gated.** If a corpus shows `SPEAKER_03`-style voices,
  diarization ran but speaker→name attribution didn't land — exploration still works, but by
  anonymous speaker id. A *structural* lens (`/relational/topics`, `topics_of`) and a
  *grounded* lens (`/cil/persons/{id}/topics`, quote-backed) coexist; pick by intent.
- **Recurring network-feed hosts are reconciled across a show (#1056).** A network-authored
  feed (author = the company, not a person) often leaves the host unnamed in some episodes.
  The exploration/relational surfaces build the graph with feed-anchored host reconciliation:
  when a show has exactly one recurring named host, its unnamed `SPEAKER_03` voices in sibling
  episodes are folded into that person — so `entity_neighborhood` / `co_speakers` connect the
  whole show, not just the named episodes. Ambiguous shows (co-hosts) keep the voice unnamed
  but surface an honest note (`recurring host of <show> — not auto-named`) instead of a bare
  `SPEAKER_03`. It's deterministic and conservative (host role only, ≥2-episode recurrence,
  feed-exclusive voices) — a wrong name is worse than none.
- The server is **read-only** and binds to one corpus; point `--corpus` at the built corpus
  you want to explore.
