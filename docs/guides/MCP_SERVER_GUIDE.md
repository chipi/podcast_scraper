# MCP server — tool reference & user guide

The **generic MCP server** (RFC-095) exposes the platform's read capabilities as composable,
provenance-bearing tools any MCP-aware agent can call — Claude Desktop, Claude Code, Cursor,
autoresearch. It wraps the Python library directly (no HTTP server needed); a corpus directory
is its read context. This guide is the single place a **human or an agent** can learn what
tools exist, how to chain them, how to configure it in Claude, and how to maintain it.

- **Design / rationale:** [RFC-095](../rfc/RFC-095-generic-mcp-server.md) · **spec:**
  [PRD-034](../prd/PRD-034-generic-mcp-server.md) · **e2e testing:**
  `docs/wip/MCP-E2E-GUIDE.md`.
- **38 tools**, stdio transport, read-only. Every tool returns a uniform envelope:
  `{ok, data, note}` (`ok=False` on a clean error — never a crash; `note` says *why* a result
  is empty so an agent never confuses "no data" with "feature off").

## Run it

```sh
# against any corpus dir (one with metadata + .gi.json / .kg.json + a search/ index)
python -m podcast_scraper.cli mcp --corpus /path/to/corpus
```

Needs the `.[dev,search]` extras (the MCP SDK rides in `[dev]`). It waits on stdio.

### Configure in Claude

**Claude Code** (this repo):

```sh
claude mcp add podcast-corpus -- \
  python -m podcast_scraper.cli mcp --corpus "$PWD/tests/fixtures/app-validation-corpus/v3"
```

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "podcast-corpus": {
      "command": "/ABS/PATH/.venv/bin/python",
      "args": ["-m", "podcast_scraper.cli", "mcp", "--corpus", "/ABS/PATH/to/corpus"]
    }
  }
}
```

## The core idea: resolve → pivot → chain

Two rules make the whole surface fluid:

1. **Resolve names to ids first.** Most tools take canonical ids (`person:…`, `topic:…`,
   `org:…`, `podcast:…`) — call `resolve_entity("Sam Altman")` to get one, or take an id from
   a `search_corpus` hit / any tool's output.
2. **Every output carries the ids that are other tools' inputs** (referential parity). A
   `search_corpus` hit carries a **`pivot`** block `{id, kind, expand_with}` naming the id and
   *which tools consume it*. So you chain across surfaces without guessing:
   `search → insight_detail → entity_neighborhood → compare_subjects`.

The **`insight_detail`** tool is the bridge: it turns a search insight-hit's `pivot.id` into
that insight's topics + mentioned entities (each id-bearing), so search results flow into the
graph.

## Tool reference (by family)

### Entry

| tool | use |
| --- | --- |
| `resolve_entity(name, kind?)` | name → canonical id. Call FIRST on a freeform name. |
| `search_corpus(query, tier?, speaker?, topic?, episode_id?, grounded_only?, top_k?)` | hybrid two-tier search; each hit carries a `pivot` handle. `tier`: `insight`\|`segment`\|`both`. |
| `corpus_briefing_pack(query, …, max_tokens?)` | one assembled, LITM-ordered brief (RFC-093) instead of raw hits. |

### Momentum

| tool | use |
| --- | --- |
| `corpus_trending(kind?, limit?)` | what's rising corpus-wide (EWMA velocity); each entity's `entity_id` pivots into the graph. |
| `topic_perspective_leaders(limit?)` | topics by distinct-speaker count — the most-debated nodes (centrality proxy). |

### Relational (canonical ids in)

| tool | use |
| --- | --- |
| `person_positions(person_id)` | what a person stated. |
| `insights_about_entity(entity_id)` | what insights say *about* a person/org. |
| `who_said_about_topic(topic_id)` | insights grouped by speaker. |
| `cross_show_synthesis(topic_id)` | top insight per distinct show — the corpus differentiator. |
| `topic_entities(topic_id)` | entities a topic's insights mention. |
| `related_insights(insight_id)` | sibling insights. |
| `insight_detail(insight_id)` | **the pivot bridge**: an insight's text + quotes + topics + entities. |
| `show_episodes(podcast_id)` | a show's episodes. |

### CIL intelligence

| tool | use |
| --- | --- |
| `person_profile(person_id)` | grounded insights across episodes. |
| `topic_timeline(topic_id)` | insights about a topic over time. |
| `position_arc(person_id, topic_id)` | how a person's position evolves. |
| `topic_conversation_arc(topic_id, insight_types?)` | weekly volume + sentiment arc. |

### GI / grounded insight

| tool | use |
| --- | --- |
| `explore_insights(topic?, speaker?, grounded_only?, min_confidence?, sort_by?, limit?)` | faceted cross-episode discovery. |
| `episode_insights(metadata_path, limit?)` | salience-ranked insights for one episode, with quotes. |
| `compare_subjects(subject_a, subject_b, q?, insight_types?)` | two-subject compare: a briefing pack per side + a deterministic judge summary. Only carrier of the `insight_type` filter. |

### Search operators

| tool | use |
| --- | --- |
| `cluster_search(query, top_k?)` | search, then group hits by topic/theme cluster. |
| `consensus_search(query, top_k?, max_pairs?)` | search, then cross-speaker consensus pairs among the surfaced topics. |

### Enrichment & speaker

| tool | use |
| --- | --- |
| `corpus_enrichment_signals()` | corpus-scope RFC-088 envelopes (similarity, consensus, grounding_rate, …). |
| `episode_enrichment_signals(metadata_path)` | per-episode signals (sentiment, density, co-occurrence). |
| `episode_speaker_roster(metadata_path)` | diarized talk-share / roster (who spoke, %, host/guest) — no HTTP route; MCP-only. |

### Connectivity / graph

| tool | use |
| --- | --- |
| `entity_neighborhood(entity_id, k?)` | everything connected to an entity in one call — the exploration keystone. |
| `person_topics(person_id)` | topics a person engages. |
| `co_occurring_entities(entity_id)` | who's discussed alongside an entity. |
| `bridge(entity_a, entity_b)` | how two entities relate (shared topics + co-occur). |
| `related_topics(topic_id)` | co-occurring topics. |
| `ego_network(entity_id, max_hops?, k?)` | multi-hop KG-proximity neighborhood (variable depth). |
| `topic_clusters(topic_id)` | semantic + theme cluster siblings. |

### Catalog / navigation

| tool | use |
| --- | --- |
| `list_feeds()` · `list_episodes(feed?, since?, limit?)` · `episode_detail(metadata_path)` · `top_people(limit?)` | browse the corpus. |

### Composites (multipliers — one call, many surfaces)

| tool | use |
| --- | --- |
| `entity_dossier(entity_id, k?)` | the full person/topic-page fan-out in one call (profile + positions/timeline + neighborhood + arc + clusters). |
| `episode_digest(metadata_path, insight_limit?)` | one call: detail + insights + enrichment signals + speaker roster. |

## Worked example — the golden cross-surface case

> *"What's the most contested topic, who's on each side, how has it evolved, and how do the
> two loudest voices differ?"*

```text
topic_perspective_leaders()          → topic:ai-development           (centrality)
corpus_trending(kind="topic")        → is it also rising?             (momentum)
who_said_about_topic(topic_id)       → the sides, grouped by speaker  (relational)
topic_conversation_arc(topic_id)     → volume + sentiment over time   (temporal)
search_corpus(q, topic=topic_id)     → grounded hits, each w/ pivot   (search)
insight_detail(hit.pivot.id)         → that insight's entities/topics (the bridge)
entity_neighborhood(entity_id)       → expand a key voice             (graph)
compare_subjects(personA, personB)   → contrast the two voices        (compare)
```

Or the one-call shortcuts: `entity_dossier(topic_id)` / `episode_digest(metadata_path)`.

## Maintenance (for the next agent)

- **Tools are plain functions** in `src/podcast_scraper/mcp/tools/*.py` (search, resolve,
  relational, cil, gi, operators, enrichment, connectivity, catalog, composites, trending,
  briefing_pack). They wrap the *same* capability functions the HTTP routes call — no
  duplicated logic. Adding a tool = add the function + register it.
- **Registration** is split into per-family `_register_*(server, ctx)` helpers in
  `src/podcast_scraper/mcp/server.py`, called from `build_server`. Add your tool's
  `@server.tool()` wrapper to the right registrar (keeps `build_server` under the complexity
  limit).
- **The envelope** is applied by `_enveloped` / `_safe` in `server.py` — return a plain dict
  (or an `{ok, …}` dict for connectivity-style tools); the wrapper normalizes it.
- **Referential parity is the contract:** a new tool's output should carry the canonical ids
  (`entity_id` / `topic_id` / `insight_id` / `metadata_path`) that other tools consume, so it
  keeps the chain intact.
- **Tests:** `tests/unit/mcp/` (per-tool + protocol dispatch + registered-tool list) and
  `tests/integration/test_mcp_pivot_chain_e2e.py` (the golden chain + episode-scoped + search
  operators/compare against the committed synthetic corpus, index built at setup). Update the
  registered-tools set in `test_server.py` when you add a tool.
