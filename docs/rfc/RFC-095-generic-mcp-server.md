# RFC-095: Generic MCP Server — Platform Capabilities as Agent Tools

- **Status**: Draft
- **Authors**: Marko
- **Stakeholders**: Core team
- **Related PRDs**:
  - `docs/prd/PRD-034-generic-mcp-server.md` — the product spec this RFC implements
- **Related RFCs**:
  - `docs/rfc/RFC-094-search-powered-surfaces-query-layer.md` — the relational-query layer (`relational_queries.py`) the tools wrap
  - `docs/rfc/RFC-090-hybrid-retrieval.md` — the `run_corpus_search` path behind `search_corpus`
  - `docs/rfc/RFC-093-litm-context-packs.md` — the `corpus_briefing_pack` tool that **registers on** this server (this RFC is the "MCP layer [that] does not exist yet" RFC-093 is gated on)
  - `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md` — canonical ids + the entity resolver behind `resolve_entity`
- **Related ADRs**:
  - `docs/adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md` — feature-flagged, additive surfaces (precedent for an opt-in server)

---

## Abstract

Expose the platform's existing **read** capabilities — hybrid search (RFC-090), the relational-query
layer (RFC-094), CIL queries, and the catalog — as a **generic MCP server**: a set of composable,
typed, provenance-bearing tools that any MCP-aware agent can call. The server **wraps the Python
library directly** (no running HTTP server required); a corpus directory is its read context. It is
**additive and opt-in** (a `[mcp]` extra + a `podcast mcp` entry point), changes no retrieval,
indexing, edge, or HTTP code, and provides the substrate on which RFC-093's briefing-pack and future
synthesized tools register.

---

## Motivation

PRD-034 makes the case: the capabilities exist but agents have no first-class entry point, and the
relational/CIL surface is unusable from natural language without a name→id resolve step. RFC-093
already records the dependency explicitly — its `corpus_briefing_pack` "requires an MCP layer that
does not exist yet." This RFC builds that layer **generically**, so the briefing pack is one consumer
rather than the reason the layer exists.

---

## Design

### 1. Transport — stdio first

v1 is a **stdio** MCP server (the dominant local-agent transport: Claude Desktop/Code, Cursor). No
network, no auth surface, runs as a child process of the agent client. HTTP/SSE (remote / prod VPS)
is a deferred slice (Open Questions OQ-1) — the tool layer is transport-agnostic, so adding it later
is a thin adapter, not a rewrite.

### 2. Architecture — library-wrap, not HTTP-proxy

The server **imports and calls the same functions** the HTTP routes do, against a resolved corpus
directory:

- `search_corpus` → `search.corpus_search.run_corpus_search(...)` (+ `router.classify_query` /
  `tier_for_doc_type` for intent/tier, mirroring `routes/search.py`).
- relational tools → `search.relational_queries.*` over a `get_corpus_graph(corpus_dir,
  derive_speaker_links=True)` (the RFC-094 layer, including the hybrid re-rank already wired at the
  route helper — extracted to a shared callable so both the route and the tool reuse it).
- `resolve_entity` → the RFC-072 entity resolver (`identity/resolver.py`).
- catalog/CIL → the existing `gi/corpus`, catalog, and `cil_queries` functions.

**Why library-wrap (not proxy the HTTP API):** no second process to run/depend on; reuses the exact,
tested functions; the process cache (`get_corpus_graph`, embedding loader) is shared in-process. The
HTTP API and the MCP server become **two thin adapters over one capability core** — which motivates a
small refactor (below).

```text
                 ┌───────────────────────────┐
   agent client ─┤  MCP server (stdio)        │
                 │   tools/  (this RFC)        │
                 └─────────────┬──────────────┘
   viewer ───────┤  FastAPI routes/ (HTTP)    │
                 └─────────────┬──────────────┘
                  shared capability core:
                  run_corpus_search · relational_queries ·
                  entity resolver · catalog · cil_queries
```

### 3. Capability core extraction (small, enabling refactor)

Some logic currently lives **inside** route handlers (e.g. the relational hybrid re-rank helper,
the search tier/intent assembly). To avoid duplicating it in tools, lift the route-internal glue into
plain functions in the `search` / capability modules that **both** the route and the MCP tool call.
This is a pure refactor (no behavior change, covered by existing route tests) and is the first slice.

### 4. Packaging & entry point

- New optional dependency group **`[mcp]`** (the MCP Python SDK, e.g. `mcp` / `fastmcp`); the core
  package and existing extras are unchanged.
- New module `src/podcast_scraper/mcp/` — `server.py` (construct + run), `tools/` (one module per
  tool group), `context.py` (corpus-dir resolution + caches).
- CLI entry **`podcast mcp --corpus <dir>`** (stdio). Documented in the
  [HTTP API Reference](../api/HTTP_API.md) sibling / a new MCP section of the Server Guide.

### 5. Tool catalogue (schemas)

Read-only. Inputs typed; outputs structured JSON with provenance. Initial set (PRD-034 capability
surface):

| Tool | Input | Output (shape) |
| --- | --- | --- |
| `resolve_entity` | `name`, `kind?` (`person`/`org`/`topic`) | ranked `[{id, kind, display_name, score}]` |
| `search_corpus` | `query`, `tier?` (`insight`/`segment`/`both`), `grounded_only?`, `feed?`, `since?`, `top_k?` | `{query_type, results:[{doc_id, source_tier, score, text, metadata, lifted?, supporting_quotes?}]}` |
| `person_positions` | `person_id`, `k?` | `[RelatedNode]` (insights stated; hybrid-re-ranked) |
| `who_said_about_topic` | `topic_id`, `k?` | `{person_id: [RelatedNode]}` |
| `cross_show_synthesis` | `topic_id`, `per_show?` | `{show_id: [RelatedNode]}` |
| `insights_about_entity` | `entity_id`, `k?` | `[RelatedNode]` |
| `topic_entities` | `topic_id`, `k?` | `[RelatedNode]` (ranked by mention frequency) |
| `related_insights` | `insight_id`, `k?` | `[RelatedNode]` |
| `show_episodes` | `podcast_id`, `k?` | `[RelatedNode]` |
| `list_episodes` / `episode_detail` / `corpus_digest` / `top_people` / `list_feeds` | catalog params | catalog JSON (reuse `/api/corpus/*` models) |
| `person_profile` / `topic_timeline` / `position_arc` | id + range | CIL JSON |

`RelatedNode` = `{id, type, text, show_id, episode_id}` (the RFC-094 projection). Tool descriptions
explicitly tell agents the **resolve-first** pattern (names → ids → relational/CIL tools).

### 6. Context, safety, honesty

- **Corpus context:** a single resolved corpus directory per server (OQ-3); all reads confined to it.
  Reuses `resolve_corpus_path_param`-style confinement so the server cannot read outside the corpus
  root (same Type-1 safety as the HTTP routes).
- **Read-only:** no tool mutates the corpus, triggers pipelines, or edits feeds (PRD-034 non-goal).
- **Honest empties:** tools return empty/partial results (never fabricated) when data is absent — e.g.
  no diarization → no speaker attribution, mirroring the surfaces.
- **Bounded output:** caps + pagination (`k`, `top_k`); no token-budget *shaping* (that is RFC-093).

### 7. The RFC-093 seam

`corpus_briefing_pack(topic|person)` registers as **one additional tool** on this server, calling
`build_briefing_pack()` (RFC-093) over the same retrieval/relational core. Nothing in this RFC depends
on it; it is the canonical example of a synthesized tool composing the primitives.

---

## Alternatives considered

- **Proxy the HTTP API** (MCP tools call `/api/*`). Rejected for v1: requires a running server, adds a
  network hop + a second failure mode, and forfeits in-process caching. (It *becomes* attractive only
  for a remote HTTP/SSE deployment, which is the deferred transport slice.)
- **One mega-tool (`ask_corpus`)** that hides the primitives. Rejected: opaque, hard for agents to
  compose or for us to evaluate; the briefing pack (RFC-093) is the place for a high-level tool, and
  even it is one tool among many.
- **Wait for RFC-093** and ship MCP only with the briefing pack. Rejected: couples a generic substrate
  to one opinionated consumer; PRD-034 explicitly decouples them.

---

## Testing strategy

- **Tool unit tests** against a synthetic corpus fixture (mirrors `tests/unit/search` /
  `tests/integration/server`): each tool returns the expected structured result; resolve→relational
  composition works; honest-empty on missing data.
- **Capability-core refactor** is covered by the existing route tests (no behavior change) plus
  direct tests on the lifted functions.
- **MCP protocol smoke test**: construct the server, list tools, invoke a tool via the SDK's in-memory
  transport, assert schema-valid output.
- No new retrieval/index tests — tools wrap existing, tested functions.

---

## Rollout & sequencing

1. **Core refactor + server skeleton + `resolve_entity` + `search_corpus`** — smallest end-to-end
   agent loop; proves transport + tool ergonomics.
2. **Relational tools** (seven RFC-094 traversals).
3. **Catalog + CIL tools.**
4. **(Deferred)** HTTP/SSE transport + auth; **RFC-093 briefing-pack tool**.

## Open questions

- **OQ-1** HTTP/SSE transport in this epic or a follow-up? (Leaning follow-up; tool layer is
  transport-agnostic.)
- **OQ-2** Expose artifacts as MCP **resources** (uri-readable) in addition to tools? (Leaning v2.)
- **OQ-3** Single-corpus server vs `corpus` arg per call. (Leaning single-context.)
- **OQ-4** MCP SDK choice (`mcp` official vs `fastmcp`) — decide at slice 1.

## Migration

Purely additive: a new optional extra, a new `mcp/` package, a new CLI entry, and a small
behavior-preserving lift of route-internal glue into shared functions. No changes to retrieval,
indexing, edges, the HTTP API, or the viewer.
