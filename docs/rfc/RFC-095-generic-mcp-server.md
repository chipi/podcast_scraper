# RFC-095: Generic MCP Server — Platform Capabilities as Agent Tools

- **Status**: Implemented (v1 — stdio); HTTP/SSE transport + RFC-093 tool deferred
- **Updated**: 2026-06-20 (reconciled with shipped code; search layer is now LanceDB-first)
- **v2 cross-reference (RFC-097, 2026-06-20)**: prerequisite — stable CIL IDs (RFC-072 + RFC-097) needed for stable tool schemas. MCP server design itself unchanged by v2; HTTP/SSE transport, MCP resources, QueryEnricher tool remain open. See [RFC-097](RFC-097-unified-kg-gi-ontology-v2.md).
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

## Implementation status & handoff (2026-06-20)

**v1 is shipped.** The generic MCP server exists, is wired into the CLI, and has
unit tests. This section is the ground-truth map + the starting point for the
next agent; the design sections below are preserved for rationale but their
"future tense" should be read as **done** unless called out as deferred.

**Where the code lives**

- `src/podcast_scraper/mcp/server.py` — `FastMCP("podcast-scraper")` factory +
  tool registration; `run_stdio(corpus_dir)` runs the stdio server.
- `src/podcast_scraper/mcp/cli_handlers.py` — `parse_mcp_argv` / `run_mcp`;
  dispatched from `src/podcast_scraper/cli.py` on `podcast mcp`.
- `src/podcast_scraper/mcp/context.py` — `CorpusContext` (single resolved corpus
  dir = read boundary; OQ-3 decided: single-context server).
- `src/podcast_scraper/mcp/tools/` — `search.py` (1), `resolve.py` (1),
  `relational.py` (7), `cil.py` (3), `catalog.py` (4). Plain functions;
  `server.py` wraps them as FastMCP tools (**16 tools total**): `resolve_entity`,
  `search_corpus`, `person_positions`, `who_said_about_topic`,
  `cross_show_synthesis`, `insights_about_entity`, `topic_entities`,
  `related_insights`, `show_episodes`, `person_profile`, `topic_timeline`,
  `position_arc`, `list_feeds`, `list_episodes`, `episode_detail`, `top_people`.
- Tests: `tests/unit/mcp/` — `test_protocol.py` (construct + list + invoke via
  in-memory transport), `test_relational_tools.py`, `test_cil_tools.py`,
  `test_catalog_tools.py`, `test_server.py`, `test_tools.py`.

**Shared capability core (the §3 refactor — DONE).** Both the HTTP routes and the
MCP tools call the same functions; nothing is duplicated:

- `search/capability.py:structured_corpus_search(...)` — the shared search entry
  (assembles hybrid retrieval + lifting). HTTP `routes/search.py` and the
  `search_corpus` tool both call it. (`run_corpus_search` in
  `search/corpus_search.py` is the layer beneath it.)
- `search/relational_capability.py:rerank_relational_insights(...)` — the lifted
  hybrid re-rank shared by `routes/relational.py` and the relational tools.
- `search/corpus_graph.py:get_corpus_graph(root, derive_speaker_links=True)` —
  process-cached cross-layer graph (RFC-094).
- `identity/` resolver behind `resolve_entity`.

**Resolved open questions** (see updated Open Questions): **OQ-3** single-corpus
context — decided (`CorpusContext`). **OQ-4** SDK — decided: the official
**`mcp`** SDK (`mcp>=1.2.0,<2.0.0`, `[dev]` extra), using its bundled `FastMCP`.

**Search-layer note (the recent change this doc was stale on).** Retrieval is now
**LanceDB-first; FAISS is retired** (BM25 + dense vectors fused via RRF over a
two-tier segment/insight index at `<corpus>/search/lance_index/`). The MCP search
tool inherits this for free because it calls `structured_corpus_search`.

**What's NOT done (next-agent backlog)**

- **OQ-1 — HTTP/SSE transport + auth** (remote / prod VPS). Still stdio-only.
  The tool layer is transport-agnostic, so this is an adapter in `server.py`
  plus auth, not a rewrite.
- **OQ-2 — MCP resources** (uri-readable artifacts) in addition to tools. Not
  started.
- **RFC-093 `corpus_briefing_pack` tool** — not yet registered (the §7 seam).
  Tracked separately (#861 / RFC-093); it plugs in as one more tool.
- **`corpus_digest` tool** — listed in the §5 catalogue but **not shipped**
  (catalog tools shipped: `list_feeds`, `list_episodes`, `episode_detail`,
  `top_people`). Easy add if wanted (wrap the existing digest capability).

---

## Abstract

Expose the platform's existing **read** capabilities — hybrid search (RFC-090), the relational-query
layer (RFC-094), CIL queries, and the catalog — as a **generic MCP server**: a set of composable,
typed, provenance-bearing tools that any MCP-aware agent can call. The server **wraps the Python
library directly** (no running HTTP server required); a corpus directory is its read context. It is
**additive** (the MCP SDK rides in the `[dev]` extra + a `podcast mcp` entry point), changes no retrieval,
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

- `search_corpus` → **`search/capability.py:structured_corpus_search(...)`** — the
  shared search entry that assembles hybrid retrieval + lifting (it calls
  `search/corpus_search.py:run_corpus_search`, with `router.classify_query` /
  tier mapping underneath). The same function backs HTTP `routes/search.py`.
  Retrieval is **LanceDB-first (FAISS retired)**: BM25 + dense vectors fused via
  RRF over a two-tier segment/insight index.
- relational tools → `search/relational_queries.py` over
  `search/corpus_graph.py:get_corpus_graph(corpus_dir, derive_speaker_links=True)`
  (the RFC-094 layer), with the hybrid re-rank lifted to
  **`search/relational_capability.py:rerank_relational_insights(...)`** so both the
  route and the tool reuse it.
- `resolve_entity` → the RFC-072 entity resolver (`identity/`).
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

### 3. Capability core extraction (small, enabling refactor) — ✅ DONE

Route-internal glue was lifted into shared functions that **both** the route and
the MCP tool call: `search/capability.py:structured_corpus_search` (search +
lifting) and `search/relational_capability.py:rerank_relational_insights`
(relational hybrid re-rank). Behavior-preserving; covered by existing route tests
plus direct tests on the lifted functions.

### 4. Packaging & entry point

- The MCP Python SDK (`mcp`, which bundles FastMCP) ships in the **`[dev]`** extra — the MCP
  server is a core dev/server capability, not a separate extra. The retrieval tools also need
  **`[search]`** (ML deps) at runtime, same as the viewer.
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

1. ✅ **Core refactor + server skeleton + `resolve_entity` + `search_corpus`**.
2. ✅ **Relational tools** (seven RFC-094 traversals).
3. ✅ **Catalog + CIL tools.**
4. ⏳ **(Deferred)** HTTP/SSE transport + auth; **RFC-093 briefing-pack tool**.

## Open questions

- **OQ-1** HTTP/SSE transport in this epic or a follow-up? — **deferred** (still
  stdio-only; tool layer is transport-agnostic, so it's an adapter + auth).
- **OQ-2** Expose artifacts as MCP **resources** (uri-readable) in addition to
  tools? — **deferred** (tools-only v1).
- **OQ-3** Single-corpus server vs `corpus` arg per call. — **resolved:
  single-context** (`mcp/context.py:CorpusContext`; one server = one corpus).
- **OQ-4** MCP SDK choice (`mcp` official vs `fastmcp`) — **resolved: official
  `mcp`** SDK (`mcp>=1.2.0,<2.0.0`, `[dev]` extra), using its bundled `FastMCP`.

## Migration

Purely additive: a new optional extra, a new `mcp/` package, a new CLI entry, and a small
behavior-preserving lift of route-internal glue into shared functions. No changes to retrieval,
indexing, edges, the HTTP API, or the viewer.
