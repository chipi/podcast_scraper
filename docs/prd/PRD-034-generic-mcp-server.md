# PRD-034: Generic MCP Server — Platform Capabilities as Agent Tools

- **Status**: Implemented (v1, stdio) — slices 1–3 shipped; HTTP/SSE + RFC-093 tool deferred
- **Author**: Marko
- **Created**: 2026-06-05
- **Updated**: 2026-06-20 (reconciled with shipped code)
- **Target**: v2.7+ (additive, opt-in)
- **Related PRDs**:
  - `docs/prd/PRD-031-search.md`, `docs/prd/PRD-032-hybrid-corpus-search.md` — the retrieval backend exposed here
  - `docs/prd/PRD-033-search-powered-surfaces.md` — the human-facing surfaces over the same capabilities
- **Related RFCs**:
  - `docs/rfc/RFC-095-generic-mcp-server.md` — the technical design implementing this PRD
  - `docs/rfc/RFC-094-search-powered-surfaces-query-layer.md` — the relational-query layer the MCP tools wrap
  - `docs/rfc/RFC-093-litm-context-packs.md` — the LITM **briefing-pack** tool that later plugs into this server (a *consumer*, not part of this PRD)
- **Related issues**: [#891](https://github.com/chipi/podcast_scraper/issues/891) (epic), [#892](https://github.com/chipi/podcast_scraper/issues/892) (Slice 1 — skeleton + resolve + search), [#893](https://github.com/chipi/podcast_scraper/issues/893) (Slice 2 — relational tools), [#894](https://github.com/chipi/podcast_scraper/issues/894) (Slice 3 — catalog + CIL); [#861](https://github.com/chipi/podcast_scraper/issues/861) (RFC-093 briefing pack — a consumer, not part of this PRD)

> **Scope note.** This PRD is deliberately **decoupled from RFC-093 / #861**. #861 is one
> opinionated, synthesized tool (the LITM `corpus_briefing_pack`). This PRD is the **generic
> substrate**: expose the platform's existing read capabilities as composable agent tools. The
> briefing pack becomes *one more tool* registered on this server, later, without changing this
> contract.

---

## Implementation status & handoff (2026-06-20)

**Shipped (v1, stdio).** Slices 1–3 are done; the generic MCP server runs and has
unit tests. Run it with **`podcast mcp --corpus <dir>`** (stdio). Code lives in
`src/podcast_scraper/mcp/` (see RFC-095 → *Implementation status & handoff* for
the file-by-file map and shared-callable details). **16 tools** are registered:
`resolve_entity`, `search_corpus`, the 7 relational tools, 3 CIL tools
(`person_profile`, `topic_timeline`, `position_arc`), and 4 catalog tools
(`list_feeds`, `list_episodes`, `episode_detail`, `top_people`). Tests:
`tests/unit/mcp/`.

**FR status:** FR1 (server+context), FR2 (resolve), FR3 (retrieve), FR4
(relational ×7), FR5 (catalog + CIL — minus `corpus_digest`, see below), FR6
(provenance/honesty), FR7 (typed descriptions) — **all met for v1**.

**Next-agent backlog (the worktree this is being prepped for):**

1. **HTTP/SSE transport + auth** (OQ-1) — remote/prod-VPS access; today stdio-only.
2. **MCP resources** (OQ-2) — expose artifacts as uri-readable resources.
3. **RFC-093 `corpus_briefing_pack` tool** (#861) — register the synthesized tool
   on this server (the §"RFC-093 seam").
4. **`corpus_digest` tool** — in the catalogue below but **not shipped**; wrap the
   existing digest capability if wanted.
5. **Agent-loop eval** — the success-metric held-out question set is not yet
   automated.

The capability surface, FRs, and metrics below describe the **product intent**;
where they read as future tense, treat slices 1–3 as done per the above.

---

## Summary

Podcast Scraper has built a rich intelligence layer — hybrid two-tier retrieval (RFC-090), a
relational-query layer over the typed corpus graph (RFC-094), cross-layer CIL queries, and a
canonical-identity layer (RFC-072). Today those capabilities are reachable two ways: the Python
library (operators/CLI) and the FastAPI HTTP API (the viewer). **Agentic clients have no first-class
way in.**

This PRD introduces a **generic [MCP](https://modelcontextprotocol.io) server** that exposes the
platform's **key read capabilities as composable tools**, so any MCP-aware agent (Claude Desktop,
Claude Code, Cursor, custom agents) can use a podcast corpus as a tool: resolve an entity, search
for grounded evidence, ask "who said what about X across shows," pull a person's positions, walk a
topic's timeline — and get back **structured, provenance-bearing** results it can compose.

This is the infrastructure expression of the objectivization mission (`docs/wip/VISION-search-and-intelligence.md`):
*agents query with intent, receive grounded evidence, and can trust the provenance.* Generic tools
first; synthesized intelligence (briefing packs, RFC-093) layers on top.

---

## Problem

- **No agent entry point.** The capabilities exist but only humans (viewer) and operators (CLI/library)
  can reach them. An agent that wants to reason over the corpus has to be hand-wired to the HTTP API
  per integration.
- **Capabilities are primitives, not a product, for agents.** Search, relational queries, and CIL
  are individually useful but undiscoverable and unschematized for tool-use. Agents need named tools
  with clear descriptions, typed inputs, and provenance-bearing outputs.
- **The identity gap.** Agents start from *names* ("Sam Altman", "inflation"); the relational tools
  need *canonical ids* (`person:sam-altman`, `topic:inflation`). Without a resolve step, the rich
  relational surface is unusable from natural language.

---

## Goals

1. **One MCP server** exposing the platform's key **read** capabilities as well-described, typed tools.
2. **Composable primitives**, not one mega-tool: agents combine `resolve_entity` → `search_corpus` /
   relational tools → catalog/CIL tools to answer multi-step questions.
3. **Provenance by default** — every result carries the ids, episode/show attribution, and grounding
   that make claims traceable (RFC-072 / GIL grounding invariants).
4. **Opt-in and additive** — ships behind an extra; no change to existing search, indexing, edges, or
   the HTTP API. The corpus is the read-only context.
5. **A clean seam for synthesized tools** — RFC-093's briefing pack and future objectivization
   enrichers register as tools on this server without re-plumbing.

## Non-goals

- **Not** the LITM briefing-pack synthesis — that is RFC-093 / #861 (a consumer of this server).
- **Not** a write/control surface — no pipeline triggering, feed editing, or corpus mutation tools in
  this generic server (operators use the CLI / viewer for that).
- **Not** a new retrieval or ranking change — tools wrap the *existing* functions.
- **Not** auth/multi-tenant infrastructure — single-operator, local-context first (remote transport
  is a later slice, see Open Questions).

---

## Users & use cases

- **Agent developers** wiring an assistant to a podcast corpus ("answer questions about what was said,
  by whom, across shows, with sources").
- **Operators using agent clients** (Claude Code/Desktop, Cursor) who point the MCP server at a local
  corpus and ask cross-episode questions interactively.
- **Downstream objectivization agents** (future) that detect narratives / track positions, running
  *against* this tool surface rather than re-implementing retrieval.

Representative tasks an agent can complete by composing tools:

- "What has *Jane Doe* said about *inflation*, and on which shows?" → `resolve_entity` →
  `who_said_about_topic` + `person_positions`.
- "Find the strongest cross-show take on *AI regulation*." → `resolve_entity` →
  `cross_show_synthesis`.
- "Ground this claim with a verbatim quote." → `search_corpus` (segment tier) + `episode_detail`.
- "How has *X*'s position on *Y* evolved?" → `position_arc` / `topic_timeline`.

---

## Capability surface (product-level tool catalogue)

Grouped by intent. Exact names/schemas are RFC-095's concern; this is the *what*.

| Group | Tools | Backed by |
| --- | --- | --- |
| **Resolve** | `resolve_entity` (name → canonical `person:`/`org:`/`topic:` id; the keystone for everything below) | RFC-072 identity / entity resolver |
| **Retrieve** | `search_corpus` (hybrid two-tier; returns tier, intent, grounded evidence; optional tier/grounded/feed/date filters) | RFC-090 `/api/search` path |
| **Relational** | `person_positions`, `who_said_about_topic`, `cross_show_synthesis`, `insights_about_entity`, `topic_entities`, `related_insights`, `show_episodes` | RFC-094 relational-query layer |
| **Catalog / navigate** | `list_episodes`, `episode_detail`, `top_people`, `list_feeds` (shipped); `corpus_digest` (planned, not yet shipped) | `/api/corpus/*` catalog |
| **Intelligence (CIL)** | `person_profile`, `topic_timeline`, `position_arc` | cross-layer CIL queries (RFC-072 bridge) |

All tools are **read-only** and return structured JSON with provenance (canonical ids, episode/show
attribution, grounding flags, scores).

---

## Functional requirements

- **FR1 — Server + context.** A runnable MCP server (e.g. `podcast mcp`) bound to a **corpus
  directory** as its read context; tool calls operate within that corpus. No corpus mutation.
- **FR2 — Resolve.** `resolve_entity(name, kind?)` returns ranked canonical-id candidates with display
  names, so agents can bridge from natural language to the relational/CIL tools. *(The single most
  important tool for usability.)*
- **FR3 — Retrieve.** `search_corpus` exposes hybrid two-tier retrieval with the tier indicator and
  detected intent (PRD-033 FR1.1/1.4 parity), returning grounded evidence (insight + supporting
  segment) and provenance.
- **FR4 — Relational.** The seven relational tools (above) expose RFC-094's traversals, returning
  `RelatedNode`-shaped results (id, type, text, show, episode), hybrid-re-ranked where applicable.
- **FR5 — Catalog & CIL.** Episode/feed/digest/person catalog + CIL person-profile / topic-timeline /
  position-arc tools for navigation and temporal intelligence.
- **FR6 — Provenance & honesty.** Every tool result is traceable to canonical ids + source episodes;
  tools degrade gracefully (empty result, never fabricated) when a corpus lacks the data (e.g. no
  diarization → no speaker attribution), mirroring the surfaces' honest-empty-state discipline.
- **FR7 — Discoverability.** Each tool ships a clear description + typed input/output schema so agents
  select and call tools correctly without bespoke prompting.

---

## Success metrics

- An agent client (Claude Code/Desktop) can, with **no custom glue**, complete each representative
  task above against a real corpus.
- **Resolve→relational composition works**: from a plain name, an agent reaches the correct
  `person:`/`topic:` id and gets non-empty, correctly-attributed results ≥ N% of the time on a held-out
  question set.
- **Zero regressions**: the server is additive; existing search/indexing/HTTP tests unaffected.
- **Provenance**: 100% of returned claims carry a canonical id + source episode (auditable).

---

## Orthogonal / out of scope (explicit)

- **Briefing packs (RFC-093 / #861)** — the LITM-synthesized `corpus_briefing_pack` is a *tool that
  registers on this server later*. This PRD provides the substrate; #861 is unchanged.
- **Write/control tools** — pipeline runs, feed CRUD, index rebuild stay on the CLI/viewer.
- **Remote / multi-tenant transport & auth** — local stdio context first; HTTP/SSE + auth is a later
  slice (Open Questions).
- **MCP *resources* / *prompts*** — this PRD is tools-first; resources (e.g. exposing artifacts as
  readable resources) and prompt templates are possible follow-ons, not required.

---

## Open questions

- **OQ-1 Transport.** — **deferred to a follow-up.** v1 is stdio-only; HTTP/SSE +
  auth is the top backlog item.
- **OQ-2 Resource exposure.** — **deferred (tools-only v1).** Resources are a
  possible follow-on.
- **OQ-3 Multi-corpus.** — **resolved: single-context** (one server = one corpus;
  run multiple servers for multiple corpora).
- **OQ-4 Result size / budgeting.** — **resolved: generic tools stay un-shaped**
  (caps + pagination only); token-budget shaping remains RFC-093's job.

---

## Rollout

1. ✅ **Substrate** — server skeleton + `resolve_entity` + `search_corpus`.
2. ✅ **Relational tools** — the seven RFC-094 traversals.
3. ✅ **Catalog + CIL tools** (minus `corpus_digest`).
4. ⏳ **(Later) HTTP/SSE transport + auth**, and **RFC-093 briefing-pack tool** registration.

Sequencing favors the smallest end-to-end agent loop first (resolve + search), proving the transport
and tool-description ergonomics before breadth.
