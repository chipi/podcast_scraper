# RFC-112: Remote MCP Transport + OAuth Authorization Server

- **Status**: Draft
- **Authors**: Marko, Claude (Opus 4.8), advisor (Fable 5)
- **Stakeholders**: Core team, platform users (bring-your-own-agent), Infra/Security
- **Related PRDs**:
  - `docs/prd/PRD-034-generic-mcp-server.md` (the MCP product spec)
  - `docs/prd/PRD-041-consolidation.md` (FR6.4 — BYO-agent north-star over the personal corpus)
- **Related RFCs**:
  - `docs/rfc/RFC-095-generic-mcp-server.md` (the shipped stdio MCP server; this resolves its OQ-1)
  - `docs/rfc/RFC-098-learning-platform-foundation.md` (per-user identity + session auth reused here)
  - `docs/rfc/RFC-114-personal-corpus.md` (its personal-scope toggle depends on RFC-114 Phase 1)
- **Related docs**: `docs/architecture/THREAT_MODEL.md` (expanded here — a new public ingress), ADR-114 (shared edge)
- **Decision provenance**: auth mechanism decided by client-capability research (2026-08-05) — see §2.

## Abstract

RFC-095 shipped a **stdio** MCP server: 38 tools over the corpus (search, graph, insights, catalog),
run as a local subprocess of the agent client. That is single-user, local-only, no network. This RFC
adds the deferred **remote transport** (RFC-095 OQ-1) so the platform hosts **one** MCP server on
production that any entitled platform user reaches with **their own** cloud agent — bring-your-own-
model, D6-safe (the model is the user's, never ours). The **same server** supports both transports
(stdio for local dev, Streamable HTTP for prod). Because claude.ai custom connectors require
**per-user OAuth** and the MCP spec makes **OAuth 2.1 + PKCE** the standard for *public* remote
servers, v1 authenticates via an **OAuth 2.1 authorization server we host**; a static
**personal-access token (PAT)** is a secondary convenience for CLI/dev clients (Claude Code, Cursor,
API). Access is gated by a new orthogonal **`mcp_access` entitlement**. v1 serves the **shared
corpus**; a per-user personal scope (RFC-114) is designed-for as a toggle.

## Problem Statement

The north-star (PRD-041 FR6.4) is "ask *your* Claude about everything you've listened to." The tools
exist; the transport + a real per-user auth surface do not. stdio can't be centrally hosted for
remote users. The concrete ask: **claude.ai works from day one** — a user connects their own
claude.ai account (or Claude Code / Cursor) to the platform's MCP server and queries the tools as
themselves. Anthropic's connector docs are explicit: *"Request headers suit services where everyone
shares one credential… If each person needs to sign in with their own account, use OAuth instead."*
Our case is per-user sign-in to private-per-user data → OAuth. This RFC therefore builds the auth
surface (an OAuth authorization server), the transport, the access gate, and the security posture for
the platform's first public authenticated machine-to-machine ingress.

## Goals

1. **One server, two transports.** stdio (local) + remote Streamable HTTP (prod), same tool set,
   launch-flag selected. No fork of the tool layer.
2. **OAuth 2.1 authorization server (primary).** claude.ai / Claude Code / Cursor connect via the
   standard MCP OAuth flow (PKCE, Protected Resource Metadata, token validation), authenticating as a
   specific platform user.
3. **PAT (secondary).** A static bearer token for CLI/dev/service use where OAuth is overkill.
4. **`mcp_access` entitlement.** An orthogonal grant (not a role rank) controlling who may connect +
   who sees the connection UI. Admin-granted.
5. **Connection management in both surfaces** — player Profile + operator viewer.
6. **Shared-corpus scope v1**, interface shaped so RFC-114's per-user personal scope drops in as a
   toggle without a contract change.
7. **D6 preserved** — no LLM on our side; we serve tools + data.
8. **Security done properly** — threat-model expansion, tenancy/session isolation, rate limiting.

## Constraints & Assumptions

- **Transport-agnostic tools (RFC-095).** Remote is an adapter + auth, not a tool rewrite.
- **We already have OAuth *client* pieces** (Google sign-in, `app_oauth.py`) + sessions — but being an
  OAuth *authorization server* is **new** work; that scope is accepted (operator-confirmed 2026-08-05).
- **v1 shared corpus.** Tools resolve one corpus (RFC-095 OQ-3). The auth context still carries the
  user id so RFC-114's scope toggle attaches later.
- **Public ingress** — reachable by external agents; behind the shared edge with TLS + rate limiting.

## Design & Implementation

### 1. Transport selection (one server)

`run_server(transport, host, port, corpus_dir)` (the `podcast_obs` MCP already has this shape:
`stdio | sse | streamable-http`) is adopted for the corpus MCP. CLI:
`podcast mcp --transport stdio` (local, auth-free, local trust) or `--transport http …` (prod, auth
required). The tool registry is built once and served over whichever transport.

### 2. Auth (OAuth 2.1 primary; PAT secondary)

**Why OAuth is primary (research-decided 2026-08-05).** claude.ai per-user connectors require OAuth;
the MCP spec (Mar/Nov 2025) mandates OAuth 2.1 + PKCE for public remote servers. A shared static
header is documented for *shared-credential* tools only. To meet "claude.ai from day one" we host an
**OAuth 2.1 authorization server**:

- **Endpoints**: `/.well-known/oauth-protected-resource` + `/.well-known/oauth-authorization-server`
  (metadata discovery), `/mcp/oauth/authorize` (with **PKCE**), `/mcp/oauth/token`, and **Dynamic
  Client Registration** (`/mcp/oauth/register`) so claude.ai can self-register without manual client
  provisioning.
- **The human leg reuses the platform session**: `/authorize` requires a logged-in platform session
  (existing Google sign-in); it renders a **consent screen** ("Allow <agent> to access your
  closelistening corpus?") and, on approval, issues an authorization code → access token bound to that
  `user_id`.
- **Access tokens**: short-lived, audience-bound (`aud` = our MCP resource URL), scope-carrying
  (`mcp:read` v1; room for `mcp:export` etc.), validated on every tool call. Refresh tokens rotate.
- **The user must hold `mcp_access` (§3)** or the consent/token step is refused.

**PAT (secondary).** For Claude Code / Cursor / API / local automation: a user with `mcp_access`
generates a PAT (`clp_mcp_…`, shown once, stored **SHA-256-hashed**, revocable, labelled, last-used
tracked). Sent as `Authorization: Bearer clp_mcp_…`. NOT the claude.ai per-user path (that's OAuth);
this is the shared-credential / power-user convenience the docs bless for non-per-user clients.

### 3. The `mcp_access` entitlement

- A boolean grant on the user profile, **orthogonal** to `listener < creator < admin` (`app_roles`)
  — a listener may have it, an admin may not. Admin-granted via user-management. Modelling it as a
  4th rank would poison every totally-ordered "at least creator" check.
- Gates: (a) the OAuth consent + token issuance, (b) PAT authentication, (c) visibility of the
  connection-management UI.
- **Data-visibility note (explicit, not by omission):** `mcp_access` grants the **full shared read
  surface** — all 38 tools, including viewer-grade perspectives (`ego_network`, `entity_dossier`,
  `corpus_enrichment_signals`, …) that are creator-gated in the *UI* — regardless of the holder's
  role. This is acceptable because it is all shared-corpus read data the player already derives; it is
  called out here so the policy is a decision, not an accident.

### 4. Runtime topology + auth plumbing (the real work, not detail)

- **Auth reaches the tools** via an ASGI **auth middleware** on the HTTP transport → resolves the
  bearer (OAuth access token *or* PAT) → sets a **contextvar** `current_user` → a thin tool-entry
  guard reads it. The shipped tool registry (plain functions over `CorpusContext`) is unchanged; the
  context is injected per request.
- **Where identity lives**: users, hashed PATs, OAuth grants, and `mcp_access` are in the FastAPI
  app's per-user store (RFC-098). **Decision**: the MCP HTTP server **calls a small internal
  verification endpoint on the app** (`POST /internal/mcp/verify` — token/hash → user_id + entitlement
  + scope) rather than mounting the file store directly, so the two processes stay decoupled and the
  store stays app-owned. (Tailnet-only internal call, same pattern as RFC-110's outbox seam.)
- **Token lookup is O(1)**: a `mcp_tokens_index` (SHA-256(token) → user_id) maintained by the token
  CRUD, not an O(users) scan per connect.
- **Deployment**: its own vhost behind the shared Caddy edge (ADR-114), TLS-terminated, public.

### 5. Connection management UI (both surfaces)

- **Player** (`web/learning-player`): a "Connected agents" section in Profile (beside notifications),
  visible only with `mcp_access`: the connector URL (for claude.ai "add custom connector"), a
  one-click OAuth test, and PAT generate/list/revoke (copy-once) for CLI clients.
- **Operator viewer** (`web/gi-kg-viewer`): the same in the shell Configuration area.

### 6. Corpus scope (v1 shared; toggle-ready)

- v1: every entitled connection queries the **shared corpus**. Per-user auth in v1 buys **gating,
  attribution, revocation, per-user rate-limit identity, and consent** — *not* cross-user
  confidentiality, because there is no per-user data yet; that's honest, and the plumbing is exactly
  what RFC-114's toggle needs.
- The token/grant carries `user_id`; RFC-114 Phase 1 introduces a `scope` (`shared | mine`) that
  filters tools to the user's `experienced` corpus without changing the transport/auth contract.

### 7. Security posture (THREAT_MODEL expansion — implementation gate)

`THREAT_MODEL.md` gains an MCP entry covering: OAuth code-interception (PKCE mandatory), token theft
(short-lived + rotating refresh + revocable; PATs hashed + last-used), **Streamable-HTTP
`Mcp-Session-Id` bound to the authenticating principal** (session-hijack), **Origin validation**
(MCP-spec requirement; DNS-rebinding class), cross-user access (every call scoped to the token's user;
vacuous in shared v1, load-bearing under RFC-114 — tests written against the future scope), DoS
(app-level per-principal rate limits — the Caddy edge is IP-level only), TLS-only, and a named
out-of-scope: transcript content flowing into the user's agent is third-party text (prompt-injection
belongs to the user's agent). Audit connect/auth-fail via `app_audit`.

## Key Decisions

1. **Same server, transport flag** — not a second codebase.
2. **OAuth 2.1 (PKCE + DCR) is primary v1; PAT is secondary** — required to support claude.ai per-user
   from day one; PAT serves CLI/dev. (Research-decided; supersedes the earlier PAT-first draft.)
3. **`mcp_access` as an orthogonal entitlement**, not a role rank.
4. **MCP server verifies via an internal app endpoint**, store stays app-owned.
5. **Shared corpus v1**, scope toggle deferred to RFC-114 Phase 1.

## Alternatives Considered

- **PAT-first, OAuth later (the original draft).** Rejected: claude.ai per-user needs OAuth, so
  PAT-first would mean building connection UI twice and not meeting "claude.ai from day one."
- **Static shared request header only.** Works on claude.ai but as one shared credential — no per-user
  identity/audit/revocation. Rejected as primary; survives as the PAT/dev path.
- **Per-user MCP process behind a gateway.** Process-per-user is operationally heavy; one multi-tenant
  HTTP server with per-request auth context is simpler.
- **Proxy the HTTP API instead of MCP.** Rejected (RFC-095 chose library-wrap; loses tool ergonomics).

## Testing Strategy

- Unit: OAuth code/PKCE/token issuance; token audience+scope validation; PAT hash/lookup/revoke;
  `mcp_access` gate; contextvar propagation into a tool call; stdio stays auth-free.
- Integration: full OAuth authorize→token→tool-call happy path; invalid/expired/revoked → 401;
  ungranted user → refused at consent; rate-limit trips; Origin/`Mcp-Session-Id` binding enforced.
  Cross-user isolation tests written against RFC-114's future personal scope (documented as such).
- Contract: identical tool schemas across stdio + HTTP (one registry).

## Rollout & Monitoring

- Ship stdio↔HTTP behind a flag; enable `mcp_access` for a small allowlist; dogfood the claude.ai
  connector end-to-end before wider grant.
- Monitor: OAuth grant/refresh rates, auth-fail rate, per-principal tool latency + rate-limit hits.
- Rollback: disable the HTTP transport (stdio unaffected); revoke tokens/ grants.

## Open Questions

- OAuth access-token TTL + refresh rotation window.
- Do we also support legacy HTTP+SSE for older clients, or Streamable-HTTP only?
- Rate-limit tiers per entitlement.
- Consent-screen scope granularity once RFC-114 scopes exist (`mcp:read:shared` vs `mcp:read:mine`).

## References

- RFC-095 (stdio MCP, OQ-1), PRD-041 FR6.4, RFC-098 (identity), RFC-114 (personal corpus — Phase 1
  dependency), `THREAT_MODEL.md`, ADR-114 (shared edge). Client-capability research (2026-08-05):
  Anthropic custom-connector docs (per-user → OAuth), MCP spec OAuth 2.1 + PKCE for public servers.
