# ADR-153: MCP stays read-only over HTTP — `mcp:write` is enforced but not minted

- **Status**: Accepted
- **Date**: 2026-09-03
- **Authors**: Marko Dragoljevic
- **Related RFCs**: [RFC-095](../rfc/RFC-095-generic-mcp-server.md) §6 (read-only),
  [RFC-112](../rfc/RFC-112-remote-mcp-transport-and-auth.md) §4 (verify endpoint, scope on the
  wire)
- **Issues**: #1916

## Context & Problem Statement

Scope was plumbed end to end and then thrown away.

The app's internal verify endpoint has always returned the token's granted `scope`
(`routes/internal_mcp.py`), and the MCP resource server never read it (`mcp/auth.py`). Every tool
therefore ran on the `mcp_access` entitlement alone — including two that mutate the corpus:

- `reenrich` — enqueues a corpus-level enrichment pass
- `reindex` — enqueues a vector reindex, optionally a full rebuild

So **any user holding `mcp_access` could have their agent trigger a corpus-wide reindex with a
read-only token.** No scope was ever checked, and the authorization server only mints one scope
(`mcp:read`), so nothing on the wire distinguished a reader from a writer.

Closing the hole raises an immediate second question: having introduced `mcp:write` to gate those
two tools, do we then **mint** it, so the capability keeps working over HTTP?

## Decision

**Enforce `mcp:write`, and do NOT add it to `_SUPPORTED_SCOPES`.**

Corpus writes over the HTTP transport are unreachable. `require_scope(SCOPE_WRITE)` refuses them
for every token the authorization server can currently issue, and `_SUPPORTED_SCOPES` stays
`{"mcp:read"}`.

stdio is unaffected. It has no transport auth and no token, is local-trust by design — the same
reasoning that already lets it run unauthenticated — and `require_scope` passes when there is no
HTTP auth context at all. Operators keep the capability locally; remote agents do not have it.

## Rationale

1. **Read-only is a recorded design intent, and minting a write scope would silently reverse it.**
   RFC-095 §6 states read-only is a PRD-034 non-goal to violate. RFC-112's anticipated next split
   is `shared | mine` — both reads. Granting write is net-new scope beyond both RFCs; it should
   arrive as its own decision with its own reasoning, not as a side effect of fixing an
   authorization bug.
2. **The blast radius is the corpus, not one user's data.** `reindex(rebuild=True)` drops and
   rebuilds the vector index everyone reads. A remote agent acting on a misread instruction — or
   on prompt injection, which RFC-095 disclaims for reads — should not be able to do that.
3. **Nobody has asked for it.** #1916's own phasing says to ship per-user READS and "see if anyone
   asks for the write". Removing an unrequested capability that was only reachable through a bug
   costs nothing observable.
4. **Failing closed is recoverable; failing open is not.** If someone does need remote corpus
   writes, granting the scope later is a small change. An unnoticed reindex triggered by an agent
   is not undoable on the same timescale.

## Consequences

**Positive**

- The corpus cannot be mutated by a remote agent, whatever its token.
- Scope is now actually enforced, so future scopes (`mcp:read:mine`, `mcp:export`) have a working
  mechanism rather than a decorative field.
- The distinction that carries the safety is explicit and tested: unset scopes = stdio (local
  trust), empty scopes = an HTTP token that granted nothing (refused).

**Negative — stated plainly**

- **This removes a capability that worked.** Anyone driving `reenrich` or `reindex` through the
  remote MCP server loses it and must use stdio or the API. It worked only because scope was
  unchecked, but "it worked yesterday" is still true for them.
- The two tools remain registered and discoverable over HTTP, and fail at call time with
  `{"ok": false, "note": "McpScopeError"}` rather than being hidden from `tools/list`. Hiding them
  per-scope is possible and was not done — an agent that can see a tool it may not call gets a
  clear refusal, which is more debuggable than a tool that silently does not exist.

## What would have to change to grant it

1. Add `mcp:write` to `_SUPPORTED_SCOPES` and to the consent screen's scope description.
2. Decide whether PATs may carry it (`app_mcp_tokens.py` has **no scope field at all** today, so
   PATs would need one — otherwise every PAT would carry every scope).
3. Answer #1916's open product question: is agent-initiated *write* a goal, or is "agents propose,
   humans commit" the intended shape?
4. Restate or mitigate the prompt-injection disclaimer, which RFC-095 made for reads only.
