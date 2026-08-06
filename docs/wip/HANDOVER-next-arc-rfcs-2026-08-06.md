# Handover — next-arc RFCs (112 / 113 / 114)

- **Date**: 2026-08-06
- **Branch**: `feat/next-arc-rfcs` (**unpushed** — push is gated on your explicit "push")
- **Author**: Claude (Opus 4.8), review by advisor (Fable 5)

## What shipped this session

Three RFCs, all built end-to-end on the branch, plus a fable-5 review + hardening pass.

| RFC | Scope | State |
| --- | --- | --- |
| **RFC-112** (#1471) remote MCP + OAuth | slices 1–4 (entitlement + PAT, HTTP transport + auth middleware, OAuth 2.1 AS, "Connected agents" UI on player + operator viewer) + discovery chain | **built + hardened** |
| **RFC-113** (#1472) graph-aware Obsidian export | v1 vault + real incremental + player button | **built + hardened** |
| **RFC-114** (#1470) personal corpus | Phase 1 faceted membership + revision log; Phase 2 strength model | **built + hardened** |

Each RFC now carries an "Implementation status (2026-08-06)" block with the detail.

## Gates (all green)

- **Python** (touched files): `black`/`isort`/`flake8`/`mypy` clean; server+mcp suite `1487 passed, 7 skipped` (skips are docker-image-gated) — includes the one RFC-114 latent test I fixed at cause (multiuser isolation now uses a *capture*, not a favorite, to put an episode in `experienced`).
- **Player**: `test:coverage` exit 0 (`388 passed`), `build` (vue-tsc + vite) exit 0.
- **Viewer**: `test:coverage` exit 0 (`2739 passed`), `build` exit 0.
- **Docs**: `make docs` (mkdocs strict) exit 0.

I did **not** run `make ci-fast`/`ci-ui-fast`/stack-test (Docker) — those are the final pre-push gates; run them before pushing.

## Review (fable-5): findings implemented

No **critical**. Implemented: **H1** verify off the event loop (DoS), **H3** consent screen discloses redirect origin + Deny + clickjacking headers, **H4** DCR client/redirect_uri caps, **M1** scope allow-list + scope in verify response, **M2** expired-grant purge, **M4** Origin allow-list (DNS-rebind), **M5** reconcile/export compute inside the lock, **M6** `since=0` truncation + docstring, **M7** YAML escaping + traversal-safe zip paths, **M8** full-export `replace_namespace`, **L1** non-ASCII PKCE fails closed, **L2** redirect_uri cap, **L4** consent CSP header. **H2** partial: nginx now proxies the root AS metadata (repo config); the Caddy side is deploy-time (below).

## Deploy plumbing — BUILT (2026-08-06), locally smoke-tested

The remote MCP surface is now wired end-to-end (no live infra was touched — configs only, validated locally):
- `mcp` service in `compose/docker-compose.player-public.yml` (same low-priv image, `podcast mcp --transport http`, corpus RO, verifies via `api`'s tailnet seam); api image bundles the MCP SDK.
- `infra/caddy/mcp.caddy` (new resource vhost, bearer-auth in-process, no coming-soon gate).
- `infra/caddy/player.caddy` exempts the OAuth AS paths from the coming-soon gate; player `nginx.conf` proxies the root AS metadata.
- `deploy-player.sh` + `deploy-player.yml` stage `INTERNAL_MCP_TOKEN` + derived issuer/resource URLs and install the `mcp` vhost **only when MCP is enabled**.
- Validated: `docker compose config` ✓, `bash -n` + shellcheck ✓, workflow YAML ✓, and a **live local run** of the exact container command → discovery `200` + token-less `POST /mcp` `401` with the `resource_metadata` pointer. Runbook: `docs/guides/PLAYER_PUBLIC_LAUNCH.md` §MCP.

**To turn it on (operator, ~3 steps):** DNS `A mcp.<domain> → VPS`; set GH secret `PLAYER_INTERNAL_MCP_TOKEN` (`openssl rand -hex 32`); run the player deploy; then flip `mcp_access` on allowed users. Unset secret = MCP stays fully inert.

## NOT done — needs your call (equal weight)
- **`aud`-binding** on tokens — advisor ruled it **defensible v1 residue** (single verify-seam consumer); trigger to add = a 2nd MCP resource. Your call if you want it now.
- **Per-principal (per-user/per-client) rate-limit** + **`app_audit`** on connect/auth-fail (RFC-112 §7) — only IP-level at the edge today. Deferred.
- **User-facing OAuth-consent revocation UI** — `revoke_consent` exists in the store; the UI lists PATs only. Admin `mcp_access` pull is the working kill-switch (immediate, connect-time).
- **RFC-114 product confirmation**: excluding whole-episode favorites from `experienced` now also lets a favorited-but-unplayed episode resurface in `auto_picks`/digest. This flows from your confirmed "favorite = save-for-later" correction — confirm it's the intended behavior.
- **`Mcp-Session-Id`↔principal binding** inside FastMCP — unexamined; per-request bearer verify mitigates.

## Docs review pass (2026-08-06)

Self-review of the arc's docs surfaced real drift; written + gated (`make docs` strict + markdownlint clean):
- **`docs/api/PLATFORM_API.md`** — added 3 route sections (personal corpus / knowledge export / MCP access) + an MCP env-var table + the internal verify seam. (This is the authoritative `/api/app` reference; it had none of the arc.)
- **`docs/rfc/RFC-095`** — OQ-1 (remote transport) marked **resolved by RFC-112**, cross-linked (3 spots).
- **`README.md`** — MCP bullet now covers both transports + OAuth; new personal-corpus/export bullet.
- **`docs/api/CLI.md`** — `podcast mcp --transport http`.
- **`docs/guides/SERVER_GUIDE.md`** — remote-transport section; fixed a stale "16 tools" → **38** (verified via `grep -c @server.tool()`).
- **`docs/guides/PLAYER_PUBLIC_LAUNCH.md`** — a "Public surfaces (vhosts)" inventory table.
- **`docs/security/THREAT_MODEL.md`** — bumped "Last reviewed" (a new public tenant is a review trigger).
- **New test** `tests/unit/mcp/test_http_app_serving.py` — closes the gap where only a manual smoke covered the deployed HTTP serving (asserts discovery 200 + token-less 401 through the real `build_http_app`).

**Deliberately NOT changed (with reason):** `CONFIGURATION.md` (pipeline-only; server env correctly lives in PLATFORM_API); `HTTP_API.md` (the *viewer/operator* API — `/api/app` doesn't belong there); `DOCKER_SERVICE_GUIDE.md` (single-container pipeline, not the player stack). PRD-034/041/046 all exist (no dangling refs). No new ADR (the RFCs are the decision record; an ADR would duplicate).

**Follow-up fixes (2026-08-06, after the review):**
- **Audit** — wired `app_audit` into the MCP surface: verify-seam denials, OAuth token issued/denied, PAT create/revoke, consent granted/revoked (T-13 residue item closed).
- **User-facing consent revocation** — `list_consents`/`revoke_client_grants` in the store; `GET`/`DELETE /api/app/mcp/connections`; "Connected apps" list + Disconnect in both the player and operator-viewer "Connected agents" UIs. A disconnect forgets consent **and drops the client's live access/refresh tokens** (dies at next tool call). T-13 residue item closed.
- **Delivery-arc docs** — added Collections + Delivery(comms/push) sections to PLATFORM_API.md (the already-merged epic #1413 routes were undocumented).

**Correction:** the earlier note that `docs/history/0002-decisions.md` is a *dangling reference* was wrong — nothing in the shipped docs links it (only the new-doc skill names the convention path). No fix needed; no decision log created (the RFCs are the decision record).

## Deferred items — now CLOSED (operator-directed)

All three T-13 residue items I'd flagged as deferred were closed on your call:
- **aud-binding** (built) — RFC 8707 audience-bound tokens; the resource server rejects a token minted for a different resource.
- **Per-principal rate-limit** (built) — `app_rate_limit` in-process window on the OAuth endpoints (DCR/IP, token/client → 429); single-worker caveat documented.
- **`Mcp-Session-Id` binding** (closed by design) — per-request bearer re-verify ⇒ no ambient session authority to bind.

Sole remaining residue: **per-user corpus scoping** (v1 is shared-corpus by design; the grant buys gating/attribution/revocation, not confidentiality — there's no per-user data yet).

## Commits (branch `feat/next-arc-rfcs`, unpushed)

```
5c2fc837 feat(mcp): close the last T-13 residue — aud-binding + per-principal rate-limit
95eef64f docs: RFC-112/113/114 arc + remote-MCP deploy + T-13; API/CLI/README drift fixes
04191219 feat(infra): remote MCP deploy plumbing so claude.ai works end-to-end
0cdaff5f feat(mcp): "Connected agents" UI — player Profile + operator viewer
b30b6623 feat(mcp): RFC-112 slice 4 + review-hardening + audit/consent-revocation
```

(Earlier on the branch: slices 1-3 + RFC-113/114 phases + RFC docs.)

## Push

Nothing is pushed. When you say "push": rebase onto `origin/main`, run `make ci-fast` + `ci-ui-fast` (+ stack-test for the viewer testid/chip additions), then push and open/keep the PR against the RFC epics (#1470/#1471/#1472).
