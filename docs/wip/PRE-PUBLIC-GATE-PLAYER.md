# Pre-public gate assessment — consumer player (player-public)

Runs the `docs/security/THREAT_MODEL.md` pre-public gate (8 items) against the
player-public stack (#1163 / ADR-116), for epic #1262 (#1266). Evidence-backed,
not assumed. **Legend:** ✅ pass · 🟡 decision/accept · 🔲 operator step.

**Bottom line:** the player-public backend is purpose-built to clear this gate
(the case the gate exists to stop — "a podcast-family API sharing the operator
api's privilege profile"). Two independent layers restrict the public surface to
`/api/app/*`, both **test-locked** (15 tests pass). No hard blockers. Open items:
`learning-app` image-pin, and the operator secrets/domain (separate children).

## The load-bearing property — public surface is `/api/app/*` only (verified two ways)

1. **Backend app-only mode** (`server/app.py` `_mount_api_routers`): with
   `PODCAST_SERVE_APP_ONLY=1`, the operator/read `/api/*` routers are **not mounted**
   at all — only `/api/health` + `/api/app/*`. So even direct backend access can't
   reach the operator surface.
2. **Player nginx** (`web/learning-player/nginx.conf`): proxies **only** `/api/app/auth/`
   + `/api/app/` to the backend; everything else is static. The operator `/api/*`
   surface is never forwarded.
3. The backend `api` is **not published** — only `learning-app`'s nginx is exposed
   (loopback `:8092` → Caddy). Browser → Caddy → player nginx (`/api/app/*` only) → backend.

**Test-locked:** `test_app_only_mode.py`, `test_player_public_compose_contract.py`,
`test_app_operator_guard.py` — **15 passed** (2026-07-22).

## Gate items

| # | Item | Verdict | Evidence |
|---|---|---|---|
| 1 | No `docker.sock`, no write-scope cloud keys on the public container | ✅ | app-only backend mounts corpus `:ro` + appdata bind — **no sock**; env has only OAuth creds + session secret + Sentry DSN (app-auth, **no provider/cloud keys**). `learning-app` = static nginx, nothing. |
| 2 | `cap_drop: ALL`, `no-new-privileges`, non-root, read-only rootfs | ✅ | `learning-app`: `cap_drop:[ALL]` + minimal caps + `no-new-privileges` + **`read_only:true`** + tmpfs. `api`: `cap_drop:[ALL]` + minimal caps + `no-new-privileges`; not read_only by design (entrypoint chown + appdata writes) — documented, matches the boot-verified operator api. |
| 3 | API needs authN/authZ, tight CORS, rate limiting | ✅ | `/api/app/*` gated by **Google OAuth** (`APP_OAUTH_PROVIDER=google`). nginx **rate limits**: `lp_api` 20r/s burst 40, `lp_auth` 2r/s burst 8, `limit_req_status 429`. CORS: player + its api proxy are **same-origin** (browser → player-domain → nginx → api), so cross-origin CORS isn't a surface. |
| 4 | Caddy `admin off`; on-demand off; catch-all denies unknown Host | ✅ | Engine-level (shared base Caddyfile), already live. |
| 5 | Egress to `169.254.169.254` blocked from the tenant | ✅ | Host `DOCKER-USER` DROP (block-metadata-egress, converged) covers all containers incl. the player. |
| 6 | Deployed image digest-pinned (or cosign-verified) | 🟡 | `api` pins `ghcr.io/...:sha-<tag>` at deploy (deploy.sh). `learning-app` is `podcast-scraper-learning-app:latest` — **locally built** (same posture as orrery). Accept for launch, or tighten to a digest later. |
| 7 | Security alerting covers the new surface | ✅ | Shared fail2ban `caddy-access` jail + Alloy→Loki cover the player vhost's access log. Player **app errors → GlitchTip project 1** (dashless DSN, browser-side). |
| 8 | Rollback proven | ✅ | `rm /etc/caddy/sites/player.caddy && sudo systemctl restart caddy` → player public down, tailnet + box + other tenants unaffected. **Smoke-test at cutover.** |

## Consolidated — no hard blockers

1. **#6 (accept/decide):** `learning-app` runs a locally-built `:latest` image (like orrery). Accept for launch or add a digest-pin later.
2. **Operator steps (separate children):** OAuth app + secrets (#1264), domain + DNS + vhost (#1263), corpus volume name (#1265), deploy + verify (#1267).
3. **At cutover:** smoke-test the rollback (#8).

**Verdict: player-public CLEARS the pre-public gate.** The two-layer `/api/app/*`
restriction is verified + test-locked; hardening, alerting, egress-guard, and
rollback all pass. Ready for the operator-owned steps (OAuth, domain, deploy).
