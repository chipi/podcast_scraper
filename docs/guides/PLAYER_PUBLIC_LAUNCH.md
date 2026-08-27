# Player public launch (consumer player → public domain)

The player-first public launch (#1163, [ADR-116](../adr/ADR-116-privilege-split-public-control-api.md)
§Sequencing). The consumer player goes public on its own domain via the shared Caddy
edge ([ADR-114](../adr/ADR-114-shared-multi-tenant-public-edge-caddy.md)); the operator /
kg-gi surface stays tailnet-only until it is hardened (RBAC #1164 + the split #1165).

**Why this is safe:** the player uses only `/api/app/*`. Its backend runs
`PODCAST_SERVE_APP_ONLY=1`, which mounts **only** `/api/app/*` + health — no operator/read
`/api/*` — so it carries **no `docker.sock` and no provider keys**. It passes the
[pre-public gate](../security/THREAT_MODEL.md#pre-public-gate-run-before-any-new-public-vhost).

## Public surfaces (vhosts)

All fronted by the one shared Caddy edge (ADR-114), TLS-terminated, routed by `Host` to a
loopback container. The operator / kg-gi surface stays **tailnet-only** and is not listed here.

| Vhost | Backend (loopback) | Auth | Purpose | Caddy file |
| --- | --- | --- | --- | --- |
| `<domain>` (apex) | `learning-app` → nginx → `api` (`:8092`) | coming-soon cookie gate → Google OAuth session | Player PWA + `/api/app/*` consumer platform **+ the OAuth 2.1 authorization server** (`/api/app/mcp/oauth/*`, `/.well-known/oauth-authorization-server` — gate-exempt) | `player.caddy` |
| `mcp.<domain>` | `mcp` (`:8009`) | **bearer** (OAuth access token or PAT), in-process | Remote MCP server (`/mcp`) + RFC 9728 discovery. No coming-soon gate — the auth is the gate. **Only when MCP is enabled.** | `mcp.caddy` |
| `telemetry.<domain>` | homelab GlitchTip | none (ingest-only) | Browser error-SDK ingest paths only | `player-telemetry.caddy` |
| `analytics.<domain>` | Umami | none (script + collect) | Cookieless analytics | `player-analytics.caddy` |

## Prerequisites

- The Caddy edge engine is live on the box (imperative-once install) and the firewall
  opens 80 + 443 (`tofu apply`, verify in-place). See [PROD_RUNBOOK](PROD_RUNBOOK.md).
- A registered **player domain** and a DNS **A record** → the VPS IP.

## OAuth setup (Google — the only real provider)

1. **Google Cloud Console** → *APIs & Services*:
   - *OAuth consent screen* → External. Add your email; publish (or keep testing + add
     test users).
   - *Credentials* → *Create OAuth client ID* → **Web application**.
   - **Authorized redirect URI**: `https://<player-domain>/api/app/auth/callback`.
   - Copy the **Client ID** and **Client secret**.
2. Generate the session signing secret: `openssl rand -hex 32`.

The three values:

| Env var | Secret? | Value |
| --- | --- | --- |
| `APP_OAUTH_PROVIDER` | no | `google` (the switch — see below) |
| `APP_OAUTH_GOOGLE_CLIENT_ID` | no | from Google |
| `APP_OAUTH_GOOGLE_CLIENT_SECRET` | **yes** | from Google |
| `APP_SESSION_SECRET` | **yes** | `openssl rand -hex 32` |

`APP_ADMIN_EMAILS` is **not** needed — player users default to `listener`.
`APP_SESSION_COOKIE_SECURE=true` is already set in the compose.

## Provider switch: mock ↔ google (one explicit config)

`APP_OAUTH_PROVIDER` is the single switch, set by the deployment (its compose is the
"profile" for a deploy-wide concern — not a per-corpus `config/profiles/*.yaml`):

- `mock` → dev/e2e only (offline fake identities; logged loudly; **never prod**). Used by
  the local demo (`docker-compose.app.yml`).
- `google` → real Google OIDC (needs the two creds above). Set in
  `docker-compose.player-public.yml`.
- unset / anything else → auth disabled.

Explicit by design: Google creds alone never enable a real provider — a half-set env can't
accidentally go live.

**Login UI adapts automatically** (no config): under `mock` the sign-in view shows the
dev-user picker; under `google` (`/auth/dev-users` returns `enabled: false`) it shows the
normal "Sign in" button. No mock accounts are ever shown on a Google deployment.

## Deliver the two secrets

Bare minimum (secure, and how prod secrets work today): stage `APP_SESSION_SECRET` +
`APP_OAUTH_GOOGLE_CLIENT_SECRET` as GH Actions secrets → the deploy renders them into the
`player-api` container env. **Never commit them.** The full sops/age file-mount delivery
([ADR-115](../adr/ADR-115-multi-tenant-secret-delivery-sops-tmpfs-files.md)) is a later
hardening, not a launch blocker.

## Deploy

**Automated (recommended): `deploy-player.yml`** — manual `workflow_dispatch`, typed
confirm `PLAYER_DEPLOY`, tailnet-only SSH. It refreshes the repo on the box, stages
`.env.player` (secrets via `/dev/shm` scp — never inline over ssh), then runs
`infra/deploy/deploy-player.sh` (compose up + vhost drop + **validate-before-reload** +
health), and probes the public domain. Stage these first (once):

- **secrets:** `TS_OAUTH_CLIENT_ID`/`_SECRET`, `PROD_SSH_PRIVATE_KEY`, `PLAYER_APP_SESSION_SECRET`,
  `PLAYER_GOOGLE_CLIENT_SECRET` (see [ADR-143](../adr/ADR-143-tailscale-oauth-migration-and-tag-self-ownership.md) for Tailscale auth)
- **vars:** `PROD_TAILNET_FQDN`, `PLAYER_DOMAIN`, `PODCAST_CORPUS_VOLUME`,
  `PLAYER_GOOGLE_CLIENT_ID`

Prereqs: the Caddy edge + firewall 80/443 already live on the box; DNS A-record for
`PLAYER_DOMAIN` → the VPS.

**Manual equivalent** (what the script does):

```bash
# On the VPS, standalone from the operator stack (shares the corpus volume read-only):
PLAYER_DOMAIN=<domain> PODCAST_CORPUS_VOLUME=<operator-stack-corpus-volume> \
  APP_SESSION_SECRET=... APP_OAUTH_GOOGLE_CLIENT_ID=... APP_OAUTH_GOOGLE_CLIENT_SECRET=... \
  infra/deploy/deploy-player.sh
```

**Rate limiting (T-06):** the player nginx rate-limits `/api/app/*` per real client IP
(`real_ip` recovers it from Caddy's `X-Forwarded-For`) — a normal API zone + a tighter
zone on the auth endpoints; excess → `429`.

## Corpus sharing (read-during-write)

The player shares the operator stack's corpus volume, mounted **read-only**
(`corpus_data:/app/output:ro`) — so a route bug can never corrupt the corpus, and reads
while the operator pipeline writes are safe by construction:

- The **serving path is read-only** — `index_pool` (ADR-099 #995) opens the LanceDB tables
  for read; the only corpus writes (`write_index_meta`) happen at *build* time, off the
  serving path.
- **No staleness** — the pool invalidates its cached handle on the index's **mtime change**,
  so a pipeline rebuild is picked up automatically.
- **LanceDB is versioned** — a reader sees a consistent snapshot while a writer commits a
  new version; concurrent read+write does not corrupt or block.

The compose contract gates the read-only mount; a deeper "search over a `:ro` index"
runtime test belongs in the CI tier that has `lancedb` installed (skipped locally).

### Backup (per-user data)

`player_appdata` (playback/notes/favorites) is **not regenerable** — it's real user data —
so it is a **host bind mount** at `/srv/podcast-scraper/player-appdata` (not a Docker
volume) and has its own backup: **`backup-player-appdata-prod.yml`** (`workflow_dispatch`)
streams it over the tailnet to the backup repo. `deploy-player.sh` creates + chowns the dir
(uid 1000) before first boot. Needs `BACKUP_REPO_TOKEN` + `PODCAST_BACKUP_REPO` (shared with
the corpus backup).

## Verify

- `https://<player-domain>/` serves the PWA; sign-in redirects to Google and back.
- **Forwarded headers (handled):** the OAuth redirect URI is derived from the request, so
  the backend must see the public `https://<player-domain>` origin through the Caddy→nginx→
  uvicorn chain. This is wired: the player nginx preserves Caddy's `X-Forwarded-Proto`
  (a `map` — it no longer overwrites it with its own `http` scheme), and the backend runs
  with `FORWARDED_ALLOW_IPS=*` + `proxy_headers` so uvicorn honors it. Still worth a
  first-deploy sanity check: if sign-in bounces with a redirect-URI mismatch, confirm those
  two are in effect.
- `https://<player-domain>/api/jobs` → 404 (the app-only backend does not mount it).

### Post-deploy live smoke (automatic + gating)

The player deploy runs a **`smoke` job** (`deploy-player.yml`, `needs: deploy`) as the last part of
the SAME run: headless Playwright (`web/learning-player/e2e/live`, `npm run test:e2e:live`) against
the just-deployed `closelistening.app`, read-only, passing the coming-soon gate with the
`PLAYER_PREVIEW_PASS` secret. Because it lives in `deploy-player` (the reusable workflow), it fires on
**every** path — a standalone dispatch **and** `deploy-all-prod` — so **the deploy is not green until
the smoke passes.**

- **A red `smoke` job = a failed deploy.** Treat it as such: read the failure (Playwright report is
  uploaded as the `playwright-live-report` artifact), then either fix-forward fast or **roll back**
  (below). Do not leave prod on a build the smoke rejected.
- **Failure also alerts GlitchTip** (`PROD_SENTRY_DSN_PLAYER_API`, tags `surface:player` /
  `stage:post-deploy-smoke`, with the run URL) — so a bad deploy reaches you even if no one is
  watching the Actions tab.
- **Agents:** after any player deploy, confirm the deploy run's `smoke` job is green before calling
  the deploy done. It is a gate, not a notification you can skip.
- **History / why this exists:** the smoke used to be a *separate* `smoke-player.yml` triggered by
  `workflow_run` on `"Deploy player — PUBLIC surface"`. A reusable-workflow call emits **no**
  `workflow_run` event, so on the `deploy-all-prod` path the smoke **never fired** — a UI regression
  ("Learning Player" → "Close Listening") shipped to prod unvalidated and unseen. `smoke-player.yml`
  is now **manual-only** (re-run the smoke without a redeploy); the gate is the in-deploy `smoke` job.

#### Per-user test account (Collections / Library)

The public + anonymous specs (`surfaces.live.spec`, `trending.live.spec`, `smoke.live.spec`) need no
login. The **per-user** surfaces — Collections, Library — run under `account.live.spec.ts` as a
dedicated **prod test account**. A headless smoke can't complete a real Google sign-in, so the spec
**mints the app's own session token** (`app_sessions.sign` — HMAC-SHA256 over the session secret,
byte-for-byte in Node) for the test user's id and calls the API with `Authorization: Bearer <token>`
(and the same value as the `lp_session` cookie for the one UI check). All writes are reversible
(create → assert → delete) and scoped to the test account, so the smoke leaves no residue.

The spec **skips cleanly** unless BOTH of these are present, so nothing breaks before setup:

| Name | Kind | Value |
| --- | --- | --- |
| `PLAYER_APP_SESSION_SECRET` | secret (already set) | the prod session-signing secret — the **same** one the backend verifies with (`deploy-player.yml` sets `APP_SESSION_SECRET` from it). Nothing to do. |
| `PLAYER_SMOKE_USER_ID` | repo **variable** | the **stored** user id of the seeded test account. |

**One-time operator setup:**

1. **Seed a test user in the prod user store.** Sign in to `closelistening.app` once with a
   dedicated account (e.g. a `+smoke` Google alias), or seed it however prod users are provisioned.
   It must be an account you're content to have a CI job create/delete Collections under.
2. **Get its *stored* id — not the email.** The id is an opaque hash like
   `u_52a6b85b8b67de5964bb0429` (from `GET /api/app/me` while signed in, or the prod user store),
   **not** the login hint/email. The mint payload's `user_id` must equal the stored id or the token
   resolves to no user and every per-user assert fails.
3. **Set the repo variable:** `gh variable set PLAYER_SMOKE_USER_ID --body 'u_…'`.

Once the variable is set, the next deploy's `smoke` job covers the per-user surfaces automatically.
Leave it unset and the deploy still gates on the public/anon specs — the per-user block just skips.

### Operator surface smoke (`deploy-operator.yml`)

The operator viewer (`operator.closelistening.app`, `gi-kg-viewer`) has the **same** gating pattern: a
`smoke` job (`needs: deploy`) runs `web/gi-kg-viewer` live specs after every operator deploy —
standalone dispatch **and** `deploy-all-prod` (which calls `deploy-operator` reusably). It covers the
public coming-soon gate, the Google sign-in redirect (HTTPS `redirect_uri` regression guard), health,
and — when the account below is seeded — the **authed operator plane** as a creator (read routes 200;
admin routes **403**, i.e. the role boundary). Full design: `docs/wip/OPERATOR-SMOKE-TEST-PLAN.md`.

**Creator-only by design** — we deliberately do NOT mint an *admin* session in CI (an admin token that
can read the user-management plane is needless overhead + a security gap for a smoke; a creator token
proves the surface works AND that the admin boundary holds). One test account, stored `u_…` id (not the
email); the authed block skips until set:

| Name | Kind | Value |
| --- | --- | --- |
| `PLAYER_APP_SESSION_SECRET` | secret (already set) | reused — the operator deploy verifies with the same secret. |
| `OPERATOR_SMOKE_CREATOR_USER_ID` | repo **variable** | stored id of a seeded **creator** test account. |

One-time setup: seed one account (e.g. a `+smoke-creator` alias), have an admin grant it `creator`,
read its stored `u_…` from `/api/app/me`, then `gh variable set OPERATOR_SMOKE_CREATOR_USER_ID --body 'u_…'`.

## Rollback

Pull the vhost + reload (`rm /etc/caddy/sites/player.caddy && systemctl reload caddy`) →
public down; or `docker compose -f compose/docker-compose.player-public.yml down`. The
operator/tailnet surface is unaffected.

## MCP — remote agent access (RFC-112)

Lets an external AI agent (claude.ai custom connector, Claude Code, Cursor) search + read the
corpus **as a signed-in platform user** over MCP, with their own model (D6-safe). Off by default;
turning it on is additive to the player deploy.

**Topology.** The app-only `api` hosts the OAuth 2.1 authorization server
(`/api/app/mcp/oauth/*` + `/.well-known/oauth-authorization-server`, on the player apex) and the
tailnet-only verify seam (`/internal/mcp/verify`). A separate **`mcp`** container (same low-priv
image, `podcast mcp --transport http`, corpus **read-only**, no keys/sock) serves the corpus tools
on `127.0.0.1:8009` and verifies every bearer against that seam. Caddy fronts it at
`mcp.<domain>`; the coming-soon gate exempts the OAuth AS paths (they're cookie-less
server-to-server calls).

**Enable (operator).**

0. **Image:** the `mcp` service + the OAuth server need the MCP SDK, added to the api image in
   RFC-112. Deploy a `PODCAST_IMAGE_TAG` built **after that change merged to main** (the deploy pins
   the newest published `main` sha automatically — just ensure the merge's image is published first).
1. **DNS:** add `A mcp.<player-domain> → VPS IP` (CF-proxied like the apex, or grey-cloud).
2. **Secret:** set GH secret `PLAYER_INTERNAL_MCP_TOKEN` to a high-entropy value
   (`openssl rand -hex 32`). This gates the verify seam **and** the `mcp` service — **unset = MCP
   stays fully inert** (verify returns 503 → every connect 401, and `deploy-player.sh` skips the
   `mcp` vhost). Optionally set var `PLAYER_MCP_ALLOWED_ORIGINS` (browser DNS-rebind guard;
   unnecessary for claude.ai, which connects server-side).
3. **Deploy:** run the player deploy workflow. It stages `INTERNAL_MCP_TOKEN` + derives
   `APP_MCP_ISSUER_URL=https://<domain>` and `APP_MCP_RESOURCE_URL=https://mcp.<domain>`, brings up
   the `mcp` service, installs `mcp.caddy`, and runs a non-fatal reachability probe.
4. **Grant users:** a platform admin flips `mcp_access` on each allowed user
   (`PATCH /api/app/admin/users/{id}` or the user-management UI). Without it, the "Connected agents"
   UI is hidden and every OAuth/PAT step is refused.

**Use (end user).** In player **Profile → Connected agents**: copy the **connector URL**
(`https://mcp.<domain>/mcp`) into claude.ai's "add custom connector" (it self-registers via DCR,
you approve a consent screen once), or mint a **PAT** for a CLI client (Claude Code / Cursor:
`Authorization: Bearer clp_mcp_…`).

**Verify.**

- `curl https://mcp.<domain>/.well-known/oauth-protected-resource` → `200` with
  `{resource, authorization_servers:[https://<domain>]}`.
- `curl -X POST https://mcp.<domain>/mcp` (no bearer) → `401` with
  `WWW-Authenticate: Bearer resource_metadata="…"`.
- `curl https://<domain>/.well-known/oauth-authorization-server` → `200` RFC 8414 metadata
  (NOT the coming-soon HTML — that's the gate exemption working).

**Rollback.** `rm /etc/caddy/sites/mcp.caddy && systemctl restart caddy` (surface down), or unset
`PLAYER_INTERNAL_MCP_TOKEN` + redeploy (fully inert). The player + operator surfaces are unaffected.

**Residue (see THREAT_MODEL T-13).** v1 shared-corpus (per-user gating/attribution, not
confidentiality); no `aud`-binding; no app-level per-principal rate-limit / audit / consent-revoke
UI (admin `mcp_access` pull is the kill-switch).
