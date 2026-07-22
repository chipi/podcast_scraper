# Player public launch (consumer player → public domain)

The player-first public launch (#1163, [ADR-116](../adr/ADR-116-privilege-split-public-control-api.md)
§Sequencing). The consumer player goes public on its own domain via the shared Caddy
edge ([ADR-114](../adr/ADR-114-shared-multi-tenant-public-edge-caddy.md)); the operator /
kg-gi surface stays tailnet-only until it is hardened (RBAC #1164 + the split #1165).

**Why this is safe:** the player uses only `/api/app/*`. Its backend runs
`PODCAST_SERVE_APP_ONLY=1`, which mounts **only** `/api/app/*` + health — no operator/read
`/api/*` — so it carries **no `docker.sock` and no provider keys**. It passes the
[pre-public gate](../security/THREAT_MODEL.md#pre-public-gate-run-before-any-new-public-vhost).

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

- **secrets:** `TS_AUTHKEY`, `PROD_SSH_PRIVATE_KEY`, `PLAYER_APP_SESSION_SECRET`,
  `PLAYER_GOOGLE_CLIENT_SECRET`
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

## Rollback

Pull the vhost + reload (`rm /etc/caddy/sites/player.caddy && systemctl reload caddy`) →
public down; or `docker compose -f compose/docker-compose.player-public.yml down`. The
operator/tailnet surface is unaffected.
