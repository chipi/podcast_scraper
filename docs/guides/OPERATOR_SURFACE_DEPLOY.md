# Operator surface deploy runbook (`operator.closelistening.app`)

**What this is:** how to deploy + operate the **public operator surface** — the gated
`gi-kg-viewer` at `operator.closelistening.app` ([RFC-108](../rfc/RFC-108-operator-public-gated-surface.md),
epic #1320). Sibling of the [player launch runbook](PLAYER_PUBLIC_LAUNCH.md); same
coming-soon gate + Google OAuth, but **operator roles** (≥creator) and a **curated
read-only** API subset ([ADR-116](../adr/ADR-116-privilege-split-public-control-api.md) privilege split).

> **Why it's a separate least-privilege surface (T-01):** the *privileged* operator/api
> (docker.sock + LLM keys + index_rebuild/ops/triggers) stays **tailnet-only**. This public
> plane runs `PODCAST_SERVE_OPERATOR_PUBLIC=1` — no socket, no keys, curated routes each
> behind `require_viewer_access` (≥creator). See [THREAT_MODEL](../security/THREAT_MODEL.md) T-01.

## Architecture (what runs)

- **Caddy vhost** `infra/caddy/operator.caddy` — coming-soon `/preview` basic-auth gate +
  distinct cookie `cl_op_preview`; loopback `127.0.0.1:8093`; Cloudflare-fronted + origin-locked.
- **Compose** `compose/docker-compose.operator-public.yml` — `api`
  (`PODCAST_SERVE_OPERATOR_PUBLIC=1`, no socket/keys, corpus `:ro`) + `viewer` (nginx, loopback
  `:8093`). Under `-p operator` (its **own** compose project, never the privileged stack's).
- **Secrets overlay** `compose/docker-compose.operator-secrets.yml` — joined only when
  `OPERATOR_SECRETS_VIA_FILES=1` (ADR-115: OAuth client secret + session secret + Sentry DSN
  delivered as tmpfs files, none at rest on disk).

## Prerequisites (one-time, operator-side)

| Kind | Name | Value / where |
| --- | --- | --- |
| Cloudflare | DNS | `operator.closelistening.app` → orange-cloud (proxied) |
| Google OAuth | redirect URI | add `https://operator.closelistening.app/api/app/auth/callback` to the **player's** OAuth client (reused per RFC-108) |
| GH secret | `PLAYER_GOOGLE_CLIENT_SECRET`, `PLAYER_APP_SESSION_SECRET` | reused from player |
| GH secret | `PROD_SENTRY_DSN_API` | homelab GlitchTip DSN |
| GH secret | `OPERATOR_PREVIEW_COOKIE` | high-entropy value for the coming-soon gate cookie |
| GH var | `OPERATOR_DOMAIN` | `operator.closelistening.app` |
| GH var | `OPERATOR_ALLOWED_EMAILS` | allowlisted sign-in emails (default-deny if empty) |
| GH var | `APP_ADMIN_EMAILS` | emails that get admin (must own the domain) |
| GH var | `OPERATOR_SIGNUP_MODE` | `allowlist` (NEVER `open` — the viewer self-grants creator; the api refuses to boot on `open`, ADR-128-era guard) |
| GH var | `OPERATOR_SECRETS_VIA_FILES` | `1` (secrets to tmpfs; the overlay must exist on the box first — merged) |
| tailnet | SSH | `deploy@prod-podcast` reachable over the tailnet |

## Deploy

The image is **baked** — app/nginx changes need a fresh build **before** the deploy.

1. **Build + publish the image** from `main` (the operator runs the same image as the
   privileged api). Dispatch "Stack test" so it publishes `sha-<7>`:

   ```sh
   gh workflow run "Stack test" --ref main
   ```

   Wait for `publish` to push `api` + `viewer` `sha-<7>` (~80 min, arm64 QEMU). **The tag is
   the 7-char short sha** — `git rev-parse --short=7 origin/main` (NOT the 8-char `git --short`).
2. **Dispatch the gated deploy**, pinned to that sha:

   ```sh
   gh workflow run deploy-operator.yml -f confirm=OPERATOR_DEPLOY -f override_image_sha=<7-char-sha>
   ```

   It **waits at the `prod` environment approval gate** — approve it (Review deployments).
   The deploy: stages `.env.operator` + tmpfs secrets → `docker compose -p operator up` →
   drops `operator.caddy` → `caddy validate` → `systemctl restart caddy` → in-container health.

## Verify (after the deploy goes green)

```sh
curl -sI https://operator.closelistening.app/            # 200, coming-soon HTML, server: cloudflare
curl -sI https://operator.closelistening.app/preview     # 401 (basic-auth doorman present)
curl -sI https://operator.closelistening.app/api/search  # 200 coming-soon (gated) — NOT open JSON
```

- **Sign-in end-to-end** (behind the doorman): `/preview` → cookie → viewer login wall →
  "Sign in" → **307 to `accounts.google.com` with
  `redirect_uri=https%3A%2F%2Foperator.closelistening.app%2Fapi%2Fapp%2Fauth%2Fcallback`**
  (the `https` callback — this is the regression the smoke guards). The `smoke-operator.yml`
  live smoke runs automatically after the deploy and asserts exactly this.
- **Corpus loads:** open in a **fresh incognito** window (the `ps_corpus_path` localStorage seed
  only sets when unset) — the SPA should land on the corpus with data (`PODCAST_DEFAULT_CORPUS_PATH=/app/output`).
- **Secrets at rest:** with `OPERATOR_SECRETS_VIA_FILES=1`, the OAuth/session secrets are in
  `/dev/shm/operator-secrets/` (RAM), dropped from `.env.operator`.

## Rollback (<5 min)

Re-dispatch the deploy pinned to the **previous good `sha-<7>`**:

```sh
gh workflow run deploy-operator.yml -f confirm=OPERATOR_DEPLOY -f override_image_sha=<previous-sha>
```

(Compose recreates from the prior image; the coming-soon gate stays up throughout.)

## Gotchas (cost real time this session)

- **`redirect_uri_mismatch`** → the viewer nginx must forward Caddy's real `X-Forwarded-Proto`
  (a `map`, not `$scheme`), or the api builds an `http://` callback and Google rejects it.
  Guarded by `test_viewer_nginx_preserves_forwarded_proto` + the live smoke.
- **7-char sha, not 8.** GHCR tags are `sha-<7>`; `git rev-parse --short=7`. An 8-char sha →
  `manifest unknown` at the pre-SSH check.
- **Deploy script exec bit.** `infra/deploy/deploy-operator.sh` must be `100755` — a `100644`
  (mode strip) → exit-126 `Permission denied`.
- **`OPERATOR_SECRETS_VIA_FILES=1` needs the overlay on the box first.** Set the var only
  *after* `docker-compose.operator-secrets.yml` is on `main` (else the deploy fails on the `-f` join).
- **Empty corpus on first login** → the viewer service needs `PODCAST_DEFAULT_CORPUS_PATH=/app/output`.

## References

- [RFC-108](../rfc/RFC-108-operator-public-gated-surface.md) · [ADR-116](../adr/ADR-116-privilege-split-public-control-api.md)
  · [PLAYER_PUBLIC_LAUNCH](PLAYER_PUBLIC_LAUNCH.md) · [THREAT_MODEL](../security/THREAT_MODEL.md) T-01
- `deploy-operator.yml` · `smoke-operator.yml` · `compose/docker-compose.operator-public.yml`
