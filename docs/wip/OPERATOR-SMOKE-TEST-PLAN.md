# Operator surface — post-deploy live smoke test plan

**Status:** planning → implementing (2026-08-27)
**Surface:** `operator.closelistening.app` (the `gi-kg-viewer` SPA + `PODCAST_SERVE_OPERATOR_PUBLIC` backend)
**Sibling:** mirrors the player post-deploy smoke (`docs/guides/PLAYER_PUBLIC_LAUNCH.md` → "Post-deploy live smoke"). This is the **third surface** (task #18) after the player public + per-user smoke.

---

## 1. What already exists (build on, do not duplicate)

- `web/gi-kg-viewer/playwright.live.config.ts` — targets `operator.closelistening.app`, **desktop-chrome only**, passes the coming-soon gate via `httpCredentials` (reuses the player's `PLAYER_PREVIEW_PASS` doorman). Login-wall aware.
- `web/gi-kg-viewer/e2e/live/smoke.live.spec.ts` — the **unauthed** smoke: public sees coming-soon; preview users reach the **login wall**; sign-in entrypoint **307s to Google** with the correct HTTPS `redirect_uri` (regression guard for the 2026-07-25 `redirect_uri_mismatch`); `/api/health` 200.
- `web/gi-kg-viewer/e2e/handoff-production/*.spec.ts` — a real-stack "handoff" suite (cold-start, concurrency, cross-entry, digest, filters, hot-state, lifecycle, repeat-click) + `_helpers.ts`. **Tier-3-style** — the closest existing analog to what we're extending.
- `web/gi-kg-viewer/e2e/*.spec.ts` (mock-backed) — dashboard, digest, library, graph, pipeline, auth-roles, person-landing, etc. These run under the local `playwright.config.ts` against mocked APIs.

## 2. THE GAP (why this is task #18)

**`deploy-operator.yml` has NO smoke job.** The operator live smoke `smoke.live.spec.ts` exists but is **never run after a deploy** — worse than the player's old gap (that at least had a `workflow_run` trigger that misfired; this has nothing). A broken operator deploy ships unvalidated and unseen. Closing this is the core of the work.

## 3. Auth model (drives the test account design)

- **Coming-soon gate** (Caddy, host-side): basic-auth doorman, same hashes as the player (`PLAYER_PREVIEW_PASS`). `httpCredentials` clears it.
- **Login wall** behind the gate: real Google OAuth on prod (`APP_OAUTH_PROVIDER=google`); the mock `?as=` login can never ship. Nothing renders until a session exists.
- **Roles** (`app_roles.py`, totally ordered `listener < creator < admin`):
  - `listener` — player only; **403** on operator routes.
  - `creator` — viewer base: **digest / library / graph** (KG curation).
  - `admin` — everything creator has **plus dashboard / ops / configuration / user management**. Dashboard is `v-if="auth.isAdmin"` in `App.vue`.
- **Operator APIs gated ≥creator** on the public surface (`_OPERATOR_PUBLIC_READ_ROUTES`, each mounted with `require_viewer_access`): `usage_routes, artifacts, index_stats, search, relational, query_activity, explore, corpus_library, corpus_binary, corpus_media, corpus_text_file, corpus_metrics, corpus_coverage, corpus_persons, corpus_digest, corpus_enrichments, corpus_topic_clusters, corpus_theme_clusters, corpus_trending, cil`. All **read-only**. `index_rebuild` / `ops` are NOT mounted on the public operator surface.

## 4. What to validate

### 4a. Unauthed (keep existing, already green in spirit)
- Public (no creds) sees coming-soon; `login-button` absent.
- Preview users reach the login wall (`Sign in to explore the knowledge graph`).
- Sign-in 307→Google with HTTPS operator `redirect_uri`.
- `/api/health` 200.

### 4b. Authed as **creator** (NEW — minted-session test account)
The operator surface is read-only, so unlike the player there is **no reversible-write** round-trip; we assert **authenticated reads succeed (200, not 401/403)** across the creator-visible plane:
- `/api/app/me` → 200, `role ∈ {creator, admin}`.
- A representative slice of `_OPERATOR_PUBLIC_READ_ROUTES` returns 200 with well-formed bodies: `corpus_digest`, `corpus_library`, `corpus_metrics`, `corpus_coverage`, `index_stats`, `search` (a trivial query), `corpus_trending`. (Pick the cheap, always-present ones; avoid heavy binary/media.)
- **Negative guard:** the *player* per-user routes and the *admin-only* routes behave correctly for a creator (see 4d).
- One **UI** check (cookie auth): past `/preview`, the SPA boots into the viewer (digest or graph shell visible), NOT the login wall.

### 4c. Authed as **admin** — DROPPED (2026-08-27 decision)
We deliberately do **not** mint an admin session in CI. An admin token from a CI secret that can read
the admin/user-management plane (`/api/app/admin/users` returns user emails) is needless overhead + a
security gap for a smoke. A **creator** token already proves the surface works AND that the admin
boundary holds (4d). If admin-surface coverage is ever wanted, it belongs in stack-test (mock
provider), not a prod smoke.

### 4d. Role-boundary guard (the operator-specific value)
- creator token → operator read routes **200**; admin-only route **403** (asserted with the *creator*
  token — the denial is the assertion; no admin token minted).

## 5. Test account design (mirror the player mechanism)

Same self-mint approach as `account.live.spec.ts` (validated: Node HMAC == `app_sessions.verify`). The operator deploy uses the **same** `PLAYER_APP_SESSION_SECRET` (`deploy-operator.yml` sets `APP_SESSION_SECRET` from it), so the identical mint works.

One seeded prod account (creator role, assigned by an admin — not self-service):

| Env (repo var) | Role | Purpose |
| --- | --- | --- |
| `OPERATOR_SMOKE_CREATOR_USER_ID` | creator | viewer read plane + the "no admin" 403 boundary |

Uses the **stored opaque `u_…` id**, not the email (same gotcha proven on the player: mock hint `smoke-user` stored as `u_52a6b85b8b67de5964bb0429`). Specs **skip cleanly** unless `PLAYER_APP_SESSION_SECRET` + the gate password + the user-id var are set.

## 6. Reuse from existing e2e

- Lift the **assertion shapes** (not the mock wiring) from the mock-backed `digest.spec.ts`, `library.spec.ts`, `dashboard.spec.ts`, `auth-roles.spec.ts` for the authed UI checks.
- `handoff-production/_helpers.ts` — reuse polling/robustness helpers where they fit a live authed context.
- Keep the live specs **read-only + fast** (a smoke, not the full tier-3 walk). The deep authed viewer flow stays in stack-test (mock provider).

## 7. Wiring into the deploy (close the gap)

Add a gating `smoke` job to `deploy-operator.yml`, mirroring `deploy-player.yml`:
- `needs: deploy`, `if: needs.deploy.result == 'success'`, `working-directory: web/gi-kg-viewer`.
- `npm ci` → `playwright install --with-deps chromium` → `npm run test:e2e:live`.
- env: `LIVE_BASE_URL=https://${OPERATOR_DOMAIN}`, `OPERATOR_PREVIEW_USER`, `OPERATOR_PREVIEW_PASS` (→ `PLAYER_PREVIEW_PASS`), `PLAYER_APP_SESSION_SECRET`, `OPERATOR_SMOKE_CREATOR_USER_ID`, `OPERATOR_SMOKE_ADMIN_USER_ID`.
- **GlitchTip alert on failure** (`PROD_SENTRY_DSN_PLAYER_API`, tags `surface:operator` / `stage:post-deploy-smoke`) + upload the Playwright report artifact.
- **A red smoke = a failed operator deploy.**

## 8. One-time operator prod setup (the "device setup" steps)

1. Seed **one** prod account (a dedicated `+smoke-creator` Google alias) and have an admin grant it `creator` (roles assigned via user-management).
2. Get its **stored** `u_…` id (from `/api/app/me` while signed in, or the user store) — not the email.
3. Set the repo var: `gh variable set OPERATOR_SMOKE_CREATOR_USER_ID --body 'u_…'`.
4. Confirm `OPERATOR_DOMAIN` + `OPERATOR_PREVIEW_USER` vars are set (they gate the deploy already).

## 9. NOT covered / risks / open questions

- **No reversible-write coverage** — the public operator plane is read-only; we assert reads + role boundaries, not mutations. `index_rebuild`/`ops` (the write/admin-heavy surface) are **not** mounted publicly, so they're out of scope for this smoke by design.
- **Local validation limit** — as with the player, the mint can be cross-checked against `app_sessions.verify` and the authed reads against a locally-booted `operator_public` server with `APP_SESSION_SECRET=e2e-secret` + seeded creator/admin users; the **prod** run can't run until the two accounts are seeded + vars set.
- **Native / desktop operator app** — out of scope (operator is a desktop web surface; no native shell).
- **Admin-only API to assert** — need to pick the specific `app_admin` read route + the exact UI marker for the Dashboard entry; confirm during implementation.
- **Does `deploy-all-prod` call `deploy-operator` reusably?** Confirm the smoke fires on the `deploy-all` path too (the player gap's root cause), not just standalone dispatch.
