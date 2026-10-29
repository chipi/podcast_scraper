# Operator surface — post-deploy live smoke test

**Status:** shipped; the specs below are in this repository, the deploy pipeline that
runs them is not. The smoke drives
`web/gi-kg-viewer/e2e/live/smoke.live.spec.ts` (unauthed) and `account.live.spec.ts`
(authed creator + role boundary). §8 is the one-time setup and stays manual.
**Surface:** the operator host of a deployment (the `gi-kg-viewer` SPA + `PODCAST_SERVE_OPERATOR_PUBLIC` backend)
**Sibling:** mirrors the player post-deploy smoke. This is the **third surface** (task #18) after the player public + per-user smoke.

---

## 1. What already exists (build on, do not duplicate)

- `web/gi-kg-viewer/playwright.live.config.ts` — targets the operator host (`LIVE_BASE_URL`), **desktop-chrome only**, passes the coming-soon gate via `httpCredentials` (reuses the player's `PLAYER_PREVIEW_PASS` doorman). Login-wall aware.
- `web/gi-kg-viewer/e2e/live/smoke.live.spec.ts` — the **unauthed** smoke: public sees coming-soon; preview users reach the **login wall**; sign-in entrypoint **307s to Google** with the correct HTTPS `redirect_uri` (regression guard for the 2026-07-25 `redirect_uri_mismatch`); `/api/health` 200.
- `web/gi-kg-viewer/e2e/handoff-production/*.spec.ts` — a real-stack "handoff" suite (cold-start, concurrency, cross-entry, digest, filters, hot-state, lifecycle, repeat-click) + `_helpers.ts`. **Tier-3-style** — the closest existing analog to what we're extending.
- `web/gi-kg-viewer/e2e/*.spec.ts` (mock-backed) — dashboard, digest, library, graph, pipeline, auth-roles, person-landing, etc. These run under the local `playwright.config.ts` against mocked APIs.

## 2. THE GAP (what must be wired)

The operator live smoke `smoke.live.spec.ts` only helps if the deploy pipeline runs it after every deploy; without that, a broken operator deploy ships unvalidated and unseen. Wire it into your deploy pipeline (§7).

## 3. Auth model (drives the test account design)

- **Coming-soon gate** (reverse proxy, host-side): basic-auth doorman, same hashes as the player (`PLAYER_PREVIEW_PASS`). `httpCredentials` clears it.
- **Login wall** behind the gate: real Google OAuth in a deployment (`APP_OAUTH_PROVIDER=google`); the mock `?as=` login can never ship. Nothing renders until a session exists.
- **Roles** (`app_roles.py`, totally ordered `listener < creator < admin`):
  - `listener` — player only; **403** on operator routes.
  - `creator` — viewer base: **digest / library / graph** (KG curation).
  - `admin` — everything creator has **plus dashboard / ops / configuration / user management**. Dashboard is `v-if="auth.isAdmin"` in `App.vue`.
- **Operator APIs gated ≥creator** on the public surface (`_OPERATOR_PUBLIC_READ_ROUTES`, each mounted with `require_viewer_access`): `usage_routes, artifacts, index_stats, search, relational, query_activity, explore, corpus_library, corpus_binary, corpus_media, corpus_text_file, corpus_metrics, corpus_coverage, corpus_persons, corpus_digest, corpus_enrichments, corpus_topic_clusters, corpus_storylines, corpus_trending, cil`. All **read-only**. `index_rebuild` / `ops` are NOT mounted on the public operator surface.

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

Same self-mint approach as `account.live.spec.ts` (validated: Node HMAC == `app_sessions.verify`). The operator deploy must use the **same** `PLAYER_APP_SESSION_SECRET` as the player (set `APP_SESSION_SECRET` from it), so the identical mint works.

One seeded account in the deployment (creator role, assigned by an admin — not self-service):

| Env (repo var) | Role | Purpose |
| --- | --- | --- |
| `OPERATOR_SMOKE_CREATOR_USER_ID` | creator | viewer read plane + the "no admin" 403 boundary |

Uses the **stored opaque `u_…` id**, not the email (same gotcha proven on the player: mock hint `smoke-user` stored as `u_52a6b85b8b67de5964bb0429`). Specs **skip cleanly** unless `PLAYER_APP_SESSION_SECRET` + the gate password + the user-id var are set.

## 6. Reuse from existing e2e

- Lift the **assertion shapes** (not the mock wiring) from the mock-backed `digest.spec.ts`, `library.spec.ts`, `dashboard.spec.ts`, `auth-roles.spec.ts` for the authed UI checks.
- `handoff-production/_helpers.ts` — reuse polling/robustness helpers where they fit a live authed context.
- Keep the live specs **read-only + fast** (a smoke, not the full tier-3 walk). The deep authed viewer flow stays in stack-test (mock provider).

## 7. Wiring into the deploy (close the gap)

Wire it into your deploy pipeline as a gating `smoke` step that runs after the deploy:

- Runs only when the deploy succeeded, with `web/gi-kg-viewer` as the working directory.
- `npm ci` → `playwright install --with-deps chromium` → `npm run test:e2e:live`.
- env: `LIVE_BASE_URL=https://<operator host>`, `OPERATOR_PREVIEW_USER`, `OPERATOR_PREVIEW_PASS` (→ `PLAYER_PREVIEW_PASS`), `PLAYER_APP_SESSION_SECRET`, `OPERATOR_SMOKE_CREATOR_USER_ID`, `OPERATOR_SMOKE_ADMIN_USER_ID`.
- Alert on failure (tags `surface:operator` / `stage:post-deploy-smoke`) and upload the Playwright report artifact.
- **A red smoke = a failed operator deploy.**

## 8. One-time operator setup (the "device setup" steps)

1. Seed **one** account (a dedicated `+smoke-creator` Google alias) and have an admin grant it `creator` (roles assigned via user-management).
2. Get its **stored** `u_…` id (from `/api/app/me` while signed in, or the user store) — not the email.
3. Set the repo var: `gh variable set OPERATOR_SMOKE_CREATOR_USER_ID --body 'u_…'`.
4. Confirm the operator domain and `OPERATOR_PREVIEW_USER` vars are set.

## 9. NOT covered / risks / open questions

- **No reversible-write coverage** — the public operator plane is read-only; we assert reads + role boundaries, not mutations. `index_rebuild`/`ops` (the write/admin-heavy surface) are **not** mounted publicly, so they're out of scope for this smoke by design.
- **Local validation limit** — as with the player, the mint can be cross-checked against `app_sessions.verify` and the authed reads against a locally-booted `operator_public` server with `APP_SESSION_SECRET=e2e-secret` + seeded creator/admin users; the **deployed** run can't run until the two accounts are seeded + vars set.
- **Native / desktop operator app** — out of scope (operator is a desktop web surface; no native shell).
- **Admin-only API to assert** — need to pick the specific `app_admin` read route + the exact UI marker for the Dashboard entry; confirm during implementation.
- **Does every deploy path run the smoke?** Confirm it fires when the operator deploy is invoked from an umbrella deploy too, not just on a standalone dispatch.
