# #1161 — Public vs private API separation: surface model, RBAC gap, container topology

- **Status**: Reassessment v2 (analysis; no code changed — reassess → propose → operator approves → implement)
- **Date**: 2026-07-08
- **For**: [#1161](https://github.com/chipi/podcast_scraper/issues/1161) (validates the D1
  accept of the RW `docker.sock` — `docs/security/THREAT_MODEL.md` T-01)
- **Note**: supersedes the first-pass framing (which wrongly treated `/api/*` as
  "operator-only, keep private"). Corrected after operator reframe.

## The three surfaces (corrected)

| # | Surface | Frontend | API it uses | Auth model | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | **Consumer player** | mobile/PWA (`web/learning-player`) | **`/api/app/*` only** | OAuth user (listener/creator) — **exists** | clear / settled |
| 2 | **kg/gi web app** | `web/gi-kg-viewer`, its own domain | **full `/api/*`** incl. privileged | **none on `/api/*` today** | the problem |
| 3 | **Operator control plane** | (drives from #2 today) | `jobs` → `docker.sock` + keys | flag-gated, tailnet-trusted | must stay private |

**Key correction:** the `gi-kg-viewer` is today the **operator console**. It calls the
*full* `/api/*` surface including `jobs` (30×), `operator-config` (11×), `feeds` (10×),
`ops`, `index` (rebuild). Taking it public on its own domain with roles (admin sees
operator controls; `creator`/`listener` see read/consume) means **per-endpoint RBAC is
the only gate — there is no network boundary once it's public**.

## The two gaps the first pass missed

### Gap 1 — `/api/*` has NO per-request authorization

`jobs`, `operator_config`, `feeds`, `index_rebuild`, `ops`, `enrichment` = **0**
`Depends`/`admin`/`403` refs. They are gated **only** by mount-time flags
(`enable_jobs_api`, …) on the assumption of tailnet trust. The role system
(`app_roles`: `listener < creator < admin`, `get_admin_user`, `at_least`, admin bootstrap
via `APP_ADMIN_EMAILS`) is wired into **`/api/app/*` only**. Taking the kg/gi web app
public requires **building RBAC across `/api/*`** — real work, not a config toggle.

### Gap 2 — `docker.sock` behind a public RBAC gate = T-01 reintroduced

`/api/jobs` spawns pipelines via `docker.sock` (host-root-equivalent) and needs provider
keys. If the public kg/gi admin can trigger jobs, the backend serving that endpoint holds
`docker.sock` **and is internet-reachable** — protected *only* by RBAC. An auth/RBAC bug
= host root. This breaks D1's premise ("accept the RW socket *because the api stays
private*"). **RBAC alone must never be the only thing between the internet and
`docker.sock`.**

## Container topology — "web app + api for all"?

The right axis is not web-vs-api; it's **privilege-vs-exposure**. A single "api for all"
that is public *and* holds `docker.sock`+keys is exactly the thing to avoid.

- **Option A — one public `api for all` (sock+keys), RBAC-gated.** ✗ Rejected: RBAC is the
  sole barrier to host-root; reintroduces T-01; violates D1.
- **Option B — split the backend by PRIVILEGE (recommended).**
  - **`public-api`** — read `/api/*` + role-gated *non-privileged* admin (user admin,
    operator-config edits, feeds metadata) + `/api/app/*`. **No `docker.sock`, no keys.**
    Privileged actions (pipeline runs) are **enqueued**, not executed.
  - **`control-api`** — private/tailnet, holds `docker.sock` + keys, **drains the job
    queue** and runs pipelines. Never internet-reachable.
  - Flow: public admin clicks "run" → `public-api` writes a job request to a queue →
    `control-api` (private) picks it up + executes. The internet never touches `docker.sock`.
- **Option C — public kg/gi read-only; admin/operator stays tailnet.** Simplest & safest:
  the public web app is `listener`/`creator` (read/consume); admin/operator functions stay
  on the tailnet console (current `viewer`+`api` unchanged). Cost: no admin management from
  the public domain.

### Recommended topology (Option B)

| Container | Surface | Domain | `docker.sock`/keys |
| --- | --- | --- | --- |
| `gi-kg-web` (frontend) | kg/gi web app | public domain | none |
| `player-web` (frontend) | consumer player | public / app | none |
| **`public-api`** | read `/api/*` + role-gated admin (non-priv) + `/api/app/*` | public (behind Caddy) | **NONE** |
| **`control-api`** | jobs executor + operator plane | **tailnet only** | `docker.sock` + keys |
| `pipeline` | spawned by `control-api` | — | — |

So "2 containers" becomes **frontends per surface + TWO api tiers**: `public-api`
(no sock) and `control-api` (private sock). The privilege boundary is enforced by **both**
network (control-api tailnet) **and** an async queue (privileged execution never sits
behind a public endpoint).

## Work this implies (updated #1161 scope)

1. **Build RBAC across `/api/*`** — wire the existing `app_roles` system into the read +
   admin endpoints (admin for config/feeds/user-admin; `creator`/`listener` for reads;
   decide anon vs authed for public reads).
2. **Split the backend** — `public-api` (no sock/keys) vs `control-api` (sock/keys,
   tailnet); move pipeline-spawn behind an **enqueue → drain** boundary so no privileged
   execution is ever reachable from a public endpoint.
3. **Per-surface frontends** on their domains via the Caddy edge (ADR-114); the
   operator/tailnet console stays as-is.

## The key fork for the operator

**Does admin need to trigger privileged actions (pipeline runs / config) from the PUBLIC
kg/gi web app?**
- **Yes** → Option B (public-api enqueues, control-api drains). More build, but admin works
  from the public domain safely.
- **No / tailnet-admin is fine** → Option C (public web app is read-only; admin stays on
  the tailnet). Far simpler; the sock/keys never go near the public plane at all.

Everything downstream (how much RBAC to build, whether there's a job queue, how many
containers) hangs on this one answer.
