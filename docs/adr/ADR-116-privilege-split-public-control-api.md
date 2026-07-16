# ADR-116: Privilege-split API — public-api (no sock/keys) vs control-api (tailnet), enqueue→drain

- **Status**: Accepted
- **Date**: 2026-07-08
- **Authors**: Marko Dragoljevic, Claude (Opus 4.8)
- **Related**: [ADR-114](ADR-114-shared-multi-tenant-public-edge-caddy.md) (public edge),
  [ADR-115](ADR-115-multi-tenant-secret-delivery-sops-tmpfs-files.md) (secret delivery),
  [ADR-011](ADR-011-secure-credential-injection.md); threat model T-01
- **Tracking**: [#1161](https://github.com/chipi/podcast_scraper/issues/1161) (umbrella).
  Children: **#1163** player-public (first), **#1164** RBAC across `/api/*`, **#1165**
  privilege split (this ADR), **#1166** CORS + rate limit (T-06).

## Context

The `gi-kg-viewer` web app is going public on its own domain. Today it is the **operator
console** — it drives the full privileged `/api/*` surface (`jobs`, `operator-config`,
`feeds`, `index_rebuild`, `ops`). Two facts make a naive "point Caddy at the api" unsafe:

1. **`/api/*` has no per-request authz** — those routes are gated only by mount-time flags
   assuming tailnet trust. The role system (`app_roles`: `listener < creator < admin`,
   `get_admin_user`, `APP_ADMIN_EMAILS`) is wired into `/api/app/*` only.
2. **`/api/jobs` spawns pipelines via `docker.sock`** + provider keys (host-root-equivalent).
   Behind a public RBAC gate, an auth/RBAC bug = host root — reintroducing T-01 and breaking
   ADR-115/D1's premise that the socket stays private.

The operator's decision (2026-07-08): **admin must be able to work from the public domain**
(Option B), not only over the tailnet.

## Decision

Split the backend by **privilege**, not by web-vs-api. `docker.sock` + keys never sit
behind a public endpoint — enforced by **both** network isolation **and** an async queue.

### 1. Two API tiers

- **`public-api`** (internet-facing, behind the Caddy edge): serves read `/api/*`,
  role-gated *non-privileged* admin (`operator-config` edits, `feeds` metadata, user
  admin), and `/api/app/*`. **No `docker.sock`, no provider keys.** Privileged actions
  (pipeline runs) it can only **enqueue**, never execute.
- **`control-api`** (tailnet-only): holds `docker.sock` + keys, runs the **drain loop** that
  picks up queued jobs and executes them via the existing `pipeline_docker_factory`. Never
  internet-reachable.

### 2. Enqueue → drain over the existing job registry (no new infra)

Reuse the file-locked job registry (`.viewer/jobs.jsonl`, `pipeline_job_registry.py`) on the
shared corpus volume as the queue:

- `public-api` **enqueues** a job request (`status=queued`) and serves job **status reads**.
  It runs with `PODCAST_PIPELINE_EXEC_MODE` unset / enqueue-only — no container spawn.
- `control-api` **drains** (the existing APScheduler sweep + registry reconcile) — claims
  `queued` rows under the file lock, executes via the Docker factory, writes status back.
- Both mount the registry; the file lock + concurrency cap
  (`PODCAST_VIEWER_MAX_PIPELINE_JOBS`) already serialize claims. The internet touches the
  queue file (via `public-api`), never the socket.

### 3. RBAC across `/api/*`

Extend the existing `app_roles` model (currently `/api/app/*` only) to `/api/*`:

- **Admin-only** (`Depends(get_admin_user)`): `operator_config`, `feeds`, `index_rebuild`,
  `ops`, and the **enqueue** side of `jobs`.
- **Reads** (`corpus/search/relational/explore/artifacts/cil/...`): decided per surface —
  `creator`/`listener` authed, or anonymous + rate-limited. (Open sub-decision.)
- `public-api` establishes the auth session on `/api/*` (reuse `app_auth`
  `get_current_user`/`get_admin_user`). `control-api` needs no user auth (tailnet + drains
  the queue).

### 4. Container topology

| Container | Surface | Domain | `docker.sock`/keys |
| --- | --- | --- | --- |
| `gi-kg-web` (frontend) | kg/gi web app | public domain | none |
| `player-web` (frontend) | consumer player | public / app | none |
| `public-api` | read `/api/*` + role-gated admin + `/api/app/*` (enqueue only) | public (Caddy) | **none** |
| `control-api` | jobs executor + operator plane; drains the queue | **tailnet only** | `docker.sock` + keys |
| `pipeline` | spawned by `control-api` | — | — |

`public-api` and `control-api` are the **same image** in different run modes (a
`PODCAST_SERVE_ROLE=public|control` flag selecting which routers mount + whether the Docker
factory + drain loop attach), so there is one artifact to build and pin.

## Consequences

**Positive**

- `docker.sock` + keys are unreachable from the internet — protected by network isolation
  **and** the queue boundary, not RBAC alone. Preserves ADR-115/D1.
- Admin works from the public domain (the operator's requirement) with real RBAC.
- Reuses the existing job registry as the queue — no Redis/broker, no new dependency.
- One image, two modes → simple build/pin story.

**Negative**

- Real build: RBAC across `/api/*`, the public/control mode split, and decomposing
  `jobs.py` into enqueue (public) vs execute (control).
- Jobs become strictly async from the public side (enqueue → poll status) — no synchronous
  execution path on the public plane. (Already the shape of the registry.)
- Two api containers to run + monitor instead of one.

**Neutral**

- The tailnet operator console (current `viewer`+`api`) can remain as a `control`-mode
  deployment during migration.

## Sequencing (operator decision, 2026-07-08)

Option B is the **target** for the kg/gi web app, but it is **not** the first thing to
ship. Order:

1. **Player public first (#1163 + #1166).** The consumer player uses `/api/app/*` only,
   **already OAuth-authed**, no `docker.sock`/keys. Take it public on its own domain/app via
   the Caddy edge — the low-risk, near-term launch. Needs the public frontend deploy +
   CORS/rate-limit (#1166) + confirm the served API is the low-privilege set.
2. **kg/gi + operator surface stays tailnet-only (interim Option C)** — unchanged, private —
   **until hardened enough**: i.e. until RBAC across `/api/*` (#1164) and the public/control
   split (#1165, this ADR) land.
3. **Then kg/gi goes public (Option B)** — once #1164 + #1165 are built and verified, expose
   the kg/gi web app on its domain with admin working from public.

Until step 3, `docker.sock`/keys never face the internet because the only public surface is
the player's low-privilege `/api/app/*`.

## Alternatives considered

- **A — one public `api for all` (sock+keys), RBAC-gated.** Rejected: RBAC is then the sole
  barrier to host-root; an auth bug = full compromise; violates D1.
- **C — public kg/gi read-only, admin stays tailnet.** Simpler and safe, but the operator
  needs admin from the public domain, which C does not provide. Rejected for this need
  (remains the fallback if B's build cost is not worth it).
- **External queue (Redis/NATS).** Rejected: new runtime dependency + another service to
  secure on a single VPS; the file-locked registry already provides the needed semantics.
