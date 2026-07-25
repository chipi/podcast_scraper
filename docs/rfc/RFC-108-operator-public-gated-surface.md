# RFC-108 — Operator viewer as a public, gated surface

**Status:** Proposed
**Date:** 2026-07-25
**Epic:** [#1320](https://github.com/chipi/podcast_scraper/issues/1320)
**Depends on / relates to:** [ADR-114](../adr/ADR-114-shared-multi-tenant-public-edge-caddy.md) (shared edge), [ADR-116](../adr/ADR-116-privilege-split-public-control-api.md) (privilege split — public vs control api), [THREAT_MODEL](../security/THREAT_MODEL.md) (T-01), the player launch (#1262).

## Context

The operator viewer (`gi-kg-viewer`) is served **tailnet-only** (`prod-podcast.tail…`, HTTPS via tailscale). That created a concrete problem: its browser telemetry can't cleanly reach the tailnet-only GlitchTip without either mixed-content hacks or egressing through a **public** tunnel — and today it wrongly egresses through **orrery's** domain (`telemetry.orrerylearn.com/1`).

Rather than patch the telemetry routing, unify the two surfaces: make the operator viewer a **public, gated** surface on `operator.closelistening.app`, reusing the **player's** launch pattern (coming-soon `/preview` SECRET-COOKIE gate → Google OAuth → role), and consume telemetry/analytics via the **public closelistening subdomains** like the player. This retires the tailnet-only serve and the internal-routing problem in one move.

## The load-bearing constraint — T-01

[THREAT_MODEL T-01](../security/THREAT_MODEL.md): the operator `api` container mounts `/var/run/docker.sock` **RW** and holds all 6 LLM keys. Its entire accepted mitigation (D1) is *"conditional on the api being provably private… the public consumer plane is a **separate least-privilege service** (no `docker.sock`, no keys)."* The **pre-public gate** mandates: no socket, no write-scope keys, least-privilege, authN/authZ + rate limiting.

**Therefore "operator viewer public" must NOT mean "expose the privileged operator api."** It means a **new least-privilege public service** that mounts only the safe **read** surface — exactly the player's model, one rung wider.

## Decision (proposed)

### 1. A new least-privilege `operator-public` service
Mirror `docker-compose.player-public.yml`: an app-only-style container with **no `docker.sock`, no LLM keys**, `cap_drop`/`no-new-privileges`/`read_only`, shared **read-only** corpus mount. A new serve posture (`PODCAST_SERVE_OPERATOR_PUBLIC=1`) mounts:
- `health` + `/api/app/*` (as today), **plus**
- a **curated safe subset** of `_OPERATOR_READ_ROUTES`.

### 2. Exposed route surface — curated, not the whole read set
`_OPERATOR_READ_ROUTES` is **not** all read-safe. Split it:

| Expose publicly (browse/query, read-only) | Keep tailnet-only (privileged / compute / write) |
|---|---|
| `search`, `explore`, `relational`, `artifacts`, `query_activity` | `index_rebuild` (rebuilds the index — compute/write) |
| `corpus_*` (library, media, text, metrics, coverage, persons, digest, enrichments, topic/theme clusters, trending, binary) | `ops` (operational controls) |
| `index_stats`, `resilience_routes`, `usage_routes` (read-only views) | anything that triggers a pipeline / mutates the corpus |

The privileged operator plane (`index_rebuild`, `ops`, pipeline triggers, socket ops) **stays on the tailnet-only api, unchanged**.

### 3. Defense in depth — the gate stack
1. **Coming-soon `/preview` SECRET-COOKIE gate** (reuse the player's; own `OPERATOR_PREVIEW_COOKIE`). Outer wall — no cookie, no app.
2. **Google OAuth** (reuse the player app).
3. **Role gate** — operator users are **admin/creator** via `APP_ADMIN_EMAILS` ([`app_roles.py`](https://github.com/chipi/podcast_scraper/blob/main/src/podcast_scraper/server/app_roles.py), `listener < creator < admin`); operator routes require **≥ creator**. A signed-in **listener** is rejected from operator routes.
4. **CF origin-lock** (ADR-118) + edge rate-limit (T-06), same as the player.

### 4. Telemetry → public closelistening
Operator viewer GlitchTip DSN → **`telemetry.closelistening.app/1`** (the podcast's own tunnel, **not** orrery's). Operator Umami → a closelistening subdomain. Drop the orrery domain and the tailnet-internal-relay idea. Sourcemaps (#1271) apply here too.

## Role model

Single identity store, one `role` per user (`app_roles.py`). Player consumers default **listener**; operator users are bootstrapped **admin** (or **creator**) via `APP_ADMIN_EMAILS`. Same Google OAuth, different authorization — the operator subdomain's routes enforce **≥ creator**; the player's `/api/app/*` stay listener-open (as today).

## Threat-model impact

This adds a **new public tenant** (the operator-public service) to the register. It **passes the pre-public gate** by construction: separate least-privilege container (no socket, no keys, hardened), curated read-only route surface, double gate (cookie + OAuth) + role authZ + CF origin-lock + rate limit. The **privileged operator api posture is unchanged** — still tailnet-only, still the basis of T-01's D1. THREAT_MODEL gets a new row (`operator-public`) and the "operator = tailnet-only" note is qualified: *the privileged plane* stays private; a *least-privilege read plane* is public behind the gate.

## Non-goals
- Exposing the privileged operator api (socket/keys/triggers) — explicitly out.
- New operator or player features; changing the player.
- Self-service role assignment (admin still grants roles).

## Consequences

**Positive:** one telemetry model (public closelistening), no tailnet-HTTPS-ingest hack, operator reachable from anywhere behind the gate, reuses proven player machinery.
**Negative:** a new public attack surface (mitigated by the gate stack + least-privilege); a curated route list to maintain (a new operator-read route must be classified expose-vs-private).
**Neutral:** the tailnet-only operator serve can be retired or kept as an admin fallback.

## Alternatives considered
- **Tailnet-internal Caddy relay for the viewer's GlitchTip DSN** (the "tiny fix"): keeps the operator private, same-origin relay to GlitchTip over the tailnet. Correct and smaller, but leaves the operator surface tailnet-only and doesn't unify the surfaces. **Rejected** in favor of the unification (operator's call).
- **Dedicated tailnet-HTTPS GlitchTip host** (`glitchtip.tail…`): homelab-side, cleaner ingest but doesn't make the operator public. Orthogonal.
- **Expose the full operator api behind the gate:** violates T-01's D1 (public plane must be least-privilege). **Rejected.**
