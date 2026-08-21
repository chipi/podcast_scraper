# Deploy gotchas — READ THIS BEFORE (and while) DEPLOYING

**Audience: the agent/human running or debugging a prod deploy.** These are the traps that have
each cost hours (some cost days) and were NOT obvious from the code. If a deploy step is red, find
it here before you form a theory. The governing rule of this whole page:

> **Verify the actual state on the box before you conclude anything. A red gate is far more often a
> delivery/timing problem than the thing it names.** Do not mutate live prod credentials or infra as
> a *first* response to a failure — that has repeatedly made things worse.

Related: `PROD_RUNBOOK.md`, `docs/adr/ADR-115` (tmpfs secrets), `docs/adr/ADR-142` (LiteLLM gateway),
`docs/wip/2026-08-19-session-handover.md` (the LiteLLM 401 post-mortem), the deploy-day runbook in
`docs/wip/OBS-MCP-DEPLOY-DAY-RUNBOOK.md`.

---

## 1. A gateway / auth **401 is almost never a wrong key** — do NOT re-mint

The single most expensive mistake in this repo's history. A LiteLLM gateway `401` was read as "the
API key is wrong," a **live production key was deleted and re-minted, and three+ deploys were burned
chasing it. The key was always correct** (2026-08-19 handover; repeated in shape on 2026-08-21).

**When you see a 401 from the LiteLLM gateway, in order:**

1. **Is the key even MOUNTED?** `docker exec <container> cat /run/secrets/litellm_api_key` (or check
   it's non-empty). A common 401 cause is the key **not mounted** → the container sent an empty
   `Authorization: Bearer` header → 401. (Fix history: `7cae2aa6` joined the secrets overlay so it
   *was* mounted.)
2. **Does it match the expected key?** `printf %s "$KEY" | sha256sum | cut -c1-12` should be
   **`3b88f1c6ee41`** (the `proj-podcast-prod` key).
3. **Does the gateway accept it right now?**
   `curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" http://100.124.111.115:4001/v1/models`
   → **200** means the key is fine and your 401 was **transient** (see #2, reload race).
4. Only if all of the above genuinely fail do you have a real key problem — and even then, **ask
   the operator before touching a live key.**

Tool: **`scripts/tools/rehearse_gateway_key_gate.sh`** runs the deploy gate's logic against a real
gateway. Use it before changing the D5 gate.

## 2. The **D5 gateway check races the gateway's key reload** (now retried)

The control-plane deploy does `compose up --force-recreate`, which restarts the `litellm` container;
it reloads its virtual keys **asynchronously**. A single post-deploy probe fired right after can see
a **transient 401 on a correct key** (observed 2026-08-21 — the delivered key returned 200 seconds
later). The D5 step now **retries (6× / 12s)** to wait out the reload. If you rewrite it, keep the
retry. The key's *validity* is separately checked pre-deploy by the "gateway-key gate."

## 3. Secrets are **file-mounted (tmpfs), not `.env`** — and re-staged EVERY deploy

`ADR-115` (`PODCAST_SECRETS_VIA_FILES=1`): LLM keys + Sentry DSNs are delivered as **tmpfs files**
(`/dev/shm/podcast-secrets/*` → mounted `/run/secrets/*`), deliberately **kept OUT of `.env`**. The
image entrypoint (`docker/secrets-shim.sh`) exports them into env at container start.

Consequences that bite:

- **GH Secrets are the source of truth; the on-box secret files are OVERWRITTEN on every deploy.** A
  stale/empty GH secret silently overwrites a working on-box key. If a key "worked yesterday, 401
  today," suspect a GH-secret drift, not the gateway.
- **The secrets overlay must be joined** or nothing is mounted: `compose/docker-compose.secrets.yml`.
  `deploy.sh` joins it by the *presence* of `/dev/shm/podcast-secrets`, so the mode can't disagree.
- **A probe that bypasses the entrypoint** (`--entrypoint python`, etc.) skips the shim, so
  `$LITELLM_API_KEY` is never exported — **read the secret FILE directly** (`/run/secrets/...`).
- Keys that are file-mounted (operator API key, LiteLLM master key) **drift silently on rotation** —
  a laptop copy or a GH secret can go stale. Re-read from `/run/secrets/*` on the box to get truth.

## 4. The **prod LiteLLM gateway is the VPS `:4001`**, not the homelab

prod (the pipeline running on the VPS) authenticates against the **VPS** LiteLLM gateway
(`http://100.124.111.115:4001/v1`). The **homelab** gateway is reached **only from the operator's
laptop**. Config: `vars.PROD_LITELLM_API_BASE` + the profile pin (D4 / #1676). If a check points the
pipeline at the homelab gateway, that's the bug (#1676) — not the key.

## 5. Deploy the **published 7-char image sha**, not a workflow-only sha

Images are tagged `:sha-<7>` (7-char short of the commit that the Stack test **publish** job built).
`deploy-all` / `deploy-*` accept `image_sha` with or without the `sha-` prefix but pin `sha-<7>`. If
you hotfix a *workflow* (no image rebuild), there is **no image at that new sha** — deploy the last
**published** sha (from the Stack test run summary), not the workflow-hotfix sha.

## 6. Reusable-workflow gotchas (`deploy-all` calls `deploy-player`/`operator`/`prod`)

`deploy-all` runs the three surface deploys as **reusable-workflow jobs** (`secrets: inherit`), so:

- **The caller's `permissions:` must cover what the called workflows request** (`packages: read` for
  GHCR pulls, `actions: read` for deploy-prod) — otherwise the run **startup_failure**s at load with
  jobs:0 ("workflow file issue"). deploy-all grants `contents/packages/actions: read`.
- **A called workflow inherits the TOP-LEVEL dispatch's inputs.** When deploy-all (dispatched with
  `confirm=DEPLOY_ALL`) calls a surface, that surface sees `inputs.confirm == "DEPLOY_ALL"` (NOT
  empty) and `github.event_name == "workflow_dispatch"` (the caller's). So a "typed-confirm" gate
  **cannot** use `event_name` to detect a call — the surfaces accept `confirm ∈ {SURFACE_DEPLOY,
  DEPLOY_ALL}`.
- **deploy-all runs the surfaces in PARALLEL against the SAME `/srv/podcast-scraper/.git`.** They all
  `git reset --hard` and collided on git's `.git/shallow.lock` (2026-08-21, player failed with
  "Unable to create shallow.lock: File exists"). The refresh is now serialized with a **shared
  `flock`** (`/tmp/podcast-scraper-git-refresh.lock`) in all three (`deploy-player.yml`,
  `deploy-operator.yml`, `deploy.sh`). Keep the lock if you touch the refresh.
- The per-surface **"Emit deploy event to VictoriaLogs"** step is `if: always()` and must stay
  `continue-on-error: true` — an unreachable telemetry sink (e.g. when an earlier step failed before
  the tailnet join) must never turn the deploy red.

## 7. Before you blame code, check the machine — and never say "pre-existing"

- A saturated box returns transient 502/504 from a healthy api. Check `uptime` (load vs 8 vCPU) and
  container state (`docker inspect <c> --format '{{.RestartCount}} {{.State.OOMKilled}}'`) before
  theorizing a code bug.
- **The branch was green before this deploy; red after = this deploy caused it.** Do not label a
  regression "pre-existing" — diff against the pre-deploy state (`docker ps` shas, the prior good
  sha) and own it. (This page exists because that framing wasted trust and time.)

## 8. The deploy checklist (the happy path)

1. **main green**; the Stack test **publish** job pushed all images at ONE `sha-<7>` — record it from
   the run summary (do NOT rely on "newest from main").
2. Dispatch **`deploy-all`**: `confirm=DEPLOY_ALL`, `image_sha=sha-<7>`. One `prod`-environment
   approval releases all three surfaces.
3. Watch it. If a surface fails, open the **step log**, then **SSH and verify the actual state** —
   `401` → delivery/reload (see #1/#2/#3), not a bad key.
4. Post-deploy: verify per `docs/wip/OBS-MCP-DEPLOY-DAY-RUNBOOK.md` (sha alignment, container health,
   the exposed surfaces, gateway auth 200).
