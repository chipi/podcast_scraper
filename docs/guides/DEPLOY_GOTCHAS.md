# Deploy gotchas — READ THIS BEFORE (and while) DEPLOYING

**Audience: the agent/human running or debugging a prod deploy.** These are the traps that have
each cost hours (some cost days) and were NOT obvious from the code. If a deploy step is red, find
it here before you form a theory. The governing rule of this whole page:

> **THE ONE RULE THAT KEEPS GETTING RE-LEARNED THE HARD WAY:** prod secrets live in a RAM-only
> directory that **does NOT persist**. Anything that creates a **new container** on the box (a
> pipeline run, the D5 probe, a reprocess/reenrich/gi-repair job) **MUST re-stage the secrets
> immediately before it**, or the container starts with **no credentials** and the gateway returns
> **401**. A gateway 401 means **the key is MISSING from the container, not that the key is wrong.**
> **Re-stage — never re-mint.** Deleting/re-minting a live key on a 401 has burned entire evenings
> (2026-08-18, 2026-08-21) and never once was the fix.

Related: `PROD_RUNBOOK.md`, `docs/adr/ADR-115` (tmpfs secrets), `docs/adr/ADR-142` (LiteLLM gateway),
the canonical re-stage action `.github/actions/stage-prod-secrets/action.yml` (read its header — it
is the best explanation of this whole failure), the gate `scripts/tools/check_prod_secret_staging.py`,
`docs/wip/2026-08-19-session-handover.md` (the LiteLLM 401 post-mortem).

---

## 1. The RAM secrets directory does NOT persist — re-stage before EVERY container creation

This is the trap behind almost every "it was working, now it 401s again."

**How secrets work here (ADR-115, `PODCAST_SECRETS_VIA_FILES=1`):** LLM keys + Sentry DSNs are
delivered to `/dev/shm/podcast-secrets/*` on the box — **RAM only, never written to disk** — and
`compose/docker-compose.secrets.yml` mounts them at `/run/secrets/*`. That "never on disk" property
is deliberate and good. Its **cost** is the thing that bites:

**The directory is not durable.** It is reaped when the SSH session that staged it ends (the box's
`systemd-logind` reaps a uid≥1000 user's `/dev/shm` on logout). So:

- Docker copies each secret **into a container at CREATE time**. Containers made *during* a deploy
  keep their keys — which is why `compose-api-1` stays healthy and **everything looks fine**.
- Any container created **later** — a fresh `docker compose run pipeline-llm`, the D5 probe, a
  reprocess job — finds `/dev/shm/podcast-secrets` **gone**, mounts nothing, and starts with an
  **empty** `/run/secrets/litellm_api_key`. Empty `Bearer` → **401** (or "Deepgram key required",
  etc.). "prod is deployed" and "prod has credentials" are **two different facts.**

**THE RULE:** immediately before any step that creates a container, **re-stage** with
`.github/actions/stage-prod-secrets` (or the inline `podcast-secrets.staged` equivalent). Every real
pipeline workflow already does this (`backfill-audio-prod`, `reprocess-prod`, `reenrich-prod`,
`gi-repair-prod`, `inspect-prod-corpus`). `deploy-prod.yml`'s D5 gateway probe re-stages right before
the probe for exactly this reason — **do not remove that step.**

**The gate — and its blind spot.** `scripts/tools/check_prod_secret_staging.py` (in `ci-fast`) fails
any prod workflow that creates a container but never stages/mounts secrets. It is **file-level**: it
only checks the tokens exist *somewhere* in the file. So a workflow that stages once for step A can
still be broken at step B if B creates a second container *after the reap* — which is exactly how D5
stayed broken while the file "passed." **The gate is necessary, not sufficient:** you must ensure a
re-stage precedes **each** container-creation that runs after a session boundary. (Improving the gate
to be proximity-aware is tracked separately.)

## 2. A gateway 401 = missing secret, not a bad key — re-stage, NEVER re-mint

The single most expensive mistake in this repo's history: a LiteLLM `401` read as "the key is
wrong," a **live production key deleted and re-minted, multiple deploys burned. The key was always
correct** (2026-08-18; repeated in shape 2026-08-21, where a bogus "reload-race retry" was added on
the same false premise and reverted).

**When D5 / the gateway returns 401, in this order — all read-only:**

1. **Is the key even in the container?** From a container that has it:
   `docker exec compose-api-1 sh -c 'wc -c </run/secrets/litellm_api_key'` — 0 bytes = missing =
   you skipped the re-stage (see §1). Fix the staging, not the key.
2. **Does the running stack authenticate right now?**
   `docker exec compose-api-1 sh -c 'K=$(cat /run/secrets/litellm_api_key); curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $K" http://100.124.111.115:4001/v1/models'`
   → **200** means the key is fine and only the *new/probe* container lacked it.
3. **Does the mounted key match the expected one?** `… | sha256sum | cut -c1-12` should be
   **`3b88f1c6ee41`** (`proj-podcast-prod`). Matches + still 401 only from a fresh container ⇒ §1.
4. Only if the key is genuinely mounted, correct-hash, and *still* rejected by the gateway do you
   have a real key problem — and even then, **ask the operator before touching a live key.**

> Do NOT reach for a retry loop, a re-mint, or "the gateway must be reloading." Those are the wrong
> tails. The right question is always **"does the container that failed actually have the secret?"**

Tool: **`scripts/tools/rehearse_gateway_key_gate.sh`** runs the deploy gate's logic against a real
gateway. Use it before changing the D5 gate.

## 3. Secret plumbing details worth knowing

- **GH Secrets are the source of truth**; the on-box RAM files are re-staged from them every time.
  A stale/empty GH secret would stage a bad value — but the *usual* cause of a 401 is §1 (not staged
  at all), not a wrong GH secret. Confirm "mounted + correct hash" before suspecting GH drift.
- **A probe that bypasses the entrypoint** (`--entrypoint python`, etc.) skips `docker/secrets-shim.sh`,
  so `$LITELLM_API_KEY` is never exported — **read the secret FILE directly** (`/run/secrets/...`),
  which is what D5 does.
- **The overlay must be joined** or nothing mounts even when the dir is present:
  `compose/docker-compose.secrets.yml`, joined by the *presence* of `/dev/shm/podcast-secrets`
  (`deploy.sh:38`). Re-staging (§1) is what makes that presence check true at probe time.

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
   `401` → the container is missing the secret; re-stage (§1/§2), do NOT re-mint.
4. Post-deploy: verify per `docs/wip/OBS-MCP-DEPLOY-DAY-RUNBOOK.md` (sha alignment, container health,
   the exposed surfaces, gateway auth 200).
