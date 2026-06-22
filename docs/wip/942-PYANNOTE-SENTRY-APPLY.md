# #942 — Pyannote-server Sentry wiring — cross-repo apply doc

**Issue**: [#942](../../issues/942) — Wire DGX services into Sentry — capture in-process errors from pyannote, vLLM-prod
**Status**: Plan ready; cross-repo apply pending operator green-light.
**Created**: 2026-06-22

This doc is the paste-ready apply surface for the homelab-side change.
The pattern matches `HOMELAB_COMPOSE_DRIFT_SYNC_2026-06-22.md`:
this repo carries the design + the runbook delta;
the actual service code lives in the operator's
`agentic-ai-homelab` repo.

## What lands where

| Change | Repo | Path |
| --- | --- | --- |
| Sentry init in pyannote-server FastAPI lifespan | `agentic-ai-homelab` | `infra/dgx/pyannote-server/app.py` |
| `sentry-sdk[fastapi]` pip install | `agentic-ai-homelab` | `infra/dgx/pyannote-server/Dockerfile` |
| Operator runbook — `SENTRY_DSN` placement + verification steps | THIS REPO | `docs/guides/DGX_RUNBOOK.md` (delta in this PR) |

## Why this shape

Per [ADR-096](../adr/ADR-096-dgx-spark-prod-primary-with-fallback.md), the
pipeline emits **client-side** breadcrumbs (`dgx.fallback`,
`dgx_fallback_active`) when a DGX call fails. Those tell us "the
client saw the failure" — they don't tell us anything about WHY the
DGX service errored internally.

The #926 pyannote deploy hit 7 distinct compat bugs (torch / libcudart /
weights_only / semver / ...) — every one of them only visible to the
operator via `ssh dgx + docker logs`. With in-service Sentry, those
would have surfaced in the dashboard automatically. This commit closes
that gap for pyannote-server now; the same pattern lands in
`vllm-prod` when that service ships (see DGX_NEXT_STEPS.md).

## Step 1 — apply in the homelab repo

In `~/agentic-ai-homelab/`:

### `infra/dgx/pyannote-server/app.py`

Add to the FastAPI `lifespan()` context (early — before any model
loads, so import errors during boot also report):

```python
# RFC-097-adjacent — #942 DGX in-process Sentry. Capture errors at
# the SERVICE side, not just the client side (ADR-096 already covers
# client-side breadcrumbs via dgx.fallback / dgx_fallback_active).
import os

import sentry_sdk

dsn = os.environ.get("SENTRY_DSN")
if dsn:  # gracefully degrade if env not set (dev / fresh boot)
    sentry_sdk.init(
        dsn=dsn,
        traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
        environment=os.environ.get("SENTRY_ENVIRONMENT", "dgx-prod"),
        server_name=os.environ.get(
            "DGX_HOSTNAME", "dgx-llm-1.tail6d0ed4.ts.net"
        ),
        release=os.environ.get("SERVICE_VERSION", "dev"),
    )
    sentry_sdk.set_tag("service", "pyannote-server")
    sentry_sdk.set_tag("dgx_host", os.environ.get("DGX_HOST_TAG", "spark-2c14"))
    sentry_sdk.set_tag("gpu", os.environ.get("GPU_TAG", "GB10"))
```

The six required tags (per #942 acceptance):
`service`, `dgx_host`, `gpu`, `environment`, `server_name`, `release`.

### `infra/dgx/pyannote-server/Dockerfile`

Add to the pip-install block:

```dockerfile
RUN pip install --no-cache-dir 'sentry-sdk[fastapi]>=1.40,<3.0'
```

Pin range explanation: `>=1.40` for FastAPI integration; `<3.0`
guards against the v3 breaking-changes wave that's expected mid-2026.

### Compose file — already injects env

The existing `agentic-ai-homelab/infra/dgx/pyannote-server/docker-compose.yml`
uses `env_file:` to pull in operator-supplied env vars. No compose
change is needed — once `SENTRY_DSN` lives in the operator's
`~/.env` on DGX (next step), the container picks it up.

## Step 2 — operator env on DGX

On the DGX host, add to `~/.env` (the same file that holds
`HF_TOKEN`):

```bash
SENTRY_DSN=https://<your-dsn>@<org>.ingest.sentry.io/<project-id>
# Optional overrides — defaults are fine for prod:
# SENTRY_ENVIRONMENT=dgx-prod
# SENTRY_TRACES_SAMPLE_RATE=0.1
# DGX_HOST_TAG=spark-2c14   # change when the box changes
# SERVICE_VERSION=pyannote-0.1.0  # bump on release tag
```

**Recommendation per #942**: use a SEPARATE Sentry project for
DGX-side events (e.g. `podcast-scraper-dgx`) rather than reusing the
pipeline project. DGX-service errors mostly trigger fallback paths;
pipeline errors block users. Different SLAs, different on-call
shape, different alerting cadence.

## Step 3 — rebuild + restart pyannote-server

```bash
cd ~/agentic-ai-homelab/infra/dgx/pyannote-server
docker compose build
docker compose up -d --force-recreate
docker logs pyannote --since 1m | grep -i sentry
# expect: "sentry_sdk._client: enabled" or similar startup line
```

## Step 4 — verify (operator-driven, ~2 min)

```bash
# Trigger a synthetic ImportError to confirm event lands tagged correctly:
docker exec pyannote python -c "import nonexistent_module_for_sentry_test"
```

In the Sentry dashboard, within ~30 seconds:

- Event appears tagged `service=pyannote-server`, `dgx_host=spark-2c14`,
  `gpu=GB10`, `environment=dgx-prod`.
- `server_name` is the tailnet FQDN.
- Release is whatever `SERVICE_VERSION` was set to.

## Acceptance (#942 checklist)

- [ ] `sentry_sdk.init` in pyannote-server's `lifespan()` with all 6 tags.
- [ ] `sentry-sdk[fastapi]` pinned in pyannote-server's Dockerfile.
- [ ] `SENTRY_DSN` documented in this repo's `DGX_RUNBOOK.md`.
- [ ] Test event lands in the right Sentry project with correct tags.
- [ ] Same pattern planned for vLLM-prod (when that service ships;
  separate checkbox on the issue).

## Out of scope

- **Speaches (faster-whisper)** — we don't control its source; FastAPI
  middleware patch is possible but probably not worth it (mature
  enough that client-side coverage suffices). Per #942 explicit
  scope-out.
- **Ollama daemon** — Go-based, separate logging surface; client-side
  breadcrumbs cover it.
- **autoresearch vLLM** — out of #942 scope (errors there are
  eval-loop-visible, not user-facing).

## Rollback

If the Sentry init causes pyannote-server to fail boot (unlikely —
guarded behind `if dsn:`), unset `SENTRY_DSN` in `~/.env` and
restart the container. The `if dsn:` guard makes the init a
no-op when the env var is empty.
