# #942 — Pyannote-server in-process Sentry — deployment runbook

**Issue**: [#942](../../issues/942) — Wire DGX services into Sentry — capture in-process errors from pyannote, vLLM-prod
**Status**: **CODE LANDED + DEPLOYED + VALIDATED 2026-06-22.** Awaits
operator-supplied `SENTRY_DSN` env var to start emitting events.
**Created**: 2026-06-22.

## What's already done

Investigation 2026-06-22 confirmed that **the source code change for
#942 already landed in commit `42a17b53`** ("Autoresearch wrap-up
batch: #985 / #987 / #943 / #942 / #923 / #947 / #996") and lives
**in this repo** (NOT cross-repo to homelab as a first read of the
issue suggested). Files:

- `infra/dgx/pyannote-server/app.py` lines 49-77 — Sentry SDK init
  block with all 6 required tags (`service`, `dgx_host`, `gpu`,
  `environment`, `server_name`, `release`).
- `infra/dgx/pyannote-server/Dockerfile` line 73 —
  `sentry-sdk[fastapi]>=2.18,<3.0` pinned.
- Both files ship to DGX via the `make dgx-deploy` (pyinfra) flow
  declared in `infra/dgx/converge/deploy.py`.

This runbook captures what was deployed + how it was validated, so a
future on-call can replicate the verification without re-discovering
the same dead ends.

## Deployment (already executed 2026-06-22)

```bash
# From laptop: push the source to DGX + rebuild + restart.
scp infra/dgx/pyannote-server/app.py \
    infra/dgx/pyannote-server/Dockerfile \
    dgx-llm-1:/tmp/
ssh dgx-llm-1 'sudo cp /tmp/app.py /opt/pyannote-server/build/app.py && \
    sudo cp /tmp/Dockerfile /opt/pyannote-server/build/Dockerfile && \
    cd /opt/pyannote-server && sudo docker compose build && \
    sudo docker compose up -d --force-recreate'
```

The proper-channel equivalent is `make dgx-deploy` from the laptop
(pyinfra runs the same effect plus the rest of the convergent
install). The scp shortcut is fine for incremental redeploys of
just pyannote-server.

## Validation (already executed 2026-06-22)

### 1. sentry-sdk in the running container

```bash
ssh dgx-llm-1 'docker exec pyannote python -c "import sentry_sdk; print(sentry_sdk.VERSION)"'
# 2026-06-22 result: 2.63.0
```

### 2. Graceful-degradation path with no DSN

```bash
ssh dgx-llm-1 'docker logs pyannote 2>&1 | grep -i sentry'
# 2026-06-22 result:
# 2026-06-22 19:16:12 INFO pyannote-server:
#   SENTRY_DSN not set — running without Sentry integration
```

Proves the `if _SENTRY_DSN:` guard works — service runs cleanly
without the env var.

### 3. Full init path with a DSN

```bash
ssh dgx-llm-1 'docker exec -e SENTRY_DSN="https://public@dummy.ingest.sentry.io/1" pyannote python -c "
import os, importlib, app
importlib.reload(app)
import sentry_sdk
hub = sentry_sdk.Hub.current
print(\"client_initialized:\", hub.client is not None)
print(\"dsn:\", str(hub.client.dsn))
with sentry_sdk.configure_scope() as scope:
    print(\"tags:\", dict(scope._tags))
"'
# 2026-06-22 result:
# Sentry SDK initialized for pyannote-server
# client_initialized: True
# dsn: https://public@dummy.ingest.sentry.io/1
# tags: {'service': 'pyannote-server', 'dgx_host': 'spark-2c14', 'gpu': 'GB10'}
```

All 3 required tags set as designed. Note: 3 tags were verified at
the scope level; the other 3 (`environment`, `server_name`,
`release`) are passed as `sentry_sdk.init()` kwargs and appear on
emitted events, not on the scope tag dict.

## What the operator still needs to do

The deployment is live but Sentry won't emit until the operator:

1. **Creates a Sentry project** for DGX-side errors (recommendation:
   call it `podcast-scraper-dgx`, separate from the pipeline
   project because the SLA shape is different — pipeline errors
   block users; DGX errors mostly trigger fallback paths per
   ADR-096).

2. **Adds `SENTRY_DSN` to `/home/markodragoljevic/.env`** on DGX,
   alongside `HF_TOKEN`:

   ```bash
   SENTRY_DSN=https://<dsn>@<org>.ingest.sentry.io/<project-id>
   # Optional overrides — defaults are fine for prod:
   # SENTRY_ENVIRONMENT=dgx-prod
   # SENTRY_TRACES_SAMPLE_RATE=0.01
   # DGX_HOST_TAG=spark-2c14
   # SERVICE_VERSION=pyannote-0.1.0
   ```

3. **Restart pyannote** (compose picks up the env_file change on
   recreate):

   ```bash
   ssh dgx-llm-1 'cd /opt/pyannote-server && sudo docker compose up -d --force-recreate'
   ```

4. **Trigger a synthetic test event** to confirm wire-up:

   ```bash
   ssh dgx-llm-1 'docker exec pyannote python -c "
   import sentry_sdk
   sentry_sdk.capture_message(\"test event from pyannote-server\", level=\"info\")
   "'
   ```

   Within ~30 seconds, the Sentry dashboard for the new project
   shows the event tagged `service=pyannote-server`,
   `dgx_host=spark-2c14`, `gpu=GB10`, `environment=dgx-prod`,
   `server_name=dgx-llm-1.tail6d0ed4.ts.net`,
   `release=<SERVICE_VERSION>`.

## Acceptance (#942 checklist)

- [x] `sentry_sdk.init` in pyannote-server's lifespan with all 6 tags
- [x] `sentry-sdk[fastapi]>=2.18,<3.0` pinned in Dockerfile
- [x] Code deployed to DGX
- [x] Container rebuilt with sentry-sdk available
- [x] Graceful-degradation verified (no DSN → no-op log + clean run)
- [x] Full init verified with a dummy DSN (tags + client initialize)
- [x] `SENTRY_DSN` documented in `DGX_RUNBOOK.md` — operator-driven
- [ ] **OPERATOR STEP** — Sentry project created + DSN added to env
- [ ] **OPERATOR STEP** — Test event lands in the right project with
  correct tags (verifies the full HTTPS round-trip, which my dummy-
  DSN test does NOT exercise)
- [ ] Same pattern queued for vLLM-prod when that service ships
  (separate checkbox; not in this PR)

## Rollback

If Sentry init causes a problem (highly unlikely — guarded behind
`if _SENTRY_DSN:`), unset `SENTRY_DSN` in `~/.env` and restart the
container. The guard turns the entire init block into a no-op.

## Out of scope

Per #942 explicit scope-out:

- **speaches (faster-whisper)** — we don't control its source.
  Client-side breadcrumbs cover its failure modes.
- **Ollama daemon** — Go-based, separate logging surface.
- **autoresearch vLLM** — eval-loop-visible (not user-facing).
- **vLLM-prod** — same pattern lands when that service ships per
  DGX_NEXT_STEPS.md.
