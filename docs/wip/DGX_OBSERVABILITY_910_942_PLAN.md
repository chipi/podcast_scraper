# DGX observability follow-ups (2026-06-22)

Disposition of the two open DGX-observability issues surfaced during
PR #1039's operator-question batch.

## Status summary

- **#910 — DGX Whisper operational hardening**: closing.
  - Subscope 1 (auto-restart + OOM/Sentry breadcrumb): **PLANNED** in
    this doc — small, tractable.
  - Subscope 2 (GPU health monitoring): **CLOSED — already covered** by
    `dcgm-exporter` + `alloy` + `cadvisor` (running on DGX today). No
    additional code needed. Operator-defined alert thresholds are an
    operational choice, not a code task.
  - Subscope 3 (log shipping): **PLANNED** — verify-then-fix.
  - Subscope 4 (multi-model serving): **SPLIT** to #1046 as its own
    multi-day product decision.
- **#942 — Sentry inside DGX services**: open, fully planned below.
  Cross-repo change (homelab); operator green-lights before execution.

## Plan: #910 subscope 1 — auto-restart + OOM/Sentry breadcrumb

### Today's state

`faster-whisper` runs in docker (`Up 8 days`). Docker's default
`restart: unless-stopped` covers crash restarts. The missing piece
is **OOM kill detection** that doesn't show up as a regular crash:
the kernel OOM-killer SIGKILLs the container's main process,
docker restarts it, and the operator sees a "restart" event with no
indication the cause was memory pressure.

### Proposed change

Two small additions in the homelab compose
(`infra/dgx/faster-whisper/docker-compose.yml` or equivalent):

1. Explicit `restart: unless-stopped` + `mem_limit` set just under
   the GB10 host's GPU memory ceiling (declarative budget rather
   than relying on the kernel). Hitting the limit triggers docker's
   own OOM event, which is greppable in journald.
2. A sidecar journald-tail + Sentry forwarder. Pattern matches
   `kernel: Out of memory: Killed process .* faster-whisper`,
   emits a Sentry event tagged `service: faster-whisper`,
   `dgx_host`, `cause: oom`.

The sidecar lives in the homelab repo; this repo only needs a
runbook delta (`docs/guides/DGX_RUNBOOK.md`) explaining how to
verify the alert fires.

### Acceptance

- Trigger: `docker exec faster-whisper python -c "x=' '*int(50e9)"`
  (or `stress-ng --vm 2 --vm-bytes 200G --timeout 60s`).
- Verify: Sentry event lands within 30s with the expected tags;
  docker restarts the container within 10s; `StartLimitBurst`
  prevents hot loops (>5 restarts in 60s → systemd-like backoff via
  docker's `restart-policy.max-retries` or compose-level cap).

### Estimated effort

Half-day (cross-repo: homelab compose + this repo's runbook).

## Plan: #910 subscope 3 — log shipping verification

### Today's state

`grafana/alloy:v1.17.0` is running on DGX. Alloy ships logs + metrics
to a central stack. **What's not verified**: whether
`faster-whisper` container logs are actually included in alloy's
collection, and whether 5xx responses from
`/v1/audio/transcriptions` are searchable end-to-end.

### Proposed change

Verify-then-fix:

1. Generate a synthetic 5xx from faster-whisper (send a malformed
   audio payload, or temporarily set `MAX_FILE_SIZE=0`).
2. Search the operator's log stack for the request id within
   5 minutes.
3. If found → **#910.3 closes as already-covered.** Document the
   verification + the search query in `docs/guides/DGX_RUNBOOK.md`.
4. If not found → write the alloy config delta to include the
   container in its scrape, then re-verify.

### Acceptance

- `docs/guides/DGX_RUNBOOK.md` has a "How to search for a DGX
  service 5xx in the log stack" section with the verified query.
- Either confirmation that the wiring works, OR the delta that made
  it work landed in the homelab repo.

### Estimated effort

1-2 hours (verification + runbook delta, no code in this repo
unless alloy config needs adjustment).

## Plan: #942 — Sentry inside DGX services

### Today's state

- **Zero** `sentry_sdk` references in the homelab repo
  (`grep -rln sentry_sdk` returns nothing under
  `infra/dgx/pyannote-server/` or anywhere else).
- pyannote-server IS running on DGX (`docker ps`:
  `pyannote | podcast-pyannote:0.1.0 | Up 10 days`).
- Client-side Sentry breadcrumbs exist per ADR-096, but the DGX
  services themselves don't report errors to Sentry.

### Motivating incident

Per #942 body: #926 pyannote deploy hit 7 compat bugs (torch
version, libcudart, weights_only, semver, etc.) — operator knew
about them only via `ssh dgx + docker logs`. With in-service Sentry,
those would have surfaced in the dashboard automatically.

### Scope (homelab-side, executable when greenlit)

**A. pyannote-server FastAPI app** (`infra/dgx/pyannote-server/app.py`)

Add in `lifespan()`:

```python
import os
import sentry_sdk

dsn = os.environ.get("SENTRY_DSN")
if dsn:  # gracefully degrade if env not set (dev / fresh boot)
    sentry_sdk.init(
        dsn=dsn,
        traces_sample_rate=0.1,
        environment=os.environ.get("SENTRY_ENVIRONMENT", "dgx-prod"),
        server_name=os.environ.get("DGX_HOSTNAME", "dgx-llm-1.tail6d0ed4.ts.net"),
        release=os.environ.get("SERVICE_VERSION", "dev"),
    )
    sentry_sdk.set_tag("service", "pyannote-server")
    sentry_sdk.set_tag("dgx_host", os.environ.get("DGX_HOST_TAG", "spark-2c14"))
    sentry_sdk.set_tag("gpu", os.environ.get("GPU_TAG", "GB10"))
```

**B. Dockerfile** (`infra/dgx/pyannote-server/Dockerfile`)

```dockerfile
RUN pip install --no-cache-dir 'sentry-sdk[fastapi]>=1.40,<3.0'
```

**C. Operator runbook delta** (this repo's
`docs/guides/DGX_RUNBOOK.md`)

- Document that `SENTRY_DSN` lives in operator's `~/.env` on DGX
  alongside `HF_TOKEN`.
- Document the separate Sentry project recommendation
  (`podcast-scraper-dgx`) — different SLA shape from pipeline errors
  (DGX-service errors mostly trigger fallback paths; pipeline errors
  block users).
- How to verify: trigger a synthetic ImportError in pyannote-server,
  confirm event lands tagged correctly.

**D. Future vLLM-prod service** (not the autoresearch vLLM —
that's a different slot)

The same pattern goes into `infra/dgx/vllm-prod/app.py` when that
service ships (per `docs/wip/DGX_NEXT_STEPS.md`). Out of scope for
this batch — tracked as a checkbox under #942 acceptance.

### What this does NOT touch

- speaches (faster-whisper transcription service) — per #942 explicit
  scope-out. Mature enough that client-side coverage is sufficient.
- Ollama daemon — Go-based, has its own logging; client-side
  breadcrumbs cover it.
- The autoresearch vLLM — not in #942 scope (its errors are eval-loop
  visible, not user-facing).

### Acceptance

- [ ] `sentry_sdk.init` lives in pyannote-server's lifespan with the
  6 required tags (service, dgx_host, gpu, environment, server_name,
  release).
- [ ] `sentry-sdk[fastapi]` pinned in pyannote-server's Dockerfile.
- [ ] `SENTRY_DSN` documented in this repo's `DGX_RUNBOOK.md`.
- [ ] Test event lands in the right Sentry project with correct
  tags — verified by operator.
- [ ] Same pattern planned for vLLM-prod (checkbox; ships with that
  service).

### Estimated effort

Half-day cross-repo (~30min code in homelab + ~30min runbook delta
in this repo + ~30min operator-attended Sentry verification).

### Cross-repo apply pattern

Per `feedback_cross_repo_apply` (memory rule). When greenlit, I write
the homelab change locally, then deliver paste-ready apply
instructions in this repo (same pattern as
`HOMELAB_COMPOSE_DRIFT_SYNC_2026-06-22.md`).

## Close-out actions

When operator greenlights:

1. Land #910.1 (OOM-Sentry breadcrumb) on this branch — runbook
   delta only; homelab compose delta delivered as apply
   instructions.
2. Run #910.3 (log shipping verification) — capture result in the
   runbook, regardless of whether wiring needed adjustment.
3. Comment on #910 with the close-out:
   - subscope 1: shipped
   - subscope 2: closed as already-covered
   - subscope 3: shipped + verified
   - subscope 4: split to #1046
4. Land #942 (pyannote-server Sentry wiring) on this branch — same
   homelab-edit + this-repo-runbook pattern.
5. Comment on #942 with the close-out once verified.

If operator wants to defer either: keep open, just leave the plan
in place.

## 2026-06-23 update — #942 closing with pyannote-only scope

The post-RFC-097-improvements PR closes #942 with pyannote-server
Sentry shipped + deployed + validated end-to-end on DGX (Sentry
dashboard captured both manual and FastAPI-handler events; runbook
delta in `docs/guides/DGX_RUNBOOK.md` § "In-process Sentry on DGX
services (#942)").

What's NOT done in this PR (the rest of #942's original scope):

- **vLLM-prod Sentry wiring** — gated on the vLLM-prod container
  actually shipping (per `docs/wip/DGX_NEXT_STEPS.md`, that container
  is itself not yet deployed; autoresearch's vLLM is operator-owned
  and out of scope).
- **whisper-server / speaches Sentry wiring** — same Sentry-init
  pattern would apply (`infra/dgx/speaches-gb10/`); not done because
  there's no #942 priority on it yet. Whisper failures already
  surface via the in-pipeline `dgx_fallback_active` breadcrumb path
  (ADR-096), so coverage is partial-but-existing today.

**Operator decision (2026-06-23)**: close #942 with pyannote-only
scope; don't open follow-up issues for vLLM-prod or speaches.
Re-open when either service is in active deployment and Sentry
wiring becomes a real ask.

The shipping pattern is documented in `docs/guides/DGX_RUNBOOK.md`
(env vars, tag conventions, LogQL search recipe), so adding Sentry
to a new DGX service is a copy-paste of:

1. `sentry-sdk[fastapi]>=2.18,<3.0` to the service's Dockerfile.
2. `sentry_sdk.init(...)` block at the top of the FastAPI app
   (mirroring `infra/dgx/pyannote-server/app.py:49-77`), with
   `service`, `dgx_host`, `gpu` scope tags.
3. `SENTRY_DSN` + `SERVICE_VERSION` + `RUN_PROFILE` exposed to the
   container's environment.
4. Verify with a one-shot `sentry_sdk.capture_message(...)` test.

This is small enough to inline-deploy when a service ships;
shouldn't need a tracking ticket.
