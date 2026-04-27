# RFC-081: Pre-prod Environment, Minimal Observability, and Mobile Control Plane

- **Status**: Draft
- **Authors**: Marko + Claude
- **Created**: 2026-04-27
- **Domain**: Infrastructure / DevOps / Observability
- **Related RFCs**:
  - [RFC-079](RFC-079-full-stack-docker-compose.md) — full-stack compose topology (the artefact this RFC deploys)
  - [RFC-078](RFC-078-ephemeral-acceptance-smoke-test.md) — CI ephemeral smoke (the test side of the same compose stack)
  - [RFC-077](RFC-077-viewer-feeds-and-serve-pipeline-jobs.md) — viewer feeds + jobs API (the runtime surface)
- **Tracking**:
  - Phase 1 umbrella (Codespaces deploy + devcontainer + corpus backup + secrets): TBD
  - Phase 1 core prep (slim profile preload + `.env` leak assessment + log hygiene gate / audit): TBD
  - Grafana Cloud observability wiring: TBD
  - Sentry integration (FastAPI + Vue): TBD
  - Critical + integration + e2e test suite review (parallel pre-work): TBD
- **Implementation**: not started; this RFC is the design.
- **Scope boundary**: this RFC covers **Phase 1 (Codespaces) only**. Always-on hosting, VPS provisioning, Tailscale, and the long-running corpus story are deferred to a follow-up RFC ("RFC-08X: Always-on pre-prod") to be drafted once Phase 1 ships.

## Abstract

Today the project has a perfectly good local stack (`make stack-test-ml-ci`)
and a CI gate (Stack-test workflow) but **no always-on environment**. There
is nowhere to point a phone, paste a feed URL, and watch a real pipeline
run from outside the laptop.

This RFC proposes a Phase 1 pre-prod environment that:

1. **Reuses the same Docker Compose topology** from RFC-079 — no new
   image surface, no new code paths.
2. **Auto-deploys on every push to `main`** that passes Stack-test.
3. **Runs entirely on GitHub Codespaces** — no third-party host,
   GitHub-native auth, $0/mo within the free-tier envelope.
4. **Streams metrics + logs into hosted Grafana Cloud** (free tier) so
   the operator can monitor everything from a phone without running a
   self-hosted Prometheus stack.
5. **Sits behind GitHub-authenticated forwarded ports** — no open
   inbound ports on the public internet, single GitHub login from any
   device.
6. **Notifies a Slack channel** on deploy / pipeline / cost events,
   making the same surface usable from home-automation hooks (iOS /
   Android push, optional Home Assistant webhook later).

Phase 2 — an always-on host (likely a small VPS) with its own auth
wall and persistence story — is **out of scope** for this RFC and
will be specified in a follow-up RFC after Phase 1 has shipped and
the operator has real signal on what does and doesn't translate.

## Problem Statement

Current state:

- **Local laptop**: `make serve-api`, `make serve-ui`, occasionally
  `make stack-test-ml-ci`. Fine for development, gone when the laptop
  closes.
- **CI Stack-test**: spins up the full compose stack on every main push,
  runs Playwright, tears down. Validates correctness, but the
  environment is dead by the time the workflow finishes.
- **No always-on URL** for: trying a new feed against the real ingestion
  path, observing GIL/KG output drift over time, sharing a graph with a
  collaborator, or hitting the pipeline from a phone "while watching TV
  on the couch".

Pain points the RFC needs to address:

1. **No place to play with the stack from outside the laptop.** Tooling
   for pre-prod / staging is the missing rung between "works on my
   machine" and "works in production".
2. **No persistent observability.** When a pipeline run misbehaves on
   CI we get a 5-minute log and a teardown; if it misbehaves
   *between* CI runs, we have nothing.
3. **No mobile-friendly control plane.** GitHub mobile shows workflow
   status, but the api / viewer / pipeline jobs are not reachable from
   the phone without ad-hoc tunnels.
4. **No automation hook.** No way to wire "pipeline failed" or
   "yesterday's cost broke $X" into Slack / Home Assistant / the
   operator's existing notification habits.

## Goals

1. **Single-host deployment of the RFC-079 stack** that survives between
   CI runs and across reboots.
2. **Push-to-deploy from `main`**: a passing Stack-test triggers a
   deploy job; no manual SSH.
3. **Container image registry** (GHCR) so deploys are `docker compose
   pull` not `docker compose build` — fast, repeatable, and the same
   bits the CI gate validated.
4. **Minimal observability** that covers metrics + logs + alerts
   without a self-hosted Prometheus/Grafana/Loki stack.
5. **One-login control plane** reachable from desktop and phone, with
   no public inbound ports on the host.
6. **Slack notifications** on deploy + pipeline + cost events, with a
   structure that lets future Home Assistant / phone-automation hooks
   subscribe to the same source of truth.
7. **Total cost: $0/month** at typical hobby load. A clear ceiling
   (and a clear paid-tier exit door) when usage exceeds free.

## Non-Goals

- Multi-region / HA — single Codespace is fine.
- Multi-tenant — the operator and at most a handful of collaborators
  (collaborators reach the env via the same Codespaces forwarded URL
  with GitHub auth, no separate identity layer).
- Replacing CI Stack-test — pre-prod is **downstream** of Stack-test,
  not a replacement.
- Always-on hosting. Codespaces auto-suspends; that is acceptable for
  Phase 1 hobby use. Always-on is the trigger for the follow-up RFC.
- Open public endpoints. Everything sits behind GitHub auth.
- Long-term log retention beyond the free tier (~14 days for Grafana
  Cloud, 90 days for Slack Free).
- Production hardening (log shipping to S3, secrets rotation, blue/
  green deploys). Pre-prod is "stable enough to play with", not "five
  nines".

## Use Cases

1. **Operator on couch with phone**: opens the bookmarked Codespaces
   forwarded URL on iOS, signs in once via GitHub, drops a new feed
   URL, watches the pipeline run, sees the resulting GI/KG graph.
2. **Operator monitoring**: opens Grafana Cloud app on phone, sees
   recent pipeline cost, today's job latency p95, recent error logs.
3. **Slack alert**: pipeline job exits with `exit_code_1`; operator
   gets a Slack push notification on the phone with a link to the
   failed job's log in Grafana.
4. **Pre-merge UAT**: collaborator wants to validate a feature against
   real corpus data before approving a PR; operator grants codespace
   access via GitHub repo permissions, collaborator opens the same
   forwarded URL.
5. **Home automation hook (future)**: Home Assistant subscribes to the
   same job-state webhook surface; out of scope for Phase 1 but the
   webhook surface this RFC defines is the contract a future
   automation can subscribe to.

## Design

The design is layered: **deployment → observability → control plane →
notifications**. Each layer is otherwise independent and can be
implemented as its own sub-step.

### Layer 1 — Deployment (GitHub Codespaces)

**Why Codespaces:** the operator already has GitHub credentials, the
same SSO controls deploy + access + secrets, and there is no third-
party vendor sign-up. This is the cheapest possible "does the pipeline
run end-to-end and can I reach it from my phone?" loop.

**Spec match (with the published image set restricted to api / viewer
/ pipeline-llm — the heavy `pipeline-ml` is never published, so disk
pressure on the codespace is much lower than the raw 15.5 GB CI
image suggests):**

| Codespaces detail | Value | Fit for our stack |
| --- | --- | --- |
| Free hours / month (Personal) | 120 core-hours; 4-core spends 2× → ~60 hrs/mo | Enough for hobby UAT |
| Machine | 4 vCPU / 16 GB RAM / 32 GB SSD | RAM and disk fit comfortably with the slimmer published image set (~2 GB total) |
| Storage | 15 GB-month free (workspace + prebuilds combined) | Enough; prune images when stale |
| Port forwarding auth | **Private by default**, GitHub auth required | Acts as the control plane — no extra VPN needed |
| Mobile | Browser-only via `vscode.dev` / `github.dev` (no native app) | Acceptable; the viewer + Grafana are HTTPS, both render in mobile Safari |
| Idle stop | 30 min default, max 240 min | Fine — auto-suspend is the cost floor |
| Wake from Actions | `gh codespace start <name>` or `POST /user/codespaces/{name}/start` | Allows scheduled wakes / on-deploy refresh |
| Prebuilds | Build the devcontainer + image pull on push to main, cached | Drops cold start from ~15 min to ~1-2 min |
| Auth scope for automation | Needs PAT / fine-grained token with `codespaces` scope (the default `GITHUB_TOKEN` does not work) | One-time setup |

**Codespaces wiring:**

- `.devcontainer/devcontainer.json` runs `docker compose up -d` for the
  api + viewer + mock-feeds (no pipeline service in the always-running
  set; pipeline is launched on demand by the api job factory).
- `.devcontainer/prebuild.yml` triggers prebuilds on every push to
  `main` so the 15.5 GB pipeline image is pre-pulled into the prebuild
  cache. Cold-start cost on a fresh codespace drops dramatically.
- A new GHA `deploy` job (gated on Stack-test green) triggers a
  rebuild-and-wake of the named pre-prod codespace via the
  `gh codespace rebuild --full` CLI, using a fine-grained PAT stored
  as a GHA secret.
- Operator workflow: open the codespace from
  `github.com/chipi/podcast_scraper/codespaces`, browse the forwarded
  port for the viewer, paste a feed URL, watch the pipeline run.

**Caveats Phase 1 must accept:**

- Auto-suspend means it's "always-resumable" not "always-on". Phone
  notifications via Slack still arrive when the codespace is asleep
  (because GHA + Grafana Cloud are independent of the host); the
  viewer just takes ~30 s to wake when tapped.
- 60 hrs/mo at 4-core ≈ 2 hrs/day. Heavy real-feed playtesting will
  blow through this — when that happens, the follow-up RFC takes over
  with an always-on host.
- 32 GB disk is tight if the published image set ever grows back
  toward including ML models. With the cloud-only published set
  decided below, this is comfortable.

#### Image registry + published image set

GHCR (`ghcr.io/chipi/podcast-scraper-stack-*`). Free for public images,
native GitHub auth, same package surface as the source repo.

**Critical scoping decision: only the cloud variant is published.** The
ML pipeline image (`pipeline-ml`, ~15.5 GB with HuggingFace cache baked
in) is **not** redistributed. It is built on CI runners for Stack-test,
used during the test run, and discarded. This sidesteps two real risks
in one move:

1. **Llama 3.2 license** — `DISLab/SummLlama3.2-3B` (the heavy local
   summarization model used by the `airgapped` profile) carries the
   Llama 3.2 Community License, which restricts redistribution without
   attribution. By not publishing the image, the question doesn't
   arise.
2. **Image size + pre-prod scope** — pre-prod runs against `cloud_thin`
   (Gemini Flash), which targets a 100-200 episode test corpus and is
   ~30× faster wall-time than local-ML on a 4-core CPU. The published
   `pipeline-llm` image is ~1 GB (cloud SDKs, no local models), which
   fits the Codespaces 32 GB SSD comfortably.

| Image | Published to GHCR? | Used where |
| --- | --- | --- |
| `api` | ✅ yes | Codespaces pre-prod (cloud_thin profile) |
| `viewer` | ✅ yes | Codespaces pre-prod |
| `pipeline-llm` | ✅ yes | Codespaces pre-prod (cloud_thin profile, no local ML) |
| `pipeline` (ML variant) | ❌ no | stack-test on CI runners only; local dev where redistribution doesn't apply |

**Tags on published images:**

- `:main` — latest passing main (auto-promoted when Stack-test goes
  green).
- `:sha-<short>` — every successful main build, retained for rollback.
- `:pr-<num>` — optional, for collaborator UAT.

**Image hygiene gates** (CI assertions before any push):

- `.dockerignore` excludes `.env*`, `**/.env*`, `*.pem`, `*.key`, plus
  the existing fixture-related paths. This is the static guard.
- A workflow assertion runs `docker run --rm <img> sh -c 'find / -name
  ".env*" 2>/dev/null'` against each candidate image and fails the
  publish job if any path matches. Belt-and-suspenders against the
  static guard drifting.
- License audit on the `pipeline-llm` image is trivial: it bakes in
  Python SDKs only (OpenAI, Gemini SDK, Anthropic SDK, Mistral) — all
  permissive licenses. No attribution-strings-attached models.

Codespaces pulls these three images into its devcontainer at boot;
the same tags will be reused unchanged by the follow-up always-on
RFC, so there is no re-tagging churn when Phase 2 lands.

### Layer 2 — Observability (minimal)

Three signal types, three free hosted services, zero self-hosted
storage:

| Signal | Source | Pipe | Sink | Retention (free) |
|---|---|---|---|---|
| **Metrics** | api `/metrics` (FastAPI Prometheus exporter) + node-exporter on host | Grafana Agent (single binary) | **Grafana Cloud** | 14 days, 10 k active series |
| **Logs** | `docker compose logs` for api + viewer + pipeline | Grafana Agent (Promtail mode) | **Grafana Cloud Loki** | 14 days, 50 GB/month ingest |
| **Healthchecks** | external probe of `/api/health` | Grafana Synthetic Monitoring | Grafana Cloud | 30 day uptime history |

Why **Grafana Cloud** instead of self-hosted Prometheus + Grafana +
Loki:

- Zero containers added to the compose stack — the agent is one extra
  service, ~30 MB RAM.
- Single login, single dashboard URL, mobile app exists.
- The free tier (14 day retention, 10 k series, 50 GB logs) is more
  than enough for hobby load.
- We can graduate to self-hosted later by pointing the agent at a
  different endpoint; nothing in the stack changes.

The api already has `/metrics` (or trivially can — FastAPI
prometheus middleware is a 3-line add). Pipeline runs as one-shot
containers; the agent picks up their stdout/stderr via the Docker
logging driver, no app changes needed.

> **Detailed sink contract deferred to a follow-up issue.** The exact
> set of metric names + cardinality budget + log streams + log
> hygiene gates (no leaking secrets / PII / copyrighted-fragment text
> via Loki) is non-trivial and warrants its own review pass. **A
> separate GitHub issue tracks the detailed observability spec** — see
> the *Tracking* header. RFC-081 itself only commits to the three
> sink categories (metrics / logs / healthchecks), the free-tier
> targets, and the principle that nothing ships off-host until the
> log hygiene audit is signed off. Phase 1B's acceptance is gated on
> that issue closing first.

A third sink — **Sentry free tier** (5 k errors/mo, 10 k transactions,
1 user) — is also planned for error tracking + performance traces.
Complementary to Grafana metrics + logs: Sentry captures stack traces
on unhandled exceptions and JS errors with full source context.
Wiring (`sentry-sdk[fastapi]` for api + `@sentry/vue` for viewer) is
tracked as a separate GitHub issue under the same parent.

### Layer 3 — Control plane

The auth wall changes shape between phases.

**GitHub-native auth on forwarded ports.** Codespaces port forwarding
is private by default; reaching `viewer:80` from a phone requires the
operator's GitHub login. No third-party VPN, no DNS plumbing, no
inbound port. The downside is mobile UX: `vscode.dev` / `github.dev`
work in mobile Safari but the GitHub mobile app does not open
Codespaces. For Phase 1 hobby use that is acceptable — the operator
bookmarks the forwarded URL and signs in once per session.

The follow-up RFC for an always-on host will introduce its own auth
wall (likely Tailscale or Cloudflare Tunnel + Access). That choice is
deferred until Phase 1 is operational and we have signal on what the
operator actually wants from a remote viewing surface.

### Layer 4 — Notifications + mobile / home automation

A single notification spine, multiple subscribers:

```
GitHub Actions ─┐
                ├─► Slack channel #pipeline-events  ─► iOS / Android push
Grafana Cloud ──┤                                   ─► Home Assistant webhook
api `/api/jobs` ┘                                   ─► future automations
   (job state webhook)
```

**Slack** is the central fan-out because:

- The operator already uses it; one app fewer to install.
- Per-message threading + mute schedules + rich formatting mature.
- Both GHA and Grafana Cloud have first-class Slack senders.
- Home Assistant has a `slack_event` integration for inbound webhooks
  if the operator wants the same events to hit a smart light, a
  speaker, or a Shortcuts automation.

Event taxonomy (initial, expandable):

| Event | Source | Severity | Channel |
|---|---|---|---|
| Deploy succeeded / failed | GHA `deploy` job | INFO / ERROR | `#pipeline-events` |
| Stack-test failed | GHA Stack-test | ERROR | `#pipeline-events` |
| Pipeline job exit_code != 0 | api → Slack webhook | ERROR | `#pipeline-events` |
| Daily cost > $X | Grafana alert on `total_cost_usd` | WARNING | `#pipeline-events` |
| `/api/health` 5-min downtime | Grafana Synthetic | ERROR | `#pipeline-events` |

The api gets a thin webhook emitter (env-configured, defaults off) for
job-state transitions. It does **not** call Slack directly — it posts
to a generic webhook URL the operator sets to point at Slack, Home
Assistant, or a Shortcuts handler. Same surface, different sinks.

## Security

- **No public inbound ports.** Codespaces port forwarding is
  private-by-default with GitHub auth, no inbound port on the public
  internet.
- **Published GHCR images are scope-limited** to api / viewer /
  pipeline-llm — never the ML pipeline image. The publish workflow
  job's image list is the explicit enforcement boundary against
  accidental redistribution of attribution-restricted models.
- **`.env` leakage CI assertion** runs against every candidate image
  before push (see Layer 1 image hygiene gates).
- **Codespaces automation token** is a fine-grained PAT scoped to the
  `codespaces` permission only, used by the deploy workflow to
  rebuild + start the named pre-prod codespace. The default
  `GITHUB_TOKEN` does not have this scope.
- **Secrets** (provider API keys, Slack webhook URL, Grafana Cloud
  token, Sentry DSN, backup-repo PAT) live in:
  - **Codespaces Secrets** — Settings → Codespaces → Secrets,
    user-level or repo-level. Injected as env vars at codespace
    boot; compose interpolates `${OPENAI_API_KEY:-}` etc.
  - **GitHub Actions Secrets** (deploy / backup workflows).
  Never in compose files, never in images.

## Costs

| Component | Plan | Cost |
| --- | --- | --- |
| GitHub Codespaces | Personal free, 120 core-hrs/mo, 15 GB-month storage | $0 |
| GHCR (public images) | included with GitHub | $0 |
| Grafana Cloud Free | 14 day metrics, 50 GB logs/mo, 1 synthetic | $0 |
| Sentry Free | 5 k errors/mo, 10 k transactions, 1 user | $0 |
| GitHub Releases (private backup repo) | ~10 GB soft cap, 2 GB per asset | $0 |
| Slack Free | 90-day message history, unlimited channels | $0 |
| **Total** | | **$0** |

The only non-zero cost is the **cloud_thin pipeline run itself** —
provider API charges (OpenAI Whisper + Gemini Flash Lite) on the
operator's own keys. A 200-episode test corpus is ~$40 one-time.

### Paid-tier exits if hobby grows

- Grafana Cloud Pro: $19/mo for 1 yr retention + 100 GB logs.
- Codespaces beyond 120 core-hrs/mo: $0.18 per core-hour (4-core
  ≈ $0.36/hr).
- Always-on hosting: covered by the follow-up RFC.

## Phased Rollout

Two phases. Each is a complete, usable environment; the second is a
**lift-and-shift** of the first onto a different host once the
operator has outgrown Codespaces. Within each phase, the four layers
(deploy / observability / control plane / notifications) ship as
sub-steps so the operator can stop at any sub-step with a working
artefact.

### Phase 1 — Codespaces (the validation rung)

**Goal:** prove the deploy → run → observe → alert chain works
end-to-end using only GitHub-native primitives, before introducing a
new vendor.

**1A — Build, test, publish (workflow restructure):**

The current `stack-test.yml` workflow is a single job that builds + tests.
Restructure into three jobs with explicit gating so build/test failures
attribute cleanly and a flaky test doesn't trigger an image rebuild:

```text
build:        # builds api + viewer + pipeline-llm + (locally-only) pipeline-ml
              # uses gha buildx cache for layer reuse
              # outputs: image digests + sha tags
              # no push yet

stack-test:   # needs: build
              # runs Playwright against the locally-built images
              # exercises pipeline-ml (airgapped_thin profile) — never published

publish:      # needs: stack-test
              # if: success() && github.ref == 'refs/heads/main'
              # re-tags + pushes ONLY api / viewer / pipeline-llm to GHCR
              # runs the .env / license assertions before push
```

The `pipeline-ml` image is built (so stack-test exercises the airgapped
path) but **never reaches the publish step**. The publish job's image
list is the explicit safety boundary against accidental redistribution.

**Profile in pre-prod: `cloud_thin`.** Reasoning recap:

- Speed: 100-200 episode test corpus runs in ~30 min wall-time vs ~10 hrs
  on Codespaces 4-core CPU with local ML.
- Image size: `pipeline-llm` is ~1 GB vs `pipeline-ml`'s 15.5 GB —
  fits Codespaces 32 GB SSD with headroom.
- License: zero attribution-required models in the published image.
- Cost: ~$40 one-time for 200 episodes (OpenAI Whisper + Gemini Flash
  Lite); orthogonal to host cost.

**Codespace devcontainer:**

- `.devcontainer/devcontainer.json` runs `docker compose -f
  compose/docker-compose.stack.yml -f compose/docker-compose.prod.yml
  up -d` for api + viewer; the api job factory launches the pipeline
  on demand using the `pipeline-llm` service (no `pipeline-ml` in this
  compose set).
- `.devcontainer/prebuild.yml` triggers prebuilds on push to `main` so
  GHCR pulls land in the prebuild cache. With the slimmed published
  image set, the cold-start hit drops from ~15 min (pulling 15.5 GB)
  to ~1-2 min (pulling ~2 GB total across the three published images).

**Separate deploy-codespace workflow** (`.github/workflows/
deploy-codespace.yml`):

- Triggers: `workflow_dispatch` (manual) + `workflow_run` (auto on
  Stack-test success on `main`).
- Steps: `gh codespace rebuild --full` against the named pre-prod
  codespace via a fine-grained PAT (`codespaces` scope).
- Healthcheck: poll the forwarded `/api/health` from the runner with
  a 5-minute budget; fail the workflow if the new stack doesn't come
  up.
- `make deploy-codespace` Makefile target as the manual escape hatch.

**Acceptance:** push to main → build/test/publish all green → published
images appear at `ghcr.io/chipi/podcast-scraper-stack-{api,viewer,
pipeline-llm}:main` → deploy-codespace workflow fires → codespace
wakes with the new bits → forwarded `:8090` → `/api/health` returns
200 to mobile Safari → loading a feed kicks off a job that uses the
cloud_thin profile end-to-end.

**1B — Observability (Grafana Cloud free):**

- Grafana Agent added as a devcontainer feature (no compose change).
- FastAPI `/metrics` endpoint via
  `prometheus-fastapi-instrumentator` (~3-line add).
- Agent ships container logs (Docker logging driver) + metrics +
  `/api/health` synthetic monitor to Grafana Cloud.
- Grafana Cloud free tier credentials live in GHA Secrets, injected
  into the codespace via `secrets:` in `devcontainer.json`.
- **Acceptance:** phone Grafana Cloud app shows last 24 h of api
  requests, last failed pipeline job's stdout, current daily cost
  rollup.

**1C — Control plane:**

- Codespaces port forwarding is the auth wall: GitHub login required,
  no inbound ports exposed.
- Operator bookmarks the forwarded viewer URL (mobile Safari bookmark,
  plus iOS share-sheet to home screen for "app-like" feel).
- Grafana Cloud SSO bound to the operator's GitHub identity.
- **Acceptance:** phone on cellular hits the forwarded URL, signs in
  once via GitHub, lands on the viewer.

**1D — Notifications + automation:**

- Slack incoming webhook URL stored in GHA Secrets + Grafana Cloud.
- GHA notify step on deploy success / failure / Stack-test failure
  → `#pipeline-events`.
- Grafana alert rules (cost threshold, healthcheck downtime, error
  spike) routed to the same Slack channel via Grafana Cloud's contact
  points.
- api gets an optional job-state webhook emitter (env-configured,
  defaults off; reused unchanged in Phase 2).
- **Acceptance:** trigger a deliberate failure (rename a feed URL),
  observe the Slack push reach the phone within 60 s.

**1E — Corpus persistence + backup:**

Codespaces' default delete-after-30-days-inactivity is a real data-loss
risk for an accumulating corpus. Two-layer mitigation, **GitHub-native
all the way (no third-party storage vendor)**:

- **In-codespace persistence:** bind-mount the corpus to
  `/workspaces/podcast_scraper/.codespace_corpus/` (workspace path,
  survives suspend) instead of an anonymous Docker volume (which
  survives suspend but is harder to back up).
- **Off-Codespace backup to a private GitHub Releases repo**:
  - One-time setup: create `chipi/podcast_scraper-backup` as a
    **private** repository (free for personal accounts; private so
    backup contents are never publicly readable).
  - New GHA workflow `.github/workflows/backup-corpus.yml`, scheduled
    daily.
  - Wakes the codespace, runs `tar -czf snapshot.tgz .codespace_corpus`,
    then `gh release create snapshot-$(date -u +%Y%m%d) snapshot.tgz
    --repo chipi/podcast_scraper-backup --notes "auto"`.
  - Retention: keep 7 daily + 4 weekly snapshots; pruner step removes
    older releases via `gh release delete`. Soft repo cap is ~10 GB
    (matches what we'd have had on R2 anyway).
  - The workflow uses a fine-grained PAT with `contents: write` on the
    backup repo only — stored as `BACKUP_REPO_TOKEN` in GHA Secrets.
  - **No new vendor sign-up.** Same GitHub identity, same Codespaces
    Secrets surface, same `gh` CLI.

**Restore path:**

- `make restore-corpus` Makefile target: pulls the latest snapshot
  via `gh release download <tag> --repo
  chipi/podcast_scraper-backup` and untars into `.codespace_corpus/`.
- Works from a fresh codespace (resurrected after delete).

Why GitHub Releases for Phase 1:

- **Persistent** (no time-based expiry, unlike Actions artifacts).
- **No new vendor.** Same GitHub identity, same `gh` CLI, same
  Codespaces Secrets surface.
- **~10 GB soft cap** on a free private repo, 2 GB per asset —
  matches what we'd get on R2's free tier anyway, and is plenty for
  corpus snapshots in the Phase 1 window.

> **Phase 2 backup is a separate decision, not a lift-and-shift of
> this.** When the always-on RFC drafts, it will likely want a real
> object store (Cloudflare R2 / Backblaze B2 / Storj) because
> always-on changes the constraints: corpus accumulates without the
> Codespaces 30-day reset; a DB backup (when one lands) wants
> point-in-time semantics that align better with object versioning;
> cross-host restore performance + bandwidth caps start to matter.
> The follow-up RFC owns that decision; the GitHub Releases
> mechanism here does **not** silently graduate.

**Exit criteria for Phase 1 (triggers for the follow-up RFC):**

- Codespaces 60 hrs/mo limit consistently hit.
- Auto-suspend latency annoying for casual phone use.
- Operator wants to run a real feed continuously for days.

When any of those bites, draft the always-on RFC; until then, Phase 1
is the contract.

## Alternatives Considered

- **Self-hosted Prometheus + Grafana + Loki**: rejected as more moving
  parts than the hobby goal warrants. Free tier of Grafana Cloud
  removes ~3 services from the codespace without losing capability.
- **Always-on host (any VPS) instead of Codespaces for Phase 1**:
  rejected for this RFC's scope — would force a vendor sign-up and a
  control-plane decision (Tailscale / Cloudflare Tunnel / etc.) before
  we have signal on whether the deploy → run → observe → alert chain
  even works for the operator. Phase 1 on Codespaces lets us validate
  the chain end-to-end with only GitHub primitives; the always-on
  follow-up RFC then makes a host decision with real usage data.
- **Self-hosted everything (Codespaces dev container running its own
  Prometheus / Grafana / Loki)**: rejected. The codespace is meant to
  be cheap and transient; pinning a stateful observability stack to it
  makes both observability and the codespace heavier.
- **GitHub Pages / Render free / Fly.io free for the published
  viewer**: rejected because they don't run the api or the pipeline,
  only static assets. We need the whole compose stack reachable, not
  just the viewer.
- **Docker Swarm / Kubernetes**: comically over-spec for a single-
  codespace hobby env.

## Open Questions

1. **Codespaces image-cache pressure on 32 GB SSD.** The 4-core tier
   has 32 GB host disk. With the slimmed published image set (no
   `pipeline-ml`, ~2 GB total) the disk is comfortable, but the
   prebuild cache + workspace checkout + dev deps still adds up.
   **Lean:** start on 4-core; if `df -h` inside the codespace
   consistently >85 %, escalate to 8-core (64 GB disk, spends 4×
   faster but lasts longer per use).
2. **Codespaces auto-suspend latency.** First-tap-to-loaded after a
   suspended codespace is ~30-60 s with prebuilds, longer without.
   Acceptable for hobby phone use; chronic annoyance is the trigger
   for the always-on follow-up RFC.
3. **Cost-spike alert thresholds.** What is "$X" for the cost alert?
   Pulled from RFC-068 (cost observability) — the operator sets a
   per-day ceiling. Default: ~$1/day; tune based on observed
   baseline.
4. **Backup retention policy.** Default proposal: keep 7 daily + 4
   weekly snapshots in the private Releases repo, prune older. Tune
   once we see actual corpus growth. If the ~10 GB soft cap on the
   backup repo gets uncomfortable, that is itself signal that the
   always-on RFC's object-store decision is overdue.
5. **Codespace deletion threshold.** Codespaces auto-delete after 30
   days of inactivity. With weekly Releases snapshots that's a soft
   loss (the corpus restores from the backup repo), but the codespace
   identity itself resets — the deploy workflow must handle "named
   codespace doesn't exist, create one" as a path, not just "rebuild
   existing".

## References

- [GitHub Codespaces docs](https://docs.github.com/en/codespaces)
- [GitHub Codespaces secrets](https://docs.github.com/en/codespaces/managing-codespaces-for-your-organization/managing-encrypted-secrets-for-your-repository-and-organization-for-github-codespaces)
- [Codespaces prebuilds](https://docs.github.com/en/codespaces/prebuilding-your-codespaces)
- [Grafana Cloud Free](https://grafana.com/pricing/)
- [Sentry Free](https://sentry.io/pricing/)
- [GitHub Releases (gh release CLI)](https://cli.github.com/manual/gh_release)
- [GHCR docs](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [Slack incoming webhooks](https://api.slack.com/messaging/webhooks)
