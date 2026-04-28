# Codespace pre-prod runbook (RFC-081 Phase 1)

Operator-facing notes for running the pre-prod stack in a GitHub
Codespace. Captures gotchas surfaced during first-boot smoke testing
that aren't obvious from the RFC or the workflow files alone.

This is a **working notes doc** — promote anything load-bearing into
permanent guides under `docs/guides/` once stable.

---

## Quick reference

```bash
# From your local laptop, with gh CLI authed (codespace scope):
make codespace-status          # Available / Shutdown / Rebuilding ...
make codespace-start           # Wake (also picks up updated Codespaces secrets)
make codespace-stop            # Pause billing
make deploy-codespace          # Full rebuild via gh codespace rebuild --full
```

| Path inside codespace | Path inside api / pipeline-llm container | What |
|---|---|---|
| `/workspaces/podcast_scraper/.codespace_corpus/` | `/app/output/` | Bind-mounted corpus root |
| `/workspaces/.codespaces/shared/.env` | n/a | Codespaces secrets file (read-only) |
| `compose/docker-compose.stack.yml` + `compose/docker-compose.prod.yml` | n/a | Compose stack the codespace boots |

The bind mount means **edit `feeds.spec.yaml` / `viewer_operator.yaml`
directly from the codespace shell** at the host path; the api sees the
same files at `/app/output/`. No `docker compose cp` needed.

---

## When does the codespace need a bounce?

| Change | Action | Why |
|---|---|---|
| Codespaces secret value (e.g., `OPENAI_API_KEY`, `GRAFANA_CLOUD_API_KEY`) | `make codespace-stop` then `make codespace-start` | Secrets are baked into the container's env at start, not read live. |
| Code on `main` (api / viewer / pipeline image rebuilt + published) | `make deploy-codespace` (full rebuild) | Pulls fresh GHCR images and re-runs `postStartCommand`. |
| `compose/docker-compose.prod.yml` change (volume defs, env wiring) | `make deploy-codespace` | postStart re-creates compose stack; start.sh's stale-volume defense recovers from any volume-spec drift. |
| `.devcontainer/devcontainer.json` (features, secrets surface, ports) | `make deploy-codespace` (full rebuild only) | Container rebuild is the only path to apply devcontainer changes. |
| Operator yaml content (`viewer_operator.yaml` body) | None | api re-reads on every job submit. |

`codespace-start` is the cheap path (~30s, no rebuild). `deploy-codespace`
is full (~5-10 min) — use when image or devcontainer changes need to
land.

---

## Pipeline-job lifecycle inside the codespace

`POST /api/jobs` flow with `PODCAST_PIPELINE_EXEC_MODE=docker` (the
prod overlay's default — see [docker-compose.prod.yml](../../compose/docker-compose.prod.yml)):

1. api validates `viewer_operator.yaml`'s `pipeline_install_extras`
   (must be `ml` or `llm`; `llm` for both cloud_balanced and cloud_thin in
   Phase 1 — pipeline-llm now ships [llm] + [search] extras so vector
   search works without dragging in [ml]'s spaCy / Whisper / Pegasus).
2. api validates `profile:` against `PODCAST_AVAILABLE_PROFILES`
   (defense in depth on top of the dropdown filter).
3. api spawns `docker compose run --rm pipeline-llm <argv>` from
   inside its own container — works because the api container has
   `/var/run/docker.sock` bind-mounted.
4. pipeline-llm container runs `python -m podcast_scraper.cli ...`
   with the operator yaml + feeds spec + corpus output dir.
5. Artifacts land at the host bind path
   (`.codespace_corpus/feeds/<feed>/run_<id>/...`).

Logs for a job: `.codespace_corpus/.viewer/jobs/<job-id>.log`.

Job state: `GET /api/jobs?path=/app/output` (or via the viewer's
Pipeline tab). Terminal states: `succeeded` / `failed` / `cancelled`.

---

## Common gotchas

### "path must be the configured corpus root or a subdirectory of it"

Symptom: setting any corpus path through the viewer returns this 400.

Cause: the api validates user-supplied paths against its container's
filesystem, not the codespace shell's. The api's corpus anchor is
`/app/output` (set via `--output-dir` in the api startup CMD).

Fix: set the viewer corpus path to `/app/output` (NOT a host path
like `/workspaces/...`). The bind mount makes `/app/output` inside the
container == `/workspaces/podcast_scraper/.codespace_corpus/` on the
host.

### Configuration tab shows nothing on first open

Cause: the viewer's Configuration dialog only loads via
`GET /api/operator-config?path=<corpus_root>`. Until you set the
corpus path in the status bar, the Configuration tabs (Feeds, Job
Profile, Operator) stay empty.

Fix: status bar → corpus path → enter `/app/output` → submit.

### grafana-agent container restart-loops

Cause #1: `GRAFANA_CLOUD_*` Codespaces secret value is wrong (empty,
truncated, or contains chars that break the agent's env-substitution
parser). Real keys start with `glc_` and are ~50+ chars; URLs are full
`https://...` strings.

Cause #2: agent.yaml was authored with bash-style `${VAR:-default}`.
Grafana Agent's `-config.expand-env=true` uses Go's `os.ExpandEnv`,
which **does not support `:-default`**. Use `${VAR}` only; default in
compose-side env wiring instead.

Diagnostic:

```bash
docker compose -f compose/docker-compose.stack.yml -f compose/docker-compose.prod.yml \
  logs --tail 20 grafana-agent
```

### Pipeline job 400s with "pipeline_install_extras must be 'ml' or 'llm'"

Cause: `viewer_operator.yaml` is missing the `pipeline_install_extras`
field, or has a value other than `ml` / `llm`. Required when
`PODCAST_PIPELINE_EXEC_MODE=docker` (the prod overlay's mode) — the
api uses it to pick which compose service to spawn (`pipeline` for
ml, `pipeline-llm` for llm).

Fix: add `pipeline_install_extras: llm`. Both cloud_balanced and
cloud_thin run on the pipeline-llm image in Phase 1 — the image now
ships [search] extras (faiss + sentence-transformers + torch CPU)
on top of [llm] so vector_search works without [ml].

### Pipeline job 400s with "profile X is not in the available profiles"

Cause: the saved `profile:` in `viewer_operator.yaml` is outside
`PODCAST_AVAILABLE_PROFILES` for this env. Phase 1 allows
`cloud_balanced` (default) and `cloud_thin`.

Fix: pick `cloud_balanced` (or `cloud_thin`) from the operator dropdown
\+ Save (overwrites the file's profile line).

### `ANTHROPIC_API_KEY` (or any other secret) shows `len:1` in `docker inspect`

Cause: the Codespaces secret value is truncated / set to a placeholder
character / corrupt. The `len=1` is a tell — real API keys are 40+
chars.

Fix: re-set the Codespaces secret with the full value:

```bash
gh secret set <SECRET_NAME> --repo chipi/podcast_scraper --app codespaces --body '<full-value>'
```

Then `make codespace-stop && make codespace-start` to refresh container
env.

### After main updates compose/docker-compose.prod.yml, codespace acts stale

Cause: `docker compose up` does NOT recreate existing named volumes
when their YAML definition changes. A pre-overlay-change codespace
keeps using the old (Docker-managed) volume; the operator-shell view
of the corpus stays empty.

Fix (automatic, since 2026-04-28): `start.sh` now probes the existing
`compose_corpus_data` volume's bind device on every boot. Mismatch =
auto teardown + recreate. So a `make codespace-start` (or
`make deploy-codespace`) is enough.

### `gh codespace ssh` fails with "Please check if an SSH server is installed in the container"

Cause: codespace was created from a branch lacking the
`ghcr.io/devcontainers/features/sshd:1` feature in
`.devcontainer/devcontainer.json`.

Fix: rebase / pull the branch onto a commit that includes the sshd
feature (added 2026-04-28), then `make deploy-codespace` to apply.

### `gh codespace ssh` fails with "Permission denied (publickey,password)"

Cause: gh CLI hasn't auto-generated `~/.ssh/codespaces.auto` yet.
Usually happens silently on first ssh; some non-interactive pipelines
trip it.

Fix:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/codespaces.auto -N '' -C 'gh-codespaces-auto'
```

Then retry. gh uploads the public key via the codespace agent on next
ssh attempt.

---

## Auth scopes — what tokens you actually need

| Token / setting | Scope | Used by |
|---|---|---|
| `gh auth login --scopes codespace` (your laptop) | `codespace` | `make codespace-*` targets, manual ssh |
| `CODESPACES_PAT` Actions secret | **classic PAT, `codespace` scope** | `deploy-codespace.yml` workflow. Fine-grained PATs scoped to a single repo **do not** work — `/user/codespaces/<name>/start` is a user-level endpoint and 403s. |
| `BACKUP_REPO_TOKEN` Actions secret | fine-grained, `Contents: Read and write` on `chipi/podcast_scraper-backup` | `backup-corpus.yml` workflow (release-asset upload). |
| Codespaces secrets (`OPENAI_API_KEY` / `GEMINI_API_KEY` / `PODCAST_SENTRY_DSN_*` / `GRAFANA_CLOUD_*`) | n/a | Injected as env into the codespace container at start. |
| Actions secret `VITE_SENTRY_DSN_VIEWER` | n/a | Read by the Stack-test publish job to bake the Sentry DSN into the viewer bundle at build time. |

---

## Backup / restore

Backup cron (`backup-corpus.yml`, daily 04:17 UTC) tarballs
`/workspaces/podcast_scraper/.codespace_corpus` and uploads to
`chipi/podcast_scraper-backup` as a release asset. Cron is gated on
`PODCAST_BACKUP_REPO_READY=true` repo variable.

Restore:

```bash
make restore-corpus  # pulls latest snapshot + untars into .codespace_corpus/
```

Run from inside the codespace (or set `WORKSPACE_DIR` for a host-side
restore). Requires gh CLI authed with read access to the private
backup repo.
