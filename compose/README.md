# Docker Compose

Compose definitions for running the platform end-to-end. **Build context is `..` (repo root)** so `docker/<image>/Dockerfile` and bind-mount `../<...>` paths resolve correctly relative to the repo root.

For the operator's run-it-on-your-machine guide, start at [`docs/guides/DOCKER_COMPOSE_GUIDE.md`](../docs/guides/DOCKER_COMPOSE_GUIDE.md).

## Files

| File | Purpose |
| ---- | ------- |
| `docker-compose.stack.yml` | **Base** — `viewer` (Nginx + SPA), `api` (FastAPI), and the `pipeline` / `pipeline-llm` services (Compose profiles `pipeline` / `pipeline-llm`, only spawned per job). Production-ready when paired with the `prod` and `jobs-docker` overlays below. |
| `docker-compose.stack-test.yml` | **Overlay** — adds `mock-feeds` (Nginx sidecar serving bundled RSS / audio / transcripts) plus the Docker job-mode wiring on the `api` service so the API factory can spawn pipeline containers. Used by `make stack-test-*` and the GitHub stack-test workflow. |
| `docker-compose.jobs-docker.yml` | **Overlay** — production equivalent of the stack-test overlay's job-mode wiring: bind-mounts the host repo + `/var/run/docker.sock` into `api` and sets `PODCAST_PIPELINE_EXEC_MODE=docker`. Merge this on top of `stack.yml` when you want production deployments to spawn jobs the same way the stack-test does, without the `mock-feeds` sidecar. |
| `docker-compose.prod.yml` | **Overlay** — restart policies for long-running services (`viewer`, `api`). Site-local volume / network changes go in your own override on top of this. |

## Quick reference

From the repo root:

```bash
# Local end-to-end (recommended path — see DOCKER_COMPOSE_GUIDE.md)
make stack-test-build
make stack-test-up
make stack-test-seed                      # default ml variant
# open http://127.0.0.1:8090

# Production-style without mock-feeds (Docker job mode + restart policies)
export PODCAST_DOCKER_PROJECT_DIR="$PWD"
docker compose \
  -f compose/docker-compose.stack.yml \
  -f compose/docker-compose.jobs-docker.yml \
  -f compose/docker-compose.prod.yml \
  up -d
```

## Profile vs image-tier mapping

The operator yaml's `pipeline_install_extras` decides which compose service the API factory spawns: `ml` → `pipeline`, `llm` → `pipeline-llm`. The packaged profile (`airgapped_thin`, `cloud_thin`, …) decides what the spawned container does. Picking a cloud profile but leaving `pipeline_install_extras: ml` (or vice versa) ships the wrong image tier — the UI sets this correctly when you save a profile.

See [Docker variants guide](../docs/guides/DOCKER_VARIANTS_GUIDE.md) for the image tier comparison.
