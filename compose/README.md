# Docker Compose

Stack, smoke overlay, and standalone pipeline definitions. **Build `context` is `..` (repo root)** so `docker/pipeline/Dockerfile` and friends resolve correctly; **bind mounts** use `../…` paths from this directory.

From the repository root:

```bash
docker compose -f compose/docker-compose.stack.yml up -d
```

`make stack-*` and `make smoke-*` use these files (see the `Makefile`).

For **`CONFIG_FILE`** on the stack, prefer an **absolute** path (e.g. `CONFIG_FILE=$PWD/config/examples/docker-stack.example.yaml`) so it does not depend on Compose path resolution.

**Docker-backed jobs (#660):** merge `docker-compose.jobs-docker.yml` so `api` gets the host Docker socket and `PODCAST_PIPELINE_EXEC_MODE=docker`. Example:

```bash
export PODCAST_DOCKER_PROJECT_DIR="$PWD"
docker compose -f compose/docker-compose.stack.yml -f compose/docker-compose.jobs-docker.yml up -d
```

See `docs/guides/DOCKER_SERVICE_GUIDE.md` § Full stack.

**Prod-style restart policies:** merge [`compose/docker-compose.prod.yml`](docker-compose.prod.yml) (restart + commented VPS hints; see guide § RFC-079 backlog).

RFC-078 helpers (from repo root): `make smoke-assert-logs` → `make smoke-export-corpus` → `make smoke-assert-artifacts` (or set **`SMOKE_EXPORT_DIR`** / **`SMOKE_CORPUS_ROOT`**).
