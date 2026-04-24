# Docker Compose

Stack, stack-test overlay, and standalone pipeline definitions. **Build `context` is `..` (repo root)** so `docker/pipeline/Dockerfile` and friends resolve correctly; **bind mounts** use `../…` paths from this directory.

From the repository root:

```bash
docker compose -f compose/docker-compose.stack.yml up -d
```

`make stack-*` and `make stack-test-*` use these files (see the `Makefile`).

For **`CONFIG_FILE`** on the stack, prefer an **absolute** path (e.g. `CONFIG_FILE=$PWD/config/examples/docker-stack.example.yaml`) so it does not depend on Compose path resolution.

**Docker-backed jobs (#660):** merge `docker-compose.jobs-docker.yml` so `api` gets the host Docker socket and **`PODCAST_PIPELINE_EXEC_MODE=docker`**. Set **`PODCAST_DOCKER_PROJECT_DIR`** to the **absolute repo root** on the host (path the Docker daemon resolves). Example:

```bash
export PODCAST_DOCKER_PROJECT_DIR="$PWD"
docker compose -f compose/docker-compose.stack.yml -f compose/docker-compose.jobs-docker.yml up -d
```

**Compose files for `docker compose run`:** the API factory defaults to **`-f compose/docker-compose.stack.yml`** only (see `PODCAST_DOCKER_COMPOSE_FILES`). The **`jobs-docker`** merge is required at **`up`** so `api` has the socket, env, and a **read-only bind mount** of the host repo at **`/podcast_repo`** (host path comes from **`PODCAST_DOCKER_PROJECT_DIR`** at `up` time; **inside** the container **`PODCAST_DOCKER_PROJECT_DIR` is `/podcast_repo`** so `docker compose -f compose/…` resolves). If you need the factory’s `compose run` to see the same merge graph as `up`, set **`PODCAST_DOCKER_COMPOSE_FILES`** to a comma-separated list (e.g. `compose/docker-compose.stack.yml,compose/docker-compose.jobs-docker.yml`).

**Profile vs image tier:** corpus **`CONFIG_FILE`** profile (e.g. `cloud_thin`) defines pipeline behavior; **`viewer_operator.yaml`** must declare **`pipeline_install_extras: ml`** or **`llm`** for Docker job mode so the correct **`pipeline`** / **`pipeline-llm`** service is used.

See `docs/guides/DOCKER_SERVICE_GUIDE.md` § Full stack.

**Prod-style restart policies:** merge [`compose/docker-compose.prod.yml`](docker-compose.prod.yml) (restart + commented VPS hints; see guide § RFC-079 backlog).

RFC-078 helpers (from repo root): `make stack-test-assert-logs` → `make stack-test-export` → `make stack-test-assert-artifacts` (or set **`STACK_TEST_EXPORT_DIR`** / **`STACK_TEST_CORPUS_ROOT`**).
