# RFC-078 stack smoke (Playwright)

Minimal checks against the **real** Nginx → FastAPI stack (not MSW mocks).

## Stack contract (RFC-078 ↔ RFC-079 / #659)

Smoke uses **`compose/docker-compose.smoke.yml`** over **`compose/docker-compose.stack.yml`**. For CI and local debugging, align with the full guide:

- **`CONFIG_FILE`:** absolute path; `output_dir` must be **`/app/output`** (see `config/examples/docker-stack.example.yaml`).
- **Ports:** smoke viewer defaults to **8090** (`SMOKE_VIEWER_PORT` / `SMOKE_BASE_URL` for Playwright).
- **Volumes:** pipeline writes to **`smoke_data`**; `make smoke-export-corpus` copies to **`.smoke-corpus/`** (or `SMOKE_EXPORT_DIR`).
- **Docker-backed jobs:** optional `compose/docker-compose.jobs-docker.yml` — **#660**, not required for default smoke.

Canonical table: `docs/guides/DOCKER_SERVICE_GUIDE.md` § **RFC-079 backlog** → *Handoff to RFC-078 smoke*.

## Local flow

From the repository root, with Docker:

1. `make smoke-build`
2. `make smoke-run-pipeline` — writes `.smoke/pipeline.log`
3. `make smoke-assert-logs` — sanity-checks that log (no traceback, expects `Wrote corpus run summary`)
4. `make smoke-export-corpus` — copies `/app/output` from the **`smoke_data`** volume to **`.smoke-corpus/`** (override dir: `SMOKE_EXPORT_DIR=...`)
5. `make smoke-assert-artifacts` — GIL/KG quality gates on **`SMOKE_CORPUS_ROOT`** (defaults to **`$(pwd)/.smoke-corpus`**)
6. `make smoke-up` then `make smoke-test-playwright` (defaults to `http://127.0.0.1:8090`)

Override the base URL: `SMOKE_BASE_URL=http://127.0.0.1:8080 make smoke-test-playwright`.

## CI

[`.github/workflows/smoke-test.yml`](../../.github/workflows/smoke-test.yml) runs the same
Make sequence (build → pipeline → **assert-logs** → **export-corpus** → **assert-artifacts**
→ stack up → Playwright) on path-filtered **`push` to `main`** and **`workflow_dispatch`**.
Prerequisite stack work is tracked on GitHub [**#659**](https://github.com/chipi/podcast_scraper/issues/659) /
[**#660**](https://github.com/chipi/podcast_scraper/issues/660) (both materialize [**RFC-079**](../../docs/rfc/RFC-079-full-stack-docker-compose.md)).
Further smoke CI changes (`workflow_run` after `python-app`, GHA BuildKit cache backends,
merge-blocking policy) need **new GitHub issues** — [`RFC-078`](../../docs/rfc/RFC-078-ephemeral-acceptance-smoke-test.md) is **design only**, not a backlog.
