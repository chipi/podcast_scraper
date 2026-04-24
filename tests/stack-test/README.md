# Stack test (Playwright)

Minimal checks against the **real** Nginx → FastAPI stack (not MSW mocks).

## Stack-test transcripts (local)

When capturing a full ``run_stack_test_ci_local.sh`` / ``make stack-test-ci-local`` run, write logs **under** ``.stack-test/`` using **short basenames** (for example ``tee .stack-test/full-run.log`` or ``tee .stack-test/ci-local-run.log``). Avoid redundant names like ``.stack-test/.stack-test-full-run.log`` (leading dot + repeated prefix). Make exports ``STACK_TEST_FULL_RUN_LOG`` and ``STACK_TEST_CI_LOCAL_LOG`` (defaults under ``STACK_TEST_LOG_DIR``); run ``make stack-test-rename-legacy-logs`` once to rename existing ``.stack-test/.stack-test-*.log`` files. Avoid creating ``.stack-test-*.log`` at the repository root; those clutter the tree. Optional: ``STACK_TEST_LOG_DIR`` (defaults to ``<repo>/.stack-test``).

## Stack contract (RFC-078 ↔ RFC-079 / #659)

Stack-test uses **`compose/docker-compose.stack-test.yml`** over **`compose/docker-compose.stack.yml`**. For CI and local debugging, align with the full guide:

- **`CONFIG_FILE`:** absolute path; `output_dir` must be **`/app/output`** (see `config/examples/docker-stack.example.yaml`).
- **Ports:** stack-test viewer defaults to **8090** (`STACK_TEST_VIEWER_PORT` / `STACK_TEST_BASE_URL` for Playwright).
- **Volumes:** pipeline writes to **`corpus_data`** (same named volume as the base stack); `make stack-test-export` copies to **`.stack-test-corpus/`** (or `STACK_TEST_EXPORT_DIR`).
- **Docker-backed jobs:** optional `compose/docker-compose.jobs-docker.yml` — **#660**, not required for default stack-test.

Canonical table: `docs/guides/DOCKER_SERVICE_GUIDE.md` § **RFC-079 backlog** → *Handoff to RFC-078 stack-test*.

## Local flow

From the repository root, with Docker:

1. `make stack-test-build`
2. `make stack-test-run` — writes `.stack-test/pipeline.log`
3. `make stack-test-assert-logs` — sanity-checks that log (no traceback, expects `Wrote corpus run summary`)
4. `make stack-test-export` — copies `/app/output` from the **`corpus_data`** volume to **`.stack-test-corpus/`** (override dir: `STACK_TEST_EXPORT_DIR=...`)
5. `make stack-test-assert-artifacts` — GIL/KG quality gates on **`STACK_TEST_CORPUS_ROOT`** (defaults to **`$(pwd)/.stack-test-corpus`**)
6. `make stack-test-up` then `make stack-test-playwright` (defaults to `http://127.0.0.1:8090`)

Override the base URL: `STACK_TEST_BASE_URL=http://127.0.0.1:8080 make stack-test-playwright`.

## CI

[`.github/workflows/stack-test.yml`](../../.github/workflows/stack-test.yml) runs the same
Make sequence (build → pipeline → **assert-logs** → **export** → **assert-artifacts**
→ stack up → Playwright) on path-filtered **`push` to `main`** and **`workflow_dispatch`**.
Prerequisite stack work is tracked on GitHub [**#659**](https://github.com/chipi/podcast_scraper/issues/659) /
[**#660**](https://github.com/chipi/podcast_scraper/issues/660) (both materialize [**RFC-079**](../../docs/rfc/RFC-079-full-stack-docker-compose.md)).
Further stack-test CI changes (`workflow_run` after `python-app`, GHA BuildKit cache backends,
merge-blocking policy) need **new GitHub issues** — [`RFC-078`](../../docs/rfc/RFC-078-ephemeral-acceptance-smoke-test.md) is **design only**, not a backlog.
