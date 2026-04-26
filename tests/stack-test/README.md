# Stack test (Playwright)

Smoke + full UI flow against the **real** Nginx → FastAPI → mock-feeds stack — no MSW mocks. Validates the same compose stack a developer or operator runs in production.

## What runs

Single Playwright run drives:

1. `stack-viewer.spec.ts` — Nginx serves the SPA shell, `/api/health` is reachable through the Nginx proxy, the corpus auto-loads, the runs-summary endpoint reports a non-zero run, and the graph canvas mounts.
2. `stack-jobs-flow.spec.ts` — full UI flow: set corpus path, add a second feed via the Configuration dialog, select a profile and Save, click **Run pipeline job**, poll `/api/jobs` until the API factory's spawned `docker compose run pipeline` container reaches `succeeded`, then validate Library / Digest / Search / Graph against the produced artifacts.

The pipeline runs as a one-shot container spawned by the API job factory (`PODCAST_PIPELINE_EXEC_MODE=docker`). The factory passes `--config <corpus>/viewer_operator.yaml --feeds-spec <corpus>/feeds.spec.yaml --profile <name>` to the CLI — no `/app/config.yaml` mount, no separate standalone CLI run.

## Local flow

Both variants run from the repo root with Docker Desktop. **`.env`** at the repo root is sourced into `make stack-test-up` so `OPENAI_API_KEY` / `GEMINI_API_KEY` propagate through Compose into the API container, then into spawned pipeline containers.

### One-shot wrappers (recommended)

```bash
make stack-test-ml                          # build → up → seed (ml) → Playwright (airgapped_thin)
make stack-test-cloud-thin                  # build + pipeline-llm → up → seed (cloud-thin) → Playwright (cloud_thin)
```

Both leave the stack **up** after success so you can `make stack-test-export` and inspect artifacts; run `make stack-test-down` when finished.

A third variant `stack-test-ml-ci` runs the same ml flow inside a shell `EXIT` trap that **always** tears down at the end (success or failure). Used by `make ci` as the final gate so a `make ci` run leaves a clean laptop.

### Step-by-step (debug / custom seeds)

#### Airgapped / ml — no cloud calls

```bash
make stack-test-build                       # builds api + viewer + pipeline (ml)
make stack-test-up
make stack-test-seed                        # default STACK_TEST_OPERATOR_VARIANT=ml
make stack-test-playwright                  # spec defaults to STACK_TEST_OPERATOR_PROFILE=airgapped_thin
```

#### Cloud-thin / llm — local only (real Gemini + OpenAI Whisper costs)

```bash
make stack-test-build                       # builds the airgapped images first
make stack-test-build-cloud                 # adds pipeline-llm ([llm] extras)
make stack-test-up
make stack-test-seed STACK_TEST_OPERATOR_VARIANT=cloud-thin
STACK_TEST_OPERATOR_PROFILE=cloud_thin make stack-test-playwright
```

The cloud-thin path does **not** run on public CI (avoids recurring API costs). When `STACK_TEST_OPERATOR_PROFILE=cloud_thin`, the spec gracefully skips the populated-FAISS semantic-search assertion (`vector_search: false` in the profile) and instead asserts `/api/search` returns the structured `error: "no_index"` response.

### Tear down

```bash
make stack-test-down                        # stop the stack, keep corpus_data
make stack-test-down STACK_TEST_DOWN_VOLUMES=1   # also drop corpus_data
```

### Inspect artifacts after a run

```bash
make stack-test-export                      # copies corpus_data → ./.stack-test-corpus/
```

## CI

[`.github/workflows/stack-test.yml`](../../.github/workflows/stack-test.yml) runs the airgapped/ml variant **as a `workflow_run`** triggered after the **Python application** workflow completes successfully on `main` (sequential post-Python gate; the workflow checks out the upstream's `head_sha` so a subsequent push doesn't shift the target). Workflow sequence mirrors the local flow above (`stack-test-build` → `up` → `seed` → `playwright` → `down`) plus an on-failure compose log dump (`api` / `viewer` / `pipeline` / `pipeline-llm`, `--tail=400`) so a pipeline `exit_code_1` is diagnosable from the workflow log instead of silently torn down. Manual `workflow_dispatch` is also accepted. Cloud-thin is local-only.

## Overrides

- `STACK_TEST_BASE_URL` (default `http://127.0.0.1:8090`) — point Playwright at a different viewer host:port.
- `STACK_TEST_OPERATOR_VARIANT` — `ml` (default) or `cloud-thin`. Picks `pipeline_install_extras` and which compose service the API factory spawns.
- `STACK_TEST_OPERATOR_PROFILE` — packaged profile the spec selects via the Job Profile dialog. `airgapped_thin` (default for ml), `cloud_thin` for llm. Other profiles work but the test only special-cases vector_search behaviour.
- `STACK_TEST_VIEWER_PORT` (default `8090`) — host port for the viewer.
- `STACK_TEST_DOWN_VOLUMES` — set to `1` on `make stack-test-down` to also drop `corpus_data`.
- `STACK_TEST_JOB_POLL_MS` (spec env) — tighten the `/api/jobs` poll interval (default 4000 ms).
