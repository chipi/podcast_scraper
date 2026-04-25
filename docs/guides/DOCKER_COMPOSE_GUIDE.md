# Docker Compose Guide

The recommended way to run the platform end-to-end on a single machine. Brings up the API, the GI/KG viewer (browser UI), and an on-demand pipeline runner — all wired together. You drive everything from the browser: add RSS feeds, choose a processing profile, click **Run pipeline job**, watch artifacts appear.

This guide is for **operators and end users**. If you want to develop the Python code or the Vue viewer, see [Polyglot repository guide](POLYGLOT_REPO_GUIDE.md) and [Development guide](DEVELOPMENT_GUIDE.md).

## Why Docker Compose

You get the whole platform with one `docker compose up`. No Python/venv setup on the host, no Node/npm setup, no managing supervisord — Docker handles the lifecycle. The same images and compose files run on your laptop, CI, or a small VPS.

Pipeline jobs run as **one-shot containers** spawned by the API on demand. There is **no long-running pipeline service** — when you click **Run** in the UI, the API calls `docker compose run --rm pipeline ...` (or `pipeline-llm` for cloud profiles), the container processes your feeds, writes artifacts to a shared named volume, and exits. The viewer then reads those artifacts directly.

## Prerequisites

- **Docker Desktop** (macOS / Windows) or **Docker Engine + Compose v2** (Linux). Compose plugin v2.20+ recommended.
- **~5 GB free disk** for the airgapped/ml image (Whisper + transformers + spaCy preloaded). The cloud-thin pipeline-llm image is ~500 MB.
- **GNU make** if you want the convenience targets (optional — every step can be run as plain `docker compose ...`).
- Optional: **`.env`** at the repo root with provider keys (only needed for cloud profiles):

  ```bash
  # .env
  OPENAI_API_KEY=sk-...
  GEMINI_API_KEY=AI...
  ANTHROPIC_API_KEY=sk-ant-...   # optional
  MISTRAL_API_KEY=...            # optional
  DEEPSEEK_API_KEY=...           # optional
  GROK_API_KEY=...               # optional
  ```

  `.env` is gitignored. Compose reads it via `make stack-test-up` (or pass `--env-file ./.env` to your own `docker compose` invocation).

## First run

From the repo root:

### 1. Build the images

```bash
make stack-test-build
```

This builds three images: `podcast-scraper-stack-api`, `podcast-scraper-stack-viewer`, and `podcast-scraper-stack-pipeline` (the airgapped/ml variant). First build takes 5–15 minutes — Whisper, spaCy, and the summarization transformers are downloaded and cached into the pipeline image so runtime stays offline.

If you plan to use cloud providers (OpenAI Whisper API, Gemini, …) also build:

```bash
make stack-test-build-cloud
```

This adds `podcast-scraper-stack-pipeline-llm` (~500 MB, no local ML).

### 2. Bring up the stack

```bash
make stack-test-up
```

Three long-running services start:

- **`viewer`** (Nginx serving the Vue SPA) — exposed on `http://127.0.0.1:8090`.
- **`api`** (FastAPI on port 8000, proxied via Nginx at `/api/*`).
- **`mock-feeds`** (an Nginx sidecar that serves a few bundled RSS fixtures plus their audio + transcripts — handy for trying the platform without configuring real feeds).

The pipeline / pipeline-llm services are **not** started here — they spawn per-job.

If `.env` exists at the repo root, `stack-test-up` sources it so provider keys propagate into the API container.

Check the API is up:

```bash
curl -fsS http://127.0.0.1:8090/api/health
```

### 3. Open the viewer

Navigate to **<http://127.0.0.1:8090>**. You'll see the GI/KG viewer.

### 4. Configure feeds and profile

In the **Configuration** dialog (footer or status-bar entry):

- **Feeds tab** — add one or more RSS URLs. For your first run you can use the bundled fixtures:
  - `http://mock-feeds/p01_fast_with_transcript.xml` — 1 episode, fast (transcript already in feed)
  - `http://mock-feeds/p01_episode_selection.xml` — 3 episodes, also transcript-only

  In production, add your own RSS feed URLs (any podcast that exposes RSS — most do).

- **Job Profile tab** — pick a packaged profile and click **Save**:
  - `airgapped_thin` — local Whisper + transformers, no cloud calls. Default. Good for evaluating the platform.
  - `cloud_thin` — OpenAI Whisper + Gemini (NER, summary, GIL, KG). Faster and better quality on real episodes; needs `OPENAI_API_KEY` + `GEMINI_API_KEY` in `.env`.
  - `cloud_balanced`, `cloud_quality` — richer cloud configurations with vector search for semantic queries.
  - `local`, `dev`, `airgapped` — full local stacks for advanced setups.

  See [`config/profiles/README.md`](../../config/profiles/README.md) for the full list and trade-offs.

  The **Save** action writes both the chosen profile name and your overrides to `<corpus>/viewer_operator.yaml` on the shared `corpus_data` volume.

### 5. Run a pipeline job

Click **Dashboard → Pipeline → Run pipeline job**. The API:

1. Validates `viewer_operator.yaml` declares which container variant to use (`pipeline_install_extras: ml` for ml/airgapped, `llm` for cloud-thin — see [Operator config](#operator-config) below for how this is set automatically when you save a profile via the UI).
2. Spawns a one-shot pipeline container via `docker compose run --rm pipeline{,-llm}` with `--config <viewer_operator.yaml> --feeds-spec <feeds.spec.yaml> --profile <name>`.
3. The pipeline container processes each feed: download → transcribe → summarize → extract GI / KG → write artifacts under `<corpus>/feeds/<stable>/run_<id>/`.
4. The container exits when done; the API's `/api/jobs` endpoint reflects the status (`running` → `succeeded` or `failed`).
5. The viewer auto-loads new artifacts: Library shows the produced episodes, Digest summarizes them, Graph/Search expose the entities and topics.

Watch the job log via the **Pipeline Jobs** card in the dashboard, or tail it from the host:

```bash
docker compose -f compose/docker-compose.stack.yml -f compose/docker-compose.stack-test.yml logs -f api
```

(`api` proxies pipeline stdout into `<corpus>/.viewer/jobs/<job_id>.log`; `docker logs compose-api-1` works too.)

A typical airgapped run on a laptop: ~60–90 s for one short episode (Whisper transcription dominates). Cloud-thin: ~30–45 s (network-bound).

### 6. Browse results

- **Library** — feed/episode list with status badges and per-episode rails.
- **Digest** — time-windowed summary of the latest episodes; honours the **All time** lens for static-date fixtures.
- **Search** — semantic FAISS search (only when the active profile sets `vector_search: true` — `airgapped_thin` does, `cloud_thin` does not).
- **Graph** — Cytoscape graph of topics, entities, episodes, and bridges.

## Persistence

| Item | Where it lives | What happens on `stack-test-down` |
| ---- | -------------- | --------------------------------- |
| Corpus artifacts (transcripts, gi/kg JSON, run logs) | named volume `compose_corpus_data` mounted at `/app/output` | preserved unless you pass `STACK_TEST_DOWN_VOLUMES=1` |
| Operator config (`viewer_operator.yaml`) + feeds (`feeds.spec.yaml`) | corpus root in `corpus_data` | preserved (same volume) |
| Vue SPA build | baked into the `viewer` image | lost on image rebuild |
| Pipeline job logs | `<corpus>/.viewer/jobs/<id>.log` in `corpus_data` | preserved |
| HuggingFace model cache (sentence-transformers, spaCy, Whisper) | baked into the `api` and `pipeline` images at build time | lost on image rebuild |

To start completely fresh (drop all data and rebuild):

```bash
make stack-test-down STACK_TEST_DOWN_VOLUMES=1
make stack-test-build
```

## Operator config

The viewer drives a thin operator-config layer. When you **Save** a profile via the UI, the API writes `<corpus>/viewer_operator.yaml` containing:

- `profile: <name>` — the packaged profile to merge from.
- A handful of operator-tunable fields (`max_episodes`, `workers`, `transcribe_missing`, …).
- `pipeline_install_extras: ml` or `llm` — tells the API which compose service to spawn (`pipeline` for ml, `pipeline-llm` for llm).

For local stack-test runs, `make stack-test-seed` lays down a starter `viewer_operator.yaml` and `feeds.spec.yaml` in the corpus volume so the UI has something to build on. To drive the cloud variant from the start, run:

```bash
make stack-test-seed STACK_TEST_OPERATOR_VARIANT=cloud-thin
```

…then choose `cloud_thin` in the **Job Profile** dropdown when saving.

## Inspect artifacts on disk

If you want to look at the produced files outside the running stack:

```bash
make stack-test-export      # copies corpus_data → ./.stack-test-corpus/
ls .stack-test-corpus/feeds/*/run_*/metadata/
```

You'll see `<episode>.metadata.json`, `<episode>.gi.json`, `<episode>.kg.json` per episode, plus `transcripts/<episode>.txt`, `index.json`, `metrics.json`, and `run.json` per run, and `corpus_run_summary.json` at the corpus root.

## Updating

After pulling new code:

```bash
make stack-test-down
make stack-test-build      # rebuilds api, viewer, pipeline (cached layers reused where unchanged)
make stack-test-up
```

The named `corpus_data` volume survives the rebuild — your existing artifacts and operator config stay intact.

## Tear down

```bash
make stack-test-down                         # stop the stack, keep the corpus
make stack-test-down STACK_TEST_DOWN_VOLUMES=1   # also drop the corpus_data volume
```

## Architecture

```text
┌──────────────┐    HTTP    ┌──────────────┐
│  Browser     │───────────▶│  viewer      │  Nginx
│  (you)       │            │  (8090:80)   │  serves SPA + proxies /api/*
└──────────────┘            └──────┬───────┘
                                   │ /api/*
                                   ▼
                            ┌──────────────┐
                            │  api         │  FastAPI
                            │              │  - operator-config endpoint
                            │              │  - jobs API (POST /api/jobs)
                            │              │  - corpus + search endpoints
                            └──────┬───────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼ /var/run/docker.sock     ▼  named volume            ▼
┌──────────────┐            ┌──────────────┐          ┌──────────────┐
│ Docker host  │            │ corpus_data  │          │ mock-feeds   │
│              │            │ (artifacts,  │          │  (Nginx,     │
│  pipeline /  │            │  transcripts)│          │   fixtures)  │
│  pipeline-   │            │              │          │              │
│  llm         │  ◀─────────┤              │          │ /feed.xml    │
│  (one-shot)  │  reads/   ▲│              │          │ /audio/*.mp3 │
└──────────────┘  writes   ││              │          │ /transcripts │
                           │└──────────────┘          └──────────────┘
                           │
                           └─ api also reads via /app/output mount
```

### Services

- **`viewer`** — Nginx serving the prebuilt Vue SPA. Proxies `/api/*` to the API container. Internal port 80; exposed on `STACK_TEST_VIEWER_PORT` (default 8090).
- **`api`** — FastAPI (uvicorn). Reads/writes the corpus volume directly (operator-config, feeds.spec, jobs registry). Talks to the host Docker daemon via the bind-mounted socket to spawn pipeline jobs. Internal port 8000.
- **`mock-feeds`** — small Nginx sidecar that serves bundled RSS fixtures, audio, and transcripts at `http://mock-feeds/...` on the compose network. Lets you exercise the platform without internet access. **Remove this service in production deployments** — see [Production hints](#production-hints).
- **`pipeline`** / **`pipeline-llm`** — ephemeral. Built into `compose/docker-compose.stack.yml` under Compose profiles `pipeline` / `pipeline-llm`. Spawned by the API factory per job; not part of `compose up`.

### Volumes

- **`corpus_data`** — read-write on `api`, `pipeline`, and `pipeline-llm`. Holds operator config, feeds spec, run artifacts, transcripts, search index. The viewer reads it indirectly through the API.

### Network

Default Compose network `compose_default`. All services resolve each other by service name (e.g. the pipeline reaches `http://mock-feeds` and `http://api:8000`).

## Profiles and image variants

Two compose service profiles map to two image tiers:

| Compose service | Image | Built by | Pipeline profile that uses it | Roughly |
| --------------- | ----- | -------- | ----------------------------- | ------- |
| `pipeline` | `podcast-scraper-stack-pipeline` (`INSTALL_EXTRAS=ml`) | `make stack-test-build` | `airgapped`, `airgapped_thin`, `local`, `dev`, `cloud_balanced`, `cloud_quality` (anything that needs local Whisper / spaCy / transformers / FAISS) | ~5 GB image; full local pipeline. |
| `pipeline-llm` | `podcast-scraper-stack-pipeline-llm` (`INSTALL_EXTRAS=llm`) | `make stack-test-build-cloud` | `cloud_thin` | ~500 MB image; cloud APIs only. |

The mapping happens via `pipeline_install_extras` in the operator yaml — when you save a profile in the UI, the API writes `pipeline_install_extras: ml` or `llm`, then the next job spawn picks the right service.

For more on image-variant trade-offs, see [Docker variants guide](DOCKER_VARIANTS_GUIDE.md).

## Troubleshooting

### `make stack-test-up` fails with "permission denied" on Docker socket

You're on Linux and your user isn't in the `docker` group, or you're on a system where the socket has an unusual GID. The platform's API entrypoint adds the `podcast` user to the socket's host group when the GID is non-zero. If that fails, the spawned pipeline container can't be created.

Workaround for development: the stack-test overlay sets `PODCAST_KEEP_ROOT=1` so the API runs as root (which always has socket access). Production deployments should fix the GID alignment instead — `usermod -aG docker $USER` on the host, then re-login.

### Pipeline job fails with "Config file not found: /app/config.yaml"

Stale state from before the cleanup that retired the `/app/config.yaml` mount. Pull the latest code, rebuild images:

```bash
git pull
make stack-test-down STACK_TEST_DOWN_VOLUMES=1
make stack-test-build
make stack-test-up
```

### Pipeline job fails with "OpenAI API key required" (cloud profiles)

Either:

- `.env` doesn't exist at the repo root, or
- `make stack-test-up` was called from a shell that didn't source `.env` (the Make target sources it automatically, but if you ran `docker compose up` directly you need `--env-file ./.env`).

Restart with the source target:

```bash
make stack-test-down
make stack-test-up
```

### "embed_failed: couldn't connect to huggingface.co" on /api/search

The API image must have the embedding model preloaded at build time. If you have a stale image, rebuild:

```bash
make stack-test-down
make stack-test-build
make stack-test-up
```

### Spawned pipeline container appears outside the project tree in Docker Desktop

Expected. The pipeline runs as a one-shot `docker compose run --rm` container (not a long-running service); Docker Desktop's Compose view groups long-running services and lists `run` containers separately even though they share the project label. To inspect anyway:

```bash
docker ps -a --filter "label=com.docker.compose.project=compose"
```

The container has the `compose_default` network and shares the `corpus_data` volume.

### Job hangs at "running" forever

Get the spawned container's stdout from the API job log:

```bash
docker logs compose-api-1 2>&1 | grep "docker job spawn"
docker exec compose-api-1 sh -c 'cat /app/output/.viewer/jobs/*.log' | tail -200
```

Common root causes: pipeline image was built without ffmpeg (rebuild), or the pipeline can't reach `mock-feeds` (verify the service is healthy: `docker ps`), or the operator yaml's profile selected an unknown provider (check `cat /app/output/viewer_operator.yaml`).

### Cloud profile + airgapped image (or vice versa)

If you select `cloud_thin` in the UI but `viewer_operator.yaml` still says `pipeline_install_extras: ml`, the API spawns the wrong (heavy ML) image and the cloud calls fall back oddly. The UI **Save** action should set the right `pipeline_install_extras` based on the profile — if it doesn't (or you're seeding manually), use:

```bash
make stack-test-seed STACK_TEST_OPERATOR_VARIANT=cloud-thin
```

…before the Playwright test or your first job.

## Production hints

The `compose/docker-compose.stack.yml` + `compose/docker-compose.stack-test.yml` setup ships everything you need for a single-corpus, single-host deployment. To take it production-ish:

1. **Drop the `mock-feeds` service.** Remove `compose/docker-compose.stack-test.yml` from your `-f` chain (or write a thin overlay that omits it). Real RSS URLs go in `feeds.spec.yaml` directly via the UI.
2. **Add restart policies.** Merge [`compose/docker-compose.prod.yml`](../../compose/docker-compose.prod.yml) — adds `restart: unless-stopped` for `api` and `viewer`.
3. **Externalise the corpus volume.** Change `corpus_data` to an external named volume so it survives `docker compose down -v`. Or bind-mount a host path.
4. **Front the viewer with HTTPS.** Put a reverse proxy (Caddy, Traefik, Nginx) in front of `viewer:80` — the SPA itself doesn't terminate TLS.
5. **Lock down the API.** The current setup assumes the viewer host is trusted. Add network policies or an authenticating proxy if `:8090` is reachable from anywhere outside the host.
6. **Provider keys.** Use Docker secrets or environment-variable injection from your orchestrator instead of a checked-in `.env`. The compose file reads `${OPENAI_API_KEY:-}` etc. at `up` time.

For full RFC-079 / GitHub #659 architecture details, see [RFC-079 — Full-stack Docker Compose](../rfc/RFC-079-full-stack-docker-compose.md).

## Reference

- [README — Quick start with Docker Compose](../../README.md#quickest-start-docker-compose) — the 4-step landing version of this guide.
- [`compose/`](../../compose/) — compose files: `docker-compose.stack.yml` (base), `docker-compose.stack-test.yml` (overlay used for the bundled mock-feeds + Docker job mode), `docker-compose.jobs-docker.yml` (production overlay enabling Docker job mode without mock-feeds), `docker-compose.prod.yml` (restart policies).
- [`docker/`](../../docker/) — image Dockerfiles for `api`, `viewer`, `pipeline`, and the `mock-feeds` Nginx sidecar.
- [Stack-test README](../../tests/stack-test/README.md) — how the Playwright suite drives the same compose stack.
- [Docker variants guide](DOCKER_VARIANTS_GUIDE.md) — pipeline image tier comparison (ml vs llm).
- [Server guide](SERVER_GUIDE.md) — `/api/*` reference, FastAPI architecture.
- [Configuration guide](../api/CONFIGURATION.md) — every config field, including operator-yaml-friendly ones.
- [Profiles README](../../config/profiles/README.md) — packaged profiles and their trade-offs.
