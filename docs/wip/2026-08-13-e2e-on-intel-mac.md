# Running the consumer e2e suite on an Intel Mac (container harness)

**Verified working 2026-08-13: `83 passed` — both Playwright projects, full suite, including the
search-dependent specs.** Before this, none of it could run on this machine.

## Why a container is needed

`.[search]` (torch + lancedb + sentence-transformers) **cannot be installed on macOS x86_64** — both
projects stopped publishing Intel-Mac wheels well below this repo's floors:

| package | repo floor | newest macOS x86_64 wheel |
| ------- | ---------- | ------------------------- |
| `torch` | `>=2.11.0` | **2.2.2** |
| `lancedb` | `>=0.33.0` | **0.25.3** |

So the LanceDB index cannot be built natively, and without it `globalSetup` fails and every
index-dependent spec (search, perspectives, consolidation) is skipped or wrong. `docker/api/Dockerfile`
installs the full stack on linux/amd64 and preloads MiniLM, which is the whole fix. See
`intel-mac-blocks-ml-extras` in the session memory for the full wheel survey.

## One-time

```bash
# BuildKit is required (the Dockerfile uses --mount=type=cache) and OrbStack ships buildx
# unlinked for non-primary users:
mkdir -p ~/.docker/cli-plugins
ln -sf /Applications/OrbStack.app/Contents/MacOS/xbin/docker-buildx ~/.docker/cli-plugins/docker-buildx

DOCKER_BUILDKIT=1 docker build -f docker/api/Dockerfile -t podcast-api:e2e-local .   # ~2.9 GB

# The corpus dir must be writable by the DOCKER DAEMON's user, not yours — see the gotcha below.
chmod 777 tests/fixtures/app-validation-corpus/v3/search

docker run --rm --entrypoint python \
  -v "$PWD/tests/fixtures/app-validation-corpus/v3:/corpus" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  podcast-api:e2e-local -m podcast_scraper.cli index-two-tier --output-dir /corpus
# → "Two-tier index built: episodes=36 segments=131 insights=124 aux=593"
```

The KMeans "more than 10% of clusters are empty" warnings are expected — 593 vectors against a
65,536 threshold. A small fixture corpus, not a fault.

## Per run

**Recreate the volume every time.** This is not optional — see the isolation note below.

```bash
docker rm -f lp-e2e-api; docker volume rm lp-e2e-appdata; docker volume create lp-e2e-appdata

docker run -d --name lp-e2e-api -p 127.0.0.1:8011:8011 \
  -v "$PWD/tests/fixtures/app-validation-corpus/v3:/corpus" \
  -v lp-e2e-appdata:/appdata \
  -e APP_OAUTH_PROVIDER=mock -e APP_SESSION_SECRET=e2e-secret -e APP_SIGNUP_MODE=open \
  -e APP_PERSONALIZED_RANKING=true -e APP_TRENDING_NOW=2026-07-20T00:00:00Z \
  -e APP_DATA_DIR=/appdata -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint python podcast-api:e2e-local \
  -m podcast_scraper.cli serve --output-dir /corpus --port 8011 --host 0.0.0.0

cd web/learning-player && npx playwright test        # stock config; reuses the running :8011
docker rm -f lp-e2e-api                              # reap when done
```

The **stock `playwright.config.ts` is used unmodified**: its API `webServer` has
`reuseExistingServer: !CI`, so a healthy `:8011` is adopted and the venv Python is never invoked;
and `globalSetup` skips its index build because the index now exists.

## Two gotchas that cost real time

**1. The container writes as the Docker daemon's user, not yours.** OrbStack runs as
the host desktop user; this checkout is owned by the agent user `claude` at mode 755. So a container write into the
repo is denied *regardless of the container's own uid* — `--user $(id -u)` does not help, because
the identity that matters is host-side. Hence the `chmod 777` on the corpus `search/` dir.

It also means files the container creates in a bind mount **cannot be deleted by you afterwards**.
If that happens, delete them with a throwaway container:

```bash
docker run --rm -v "$PWD/web/learning-player/e2e:/e2e" --entrypoint sh podcast-api:e2e-local \
  -c 'rm -rf /e2e/.app-state'
```

**2. Test isolation moved from `globalSetup` to the volume — and it is easy to lose.** Normally
`globalSetup` wipes `e2e/.app-state` so `signInIsolated`'s *stable* per-(spec, project) user ids
start clean each run. With `APP_DATA_DIR` on a Docker volume there is no host dir to wipe, so that
wipe becomes a no-op and **state persists across runs**.

Observed directly: leaving the volume in place produced one failure —
`consolidation.spec.ts` expecting "Nothing to revisit right now" while a previous run's captures were
still there. That is the exact class of leak `globalSetup`'s comment warns about, reintroduced by
the harness rather than the app. Recreating the volume made the suite green.

If this harness is ever promoted beyond local use, that responsibility should move into
`globalSetup` (or a make target) rather than staying a step someone has to remember.

## Not covered

The **operator** viewer's suite is not addressed here — 33 of its 38 specs still route-mock rather
than using a real backend (#1619). Its tier-3 `validation/` specs have their own config and a
separately-booted stack, untested on this machine.
