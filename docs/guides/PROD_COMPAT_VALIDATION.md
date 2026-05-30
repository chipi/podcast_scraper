# Prod code/content compatibility — validation guide

**Companion to:** [Prod Runbook](PROD_RUNBOOK.md) § [Code/content compatibility](PROD_RUNBOOK.md#codecontent-compatibility)

Operator and developer checklist for verifying [#796](https://github.com/chipi/podcast_scraper/issues/796) (version contract, deploy hardening), [#797](https://github.com/chipi/podcast_scraper/issues/797) (smoke script, dependency map, release matrix), [#798](https://github.com/chipi/podcast_scraper/issues/798) (compose backup-restore verify), and [#799](https://github.com/chipi/podcast_scraper/issues/799) (scheduled DR drill). Run **Tier 1–2** before every PR; **Tier 3–4** before release; **Tier 5–8** only on drill/prod hosts or post-merge on `main`.

---

## What each tier proves

| Tier | Scope | Time | Requires |
| --- | --- | --- | --- |
| **1 — Local automated** | Lint, types, pytest, Vitest, Playwright, docs, workflow YAML | ~15 min | Repo checkout only |
| **2 — Local API + curl** | Live `/api/health?path=`, six surface probes on `make serve-api` | ~10 min | Port 8000 free |
| **3 — Operator CLI** | `make corpus-compat-check`, release-script gates | ~2 min | Corpus directory |
| **4 — CI** | GitHub job `N-1 corpus compat (current code × prior fixture)` | CI | Push / PR |
| **5 — Drill deploy** | `drill-deploy.yml` SHA validation + `DEPLOY_GIT_SHA` on drill VPS | ~25 min | Drill secrets + GHCR image |
| **6 — Prod deploy + smoke** | `deploy-prod.yml` + `make smoke-prod` over tailnet HTTPS | ~30 min | Prod secrets, populated corpus |
| **7 — Backup restore verify (#798)** | `verify-backup-restore.yml` on `main` (compose + smoke) | ~10 min | `BACKUP_REPO_TOKEN`, GHCR read |
| **8 — DR drill exercise (#799)** | `drill-exercise.yml` full cycle + `assert-post-conditions` | ~45 min | Drill env secrets, Hetzner drill project |

**Stop before Tier 6** when validating a branch locally — prod dispatch is operator-only and restarts prod CI. Tiers **7–8** run on **`main`** after merge (cron or manual dispatch).

---

## Tier 1 — Local automated

From repo root:

```bash
make lint && make type

.venv/bin/python -m pytest \
  tests/integration/server/test_corpus_version_compat.py \
  tests/integration/server/test_viewer_api.py::test_health_ok \
  tests/unit/podcast_scraper/workflow/test_corpus_operations_incident_rollup.py::TestWriteCorpusManifestProducedBy \
  tests/unit/scripts/test_create_release_notes_draft.py \
  -v --tb=short

cd web/gi-kg-viewer
npm run test:unit -- src/stores/shell.health.test.ts
./node_modules/.bin/playwright install firefox   # once per machine / CI cache miss
npm run test:e2e -- e2e/corpus-version-warning.spec.ts

cd ../..
make lint-markdown && make docs
actionlint .github/workflows/drill-deploy.yml .github/workflows/deploy-prod.yml \
  .github/workflows/drill-exercise.yml .github/workflows/drill-e2e.yml \
  .github/workflows/verify-backup-restore.yml .github/workflows/python-app.yml
```

**Pass:** all commands exit 0.

**Maps to:** N-1 compat tests, `produced_by` manifest stamp, release-notes `COMPATIBILITY.md` gate, viewer warning banner E2E, workflow YAML syntax.

---

## Tier 2 — Local API + curl

Ensure nothing else is bound to port **8000** (stale servers omit `code_version` / `corpus_version_warning`).

```bash
mkdir -p .test_outputs
make serve-api SERVE_OUTPUT_DIR=./.test_outputs
```

In another terminal:

```bash
# Default output_dir
curl -fsS http://127.0.0.1:8000/api/health | jq \
  '.code_version, .min_supported_corpus_code_version, .corpus_version_warning'

# Viewer-entered path (must be under SERVE_OUTPUT_DIR anchor)
CORPUS="$(pwd)/.test_outputs"
curl -fsS --get 'http://127.0.0.1:8000/api/health' \
  --data-urlencode "path=$CORPUS" | jq \
  '.corpus_code_version, .corpus_version_warning'

# Escape attempt → 400
curl -sS -w '\nHTTP %{http_code}\n' --get 'http://127.0.0.1:8000/api/health' \
  --data-urlencode 'path=/etc/passwd'
```

**Pass:**

- `code_version` and `min_supported_corpus_code_version` present.
- Empty corpus under anchor → non-null `corpus_version_warning` (no `produced_by`).
- `/etc/passwd` → HTTP **400**, not 500.

**Note:** N-1 fixture paths outside the server anchor return **400** on live curl — that case is covered by pytest (`test_corpus_version_compat.py`) which mounts the fixture as `output_dir`.

### Six surface probes (smoke script parity)

Substitute localhost for tailnet HTTPS when dry-running locally:

```bash
CORPUS="$(pwd)/.test_outputs"
ENC=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$CORPUS")
BASE=http://127.0.0.1:8000

for route in \
  "/api/health" \
  "/api/corpus/episodes?path=${ENC}&limit=1" \
  "/api/corpus/digest?path=${ENC}&window=all" \
  "/api/artifacts?path=${ENC}" \
  "/api/corpus/topic-clusters?path=${ENC}" \
  "/api/search?path=${ENC}&q=ai&top_k=1"; do
  code=$(curl -sS -o /dev/null -w '%{http_code}' "${BASE}${route}")
  echo "$code $route"
done
```

**Pass:** health/episodes/digest/artifacts/search → **200**; topic-clusters → **200** or **404** when index not built (empty `.test_outputs` often **404** — acceptable locally; prod smoke with `EXPECT_POPULATED=1` requires clusters).

---

## Tier 3 — Operator CLI

```bash
# Expect exit 1 + WARNING when corpus lacks produced_by
make corpus-compat-check CORPUS_DIR=./.test_outputs; echo "exit=$?"

# Release gates
.venv/bin/python scripts/pre_release_check.py
.venv/bin/python scripts/tools/create_release_notes_draft.py
```

**Pass:** compat check prints `server=…`, `corpus_code_version=…`, exits **1** when warning would appear in health, **0** when compatible; release scripts exit **0** when `docs/COMPATIBILITY.md` lists the shipping version.

Optional end-to-end stamp:

```bash
make reprocess-corpus-from-transcripts CORPUS_DIR=/path/to/corpus
jq '.produced_by' /path/to/corpus/corpus_manifest.json
make corpus-compat-check CORPUS_DIR=/path/to/corpus
```

---

## Tier 4 — CI

Push branch; confirm green job:

**`N-1 corpus compat (current code × prior fixture)`** — runs `tests/integration/server/test_corpus_version_compat.py` (includes `test_health_path_query_scopes_version_warning`).

---

## Tier 5 — Drill deploy (not local)

```bash
gh workflow run drill-deploy.yml --repo chipi/podcast_scraper
gh run watch --repo chipi/podcast_scraper
```

**Pass in run log:**

1. SHA regex rejects invalid `override_image_sha`.
2. **validate: GHCR image manifest exists** before SSH.
3. Deploy SSH includes `DEPLOY_GIT_SHA=… DEPLOY_IMAGE_SHA=… deploy.sh`.
4. Post-deploy `/api/health` via SSH OK.

Negative: dispatch with `override_image_sha=notahex` → fails before SSH.

---

## Tier 6 — Prod deploy + smoke (operator only)

Do **not** run as part of local branch validation unless you intend to change prod.

```bash
gh workflow run deploy-prod.yml --repo chipi/podcast_scraper \
  -f confirm=PROD_DEPLOY
gh run watch --repo chipi/podcast_scraper

export PROD_TAILNET_FQDN=prod-podcast.tail-xxxxx.ts.net
# In-container corpus root (PODCAST_DEFAULT_CORPUS_PATH), not the host bind path
export SMOKE_CORPUS_PATH=/app/output
make smoke-prod

# Optional: fail fast before dispatch (same check as deploy-prod preflight step)
bash scripts/ops/preflight_prod_corpus_path.sh deploy@prod-podcast.tail-xxxxx.ts.net
```

**Pass:** workflow green; preflight step confirms host `PODCAST_CORPUS_HOST_PATH` exists; smoke exits **0** with six probes including **`artifacts count≥1`** on populated prod.

See [Prod Runbook — Code/content compatibility](PROD_RUNBOOK.md#codecontent-compatibility) for the decision tree before dispatch.

---

## Tier 7 — Backup restore verify (#798, post-merge on `main`)

```bash
gh workflow run verify-backup-restore.yml --repo chipi/podcast_scraper
gh run watch --repo chipi/podcast_scraper
```

**Pass:** job green; log shows restore from **`snapshot-prod-*`**, compose up, **`post_deploy_smoke`** six probes on `http://127.0.0.1:8090`, teardown.

**Cron:** Sundays 04:00 UTC. Requires **`BACKUP_REPO_TOKEN`** (or readable backup repo via `GITHUB_TOKEN`). Optional **`SMOKE_WEBHOOK_URL`** for failure alerts.

---

## Tier 8 — DR drill exercise (#799, post-merge on `main`)

```bash
gh workflow run drill-exercise.yml --repo chipi/podcast_scraper \
  -f confirm=DRILL_FULL_CYCLE
gh run watch --repo chipi/podcast_scraper
```

**Pass:** all orchestrator jobs **`success`**; **`assert-post-conditions`** green; **`delete_drill_hetzner_orphans --check-only`** finds no orphans.

**Cron:** Wednesdays 02:00 UTC (no typed confirm). Scheduled failures ping **`SMOKE_WEBHOOK_URL`** when set.

---

## Viewer manual check (optional)

After `make serve` (API + UI):

1. Open `http://localhost:5173/`.
2. Set status-bar corpus path to a directory without `produced_by`.
3. **Retry health** — yellow banner `corpus-version-warning-banner`.
4. Network tab: `GET /api/health?path=…` matches the path field.

Automated coverage: `corpus-version-warning.spec.ts`, `shell.health.test.ts`.

---

## Minimum merge bar

Before merging a compat/smoke PR without prod deploy:

1. Tier 1 all green.
2. Tier 2 health + bad-path curl on fresh `make serve-api`.
3. Tier 3 `corpus-compat-check` on a known stale corpus (exit 1).
4. Tier 4 CI job green on the PR.

---

## Related docs

| Doc | Role |
| --- | --- |
| [PROD_RUNBOOK.md](PROD_RUNBOOK.md) | Deploy, rollback, compat decision tree |
| [CORPUS_ARTIFACTS_AND_SURFACES.md](../architecture/CORPUS_ARTIFACTS_AND_SURFACES.md) | Artifact ↔ viewer surface map |
| [COMPATIBILITY.md](../COMPATIBILITY.md) | Release matrix |
| [CORPUS_MULTI_FEED_ARTIFACTS.md](../api/CORPUS_MULTI_FEED_ARTIFACTS.md) | `corpus_manifest.json` + `produced_by` schema |
