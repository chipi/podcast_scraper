# Observability extensions (optional 3rd-party o11y)

Every 3rd-party observability integration (error reporting, LLM tracing, …) is a
**fully optional, opt-in extension**. The core package carries **zero** o11y-SaaS
dependencies and runs identically whether or not any of them are installed. You opt
them into the **images you deploy** — not into the library everyone installs.

## The model

There are two distinct kinds of o11y dependency, kept separate on purpose:

| Kind | What | Dependency | Lives in |
| --- | --- | --- | --- |
| **Read** | The control plane *queries* o11y backends (Grafana / Loki / Sentry API / Langfuse API) | `httpx` only — no vendor SDK | `[observability]` (the light `podcast_obs` control plane) |
| **Emit** | The app *sends* data to an o11y service (Sentry exceptions, Langfuse spans) | the service's vendor SDK | one **per-service extra** — see below |

### Per-service emit extras

| Extra | Service | Used by | Secret gate (no-op until set) |
| --- | --- | --- | --- |
| `[sentry]` | Sentry error reporting | api (FastAPI) + pipeline (cli) | `PODCAST_SENTRY_DSN_API` / `PODCAST_SENTRY_DSN_PIPELINE` |
| `[langfuse]` | Langfuse LLM tracing | pipeline (every LLM call) | `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` |

- **Core** (`[project.dependencies]`): none of these. The package imports and runs
  without any of them.
- **`[dev]`**: self-references *all* emit extras (`podcast-scraper[sentry]`,
  `podcast-scraper[langfuse]`) so tests/CI/`make install` exercise both the
  present-path (mocked SDK) and the absent-path (simulated).
- **Deployed images** opt in only what they use:
  - `docker/api` → `.[search,sentry]` (inits Sentry; makes no LLM calls)
  - `docker/pipeline` → `.[<ml|llm>,search,sentry,langfuse]` (errors + LLM tracing)
  - `docker/observability` (control plane) → **neither** — it only *reads*.

## The three contracts every extension obeys

1. **Lazy, guarded import** — never a top-level `import <sdk>`. Import inside the
   function that needs it, wrapped in `try/except ImportError` → no-op. So an install
   *without* the extra never breaks.
2. **Config-gated** — even when the SDK is installed, the integration is a true no-op
   until its secret(s) are present (Sentry DSN, Langfuse keys). Default boots stay silent.
3. **Never breaks the app** — any failure inside the integration is caught and logged at
   debug; the pipeline/api keeps running. The worst case is a missing span, never a crash.

These three are what make an o11y dep safe to keep out of core.

## Enable one

```bash
# install the service you want (or bake it into your deployed image)
pip install -e '.[sentry]'      # or .[langfuse], or .[sentry,langfuse]
# then set its secret(s) — until then it stays a no-op even when installed
export PODCAST_SENTRY_DSN_PIPELINE=...     # Sentry
export LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=...   # Langfuse
```

Don't want any of it? Install nothing — the core no-ops.

## How to add a new o11y extension

Say you want to add `[honeycomb]` (or any o11y SaaS). Five steps, mirroring `[sentry]` /
`[langfuse]`:

1. **Extra** — add a per-service block in `pyproject.toml`
   `[project.optional-dependencies]`:

   ```toml
   honeycomb = [
     "libhoney>=2.4,<3.0",  # no-op without HONEYCOMB_API_KEY
   ]
   ```

2. **`[dev]` self-ref** — add `"podcast-scraper[honeycomb]"` to `[dev]` so tests/CI get it.
3. **Integration module** — a small `utils/<service>_*.py` that obeys the three contracts:
   lazy guarded import, config-gated (`os.environ.get("HONEYCOMB_API_KEY")`), wrapped so it
   can never raise into the caller. Call it from the right hook (e.g. the provider cost
   choke point `record_provider_call_cost`, or `init_*` at app startup).
4. **Tests** — cover both paths:
   - present + configured → emits the right shape (mock the SDK);
   - **absent** → no-op (`monkeypatch.setitem(sys.modules, "libhoney", None)`), like
     `test_noop_when_sdk_unimportable` / `test_returns_false_when_sentry_sdk_unimportable`.
5. **Compose** — add `,honeycomb` to the install line of each **deployed image** that should
   emit it (`docker/api`, `docker/pipeline`), and document the secret in
   `config/examples/.env.example` + this table.

That's it — core stays clean, library users can decline, and your deployed images carry
exactly the o11y you chose.

## Why this shape

You deploy your own images, so that's where 3rd-party o11y belongs — opted in per image.
Anyone consuming the library (or running the light control-plane container) gets none of
it by default and can add only what they want. One principle, no special cases.

## Operator alerting — Sentry + Grafana

Alerts are configured **operator-side, on the o11y vendor**, not in this codebase. The
deployed images already emit everything alerts need; you only wire the thresholds on the
vendor that owns the signal. This is the path RFC-043 redirected to: keep the codebase
neutral, let operators tune alert rules where they belong.

### What gets emitted today

| Signal | Surface | Source |
| --- | --- | --- |
| Unhandled exceptions (api + pipeline) | Sentry | `[sentry]` extra + `PODCAST_SENTRY_DSN_*` |
| LLM call spans (latency, tokens, model, cost) | Langfuse | `[langfuse]` extra + Langfuse keys |
| Metric snapshots (`metrics/latest.json`, `nightly-latest.json`) | GitHub Pages JSON | `scripts/dashboard/generate_metrics.py` (always on in CI) |
| Nightly regression alerts (`alerts[]` array in nightly bundle) | GHA job summary | `detect_deviations()` in `generate_metrics.py` |

The Grafana control plane (`docker/observability`) reads from Sentry / Loki / Langfuse /
the GitHub Pages JSON — it doesn't host its own metric store. Alert rules live in Grafana
panels and Sentry projects.

### Sentry — error and performance alerts

Sentry handles **unhandled exceptions** and **performance regressions** on the API and
pipeline images.

1. Enable for the image:

   ```bash
   pip install -e '.[sentry]'
   export PODCAST_SENTRY_DSN_API=https://...        # FastAPI image
   export PODCAST_SENTRY_DSN_PIPELINE=https://...   # pipeline image
   ```

2. In Sentry → **Alerts → Create Alert Rule**:

   - **Issue alerts** — fire on `An issue is created` for the project, route to
     email / Slack / PagerDuty. This is the high-signal error-rate alert.
   - **Metric alerts** — fire on `Number of errors` > N in M minutes, or on
     `Failure rate` > X% over a rolling window. Use these to catch regression
     spikes without waiting for someone to triage individual issues.
   - **Performance alerts** — fire on transaction p95 latency exceeding a baseline.
     Useful for the FastAPI image (`/api/*` routes) once a baseline is established.

3. Set environments to `prod` and `pre-prod` so non-deployed test failures don't page.

No code change needed — the Sentry SDK is already initialised conditionally on DSN
presence.

### Grafana — metric trend alerts

Grafana handles **trend-over-time alerts** on the JSON metric series the dashboard
already pulls.

1. Add a Grafana panel sourced from
   `https://<gh-pages-host>/metrics/history.jsonl` (the file the dashboard already
   reads). Use the JSON datasource or the Infinity plugin.

2. On the panel → **Alert → Create alert rule**:

   - **Coverage drop** — `combined_coverage_pct` decreasing by > 1pp vs. 7-day average.
   - **Test runtime regression** — `ci_total_seconds` increasing by > 20% vs. baseline.
   - **Test failures on main** — `failed_tests_total > 0` on `branch == main`.
   - **Nightly alert count spike** — `len(alerts) > 0` (the field
     `detect_deviations()` already populates).

3. Route via Grafana **Notification Policies** → Slack / email / PagerDuty.

The `metrics/history.jsonl` file is append-only and committed to `gh-pages`, so the
alert rule reads a stable, versioned source. Alert thresholds live in Grafana, not in
the repo — operator tunes them per environment.

### LLM cost alerts

Langfuse traces every LLM call with model / token / cost dimensions. In Langfuse →
**Alerts**, set thresholds on:

- Daily cost per model exceeding `$X`.
- Average latency per model exceeding `Y ms` over a 1-hour window.
- Error rate per provider (e.g. OpenAI 429s spiking).

This is the right place for cost alerts because Langfuse already has the dimensional
data; replicating it in Sentry or Grafana would require shipping the same spans twice.

### Why not PR comments + webhook scripts

The original RFC-043 plan was to ship `scripts/generate_pr_comment.py` and
`scripts/send_webhook_alert.py`. That path was abandoned because:

- Sentry + Grafana already cover the **production** alerting need (errors, latency,
  cost, metric trends).
- PR comments on metric drift would duplicate the GHA job summary the dashboard
  already renders — adding cognitive noise without adding signal.
- Webhook scripts would have been a third place to maintain alert rules; keeping
  thresholds in the vendor that owns the signal is simpler.

If a future need surfaces for an automated PR-comment narration (e.g. coverage
deltas summarised on each PR), revisit it then. The o11y surface here is sufficient
for the proactive-alerting goal.

## RFC-088 enrichment-layer MCP tools

The `podcast_obs` control plane (read-only) ships eight MCP tools that
operate on the same enrichment surface the viewer Configuration popup
consumes. Available wherever `podcast-obs serve` runs (stdio / sse /
streamable-http):

| Tool | What it answers | HTTP backend |
| ---- | --------------- | ------------ |
| `enrichment_run_status` | Last enrichment status snapshot | `GET /api/enrichment/status` |
| `enrichment_recent_runs(limit=10)` | Newest enrichment-only jobs | `GET /api/jobs` filtered to `command_type=corpus_enrichment` |
| `enrichment_health(enricher_id?)` | Per-enricher health (or a single record) | `GET /api/enrichment/health` |
| `enrichment_metrics(window="24h")` | Rollup metrics window | `GET /api/enrichment/metrics` |
| `enrichment_recent_events(enricher_id?, event_type?, limit=50)` | JSONL event tail | `GET /api/enrichment/events` |
| `enrichment_eval_history(eval_root?, limit=10)` | Enrichment-tagged eval runs on disk | local scan (operator-side; eval artefacts are frozen-once-written) |
| `enrichment_re_enable(enricher_id, reason)` | Operator manual recovery | `POST /api/enrichment/health/{id}/re-enable` |
| `enrichment_cancel(job_id)` | Cancel a running/queued enrichment job | `POST /api/jobs/{id}/cancel` |

`prod_correlate(run_id)` joins enrichment events for a `run_id` into
its cross-layer view (pipeline trace + Langfuse + Loki + Sentry +
enrichment). `prod_summary` adds three enrichment subsections
(`enrichment_status`, `enrichment_health`, `enrichment_events`) so a
half-configured deploy still gives a useful glance.

See [Enrichment Layer Guide](./ENRICHMENT_LAYER_GUIDE.md) for the
operator runbook (auto-disabled recovery, cost cap behaviour, adding a
new enricher / profile).
