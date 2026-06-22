# Prod observability control plane (`podcast_obs`, #803)

A small, **standalone** control plane that answers *"what is a deploy doing right now?"* —
health, version, recent pipeline runs and deploys, today's LLM cost, recent **error logs** (Loki)
and Sentry issues, and current Grafana alerts — for **any** deploy (your local stack, prod over
Tailscale, a drill). It's **target-agnostic** and **degrades gracefully**: a source that isn't
wired for a target just reports `configured=false`.

It lives in its own light package (`podcast_obs`) with **zero coupling** to the heavy
`podcast_scraper` pipeline, so it runs cheaply as a plain process or a small container anywhere on
the tailnet.

## Layers

| Layer | What | Entry point |
| --- | --- | --- |
| **Core** | probe/aggregate functions, one per backing system | `podcast_obs.sources.*` |
| **CLI ("the basics")** | probe any deploy directly, scriptable, no MCP | `python -m podcast_obs <cmd>` |
| **MCP server** | the same probes as agent tools | `python -m podcast_obs serve` |
| **Docker** | the standalone container control plane | `docker/observability/` |

## Install

`httpx` + `PyYAML` are base deps; the agent-facing layer adds `mcp`:

```bash
pip install -e '.[observability]'   # in this repo
```

For a slim standalone deploy, the container installs only `httpx`/`PyYAML`/`mcp` (see Docker below).

## Configure

Target-agnostic. Either a single target from env vars, or multiple targets from YAML.

**Env (single target):** prefix `PODCAST_OBS_`.

| Var | Used by | Notes |
| --- | --- | --- |
| `PODCAST_OBS_API_BASE` | health/version/runs | e.g. `http://localhost:8080` or `https://prod-podcast.<tailnet>` |
| `PODCAST_OBS_GITHUB_REPO` / `_GITHUB_TOKEN` | deploys | default repo `chipi/podcast_scraper` |
| `PODCAST_OBS_SENTRY_ORG` / `_SENTRY_PROJECTS` / `_SENTRY_TOKEN` / `_SENTRY_ENV` | errors | projects CSV; env default `prod` |
| `PODCAST_OBS_GRAFANA_URL` / `_GRAFANA_TOKEN` | alerts (+ Loki token) | Grafana stack URL |
| `PODCAST_OBS_LOKI_URL` / `_LOKI_USER` | cost/logs | Loki push or query URL (suffix is normalised) |
| `PODCAST_OBS_ENV_LABEL` | cost/logs | the deploy's Loki `env` label (default `prod`) |
| `PODCAST_OBS_TIMEOUT` | all | per-request seconds (default 10) |

**YAML (multi-target):** point `PODCAST_OBS_CONFIG` at a file. Secrets stay out of the file via
`<field>_env:` indirection (the value is read from the named env var). See
[`config/observability.example.yaml`](https://github.com/chipi/podcast_scraper/blob/main/config/observability.example.yaml):

```yaml
default_target: local
targets:
  local:
    api_base: http://localhost:8080
  prod:
    api_base: https://prod-podcast.tail-xxxxx.ts.net
    env_label: prod
    github:
      repo: chipi/podcast_scraper
      token_env: PODCAST_OBS_GH_TOKEN
    sentry:
      org: your-org
      projects: [api, pipeline, viewer]   # real slugs (Settings → Projects), not DSN names
      token_env: PODCAST_OBS_SENTRY_TOKEN
    grafana:
      url: https://your-stack.grafana.net
      token_env: PODCAST_OBS_GRAFANA_TOKEN      # service-account token (glsa_) for alerting
      loki_url: https://logs-prod-xxx.grafana.net
      loki_user: "123456"
      loki_token_env: PODCAST_OBS_LOKI_TOKEN    # access-policy token (glc_), logs:read
```

### Tokens and scopes (read-only — the control plane never mutates)

| Source | Credential | Scope / gotcha |
| --- | --- | --- |
| `prod_api` | none | Reachability only (tailnet-gated in prod). |
| `github` | fine-grained PAT, or the `gh` CLI | **Actions: read**. |
| `sentry` | a Sentry **auth token** | **Issue & Event: Read** (`event:read`) — `project:read` alone is not enough. **NOT the DSN** (the staged `PROD_SENTRY_DSN_*` can't query the API). `prod_recent_errors`/D2 also want **Release: Admin**. Note: project **slugs** ≠ DSN names (check Settings → Projects; e.g. `podcast-scraper-api`). |
| `grafana` (alerts) | a Grafana **service-account** token (`glsa_`) | alerting read. Grafana-API only. |
| `loki` (cost/logs) | `loki_user` + a Cloud **access-policy** token (`glc_`) | **`logs:read`**. A *different token type* from the alerting one — Grafana Cloud splits the data plane (Loki, `glc_`) from the Grafana API (`glsa_`). The agent's `GRAFANA_CLOUD_API_KEY` is `logs:write` and 401s. (Falls back to the grafana token for self-hosted setups where one token serves both.) |
| `langfuse` (traces) | a Langfuse **public + secret key** pair (Basic auth) | Read-only public API. **SDK-native bare env names** `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` (not `PODCAST_OBS_*`) — the *same* pair the pipeline traces with, so one set drives both emit + probe. `LANGFUSE_BASE_URL` optional (unset = Cloud). |

## CLI (the basics)

```bash
python -m podcast_obs summary    --target prod      # control-plane glance, all sources
python -m podcast_obs health     --target local
python -m podcast_obs version    --target prod
python -m podcast_obs runs        --limit 5         # recent pipeline runs (/api/jobs)
python -m podcast_obs deploys     --limit 5         # deploy-prod.yml runs + failure rate
python -m podcast_obs cost-today                     # 24h LLM spend (Loki)
python -m podcast_obs logs --service pipeline --window 6h --contains OOM
python -m podcast_obs errors --window 24h            # Sentry issues
python -m podcast_obs alerts                          # Grafana alerts
python -m podcast_obs traces --limit 10              # recent Langfuse LLM traces
```

Every command prints a uniform JSON envelope — `{ok, source, data | error, configured}`. Exit code
is `0` on success, `1` when the probe failed (unreachable / not configured), `2` on a config error.
`summary` buckets sources into **live / unconfigured / failed**, so a local-only target still gives
a useful glance (externals report `unconfigured`).

The **`logs`** command is the signal Sentry misses: it reads raw container logs from Loki
(error-ish by default), including stderr tracebacks from pipeline subprocesses and `ERROR`/`WARNING`
lines the Sentry SDK never wrapped.

## MCP server (agent-facing)

Same probes, exposed as 10 tools (`prod_health`, `prod_version`, `prod_recent_runs`,
`prod_recent_deploys`, `prod_cost_today`, `prod_recent_logs`, `prod_recent_errors`,
`prod_recent_alerts`, `prod_recent_traces`, `prod_summary`) — each takes an optional `target`.

```bash
python -m podcast_obs serve --transport stdio                       # local agent
python -m podcast_obs serve --transport http --host 0.0.0.0 --port 8848   # networked control plane
```

Use **stdio** for a co-located agent; **sse / http** (streamable-http, default path `/mcp`) for a
container other tailnet boxes can reach.

## Docker (standalone control plane)

A light image (`python:3.12-slim` + `httpx`/`PyYAML`/`mcp` + the package — no pipeline deps) you run
on your MBP, an Orb, or a Mac mini in the tailnet.

```bash
docker build -f docker/observability/Dockerfile -t podcast-obs:latest .

# one-shot probe
docker run --rm -e PODCAST_OBS_API_BASE=https://prod-podcast.<tailnet> podcast-obs summary

# the agent-facing control plane (MCP over http)
docker run --rm -p 8848:8848 \
  -v "$PWD/config/observability.yaml:/config/observability.yaml:ro" \
  -e PODCAST_OBS_CONFIG=/config/observability.yaml podcast-obs
```

See [`docker/observability/docker-compose.example.yml`](https://github.com/chipi/podcast_scraper/blob/main/docker/observability/docker-compose.example.yml).
To reach prod over Tailscale **and** be reachable by remote agents, run it on a tailnet host (host
networking) or add a tailscale sidecar.

## Task an agent to use it (with Grafana MCP and others)

Register `podcast_obs` as an MCP server in your agent. Claude Code, local stdio:

```json
{
  "mcpServers": {
    "podcast-obs": {
      "command": "python",
      "args": ["-m", "podcast_obs", "serve", "--transport", "stdio"],
      "env": { "PODCAST_OBS_CONFIG": "/path/to/observability.yaml" }
    }
  }
}
```

Or a remote container over the tailnet:

```json
{
  "mcpServers": {
    "podcast-obs": { "url": "http://mac-mini.tail-xxxxx.ts.net:8848/mcp" }
  }
}
```

Add the **Grafana MCP** server alongside it the same way. Division of labour:

- **`podcast_obs`** — the fast cross-source glance: health, version, deploys, cost-today, **raw
  error logs**, Sentry issues, alerts. One call (`prod_summary`) tells you *what's off*.
- **Grafana MCP** — deep drill-down: render a dashboard panel, query a specific metric over a range,
  inspect an alert rule.

Example you can hand an agent:

> "Use `podcast-obs` `prod_summary` for prod. If cost-today is unusually high or there are recent
> error logs, pull the relevant Grafana dashboard via the Grafana MCP and summarise the spike; if a
> deploy failed, show its recent `prod_recent_logs` for the `pipeline` service."

The agent gets a cheap, structured first look from `podcast_obs` and escalates to Grafana only when
something needs investigation — no operator `ssh` or dashboard-clicking.

## Langfuse LLM tracing (#1052)

Langfuse is the **AI-quality lens** the cost/ops sources don't give: where `cost-today` answers
*"how much did we spend"*, Langfuse answers *"what did each LLM call do"* — a `generation` span per
call (model / token usage in·out·total / cost / stage), grouped per run. (Per-call **latency** is a
phase-2 item — the span is emitted at the post-call cost choke point, which carries no timing yet.)
It **coexists** with the own
solution (Loki `llm_cost` + `corpus_manifest.cost_rollup` + Sentry stay the source of truth for
cost/ops); Langfuse is additive, not a replacement.

**Two surfaces, one key pair** (`LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY`; `LANGFUSE_BASE_URL`
optional, unset = Cloud — these are the Langfuse SDK's own env names):

1. **Pipeline emits** — a hook at the provider cost choke point
   (`record_provider_call_cost`) emits one generation span per LLM call across all 8 providers.
   Enable-when-secret-present (Sentry pattern): a **true no-op** unless both keys are set, so dev /
   CI / offline runs stay silent. The SDK ships in `[dev]`; a runtime-only install adds the
   `[langfuse]` extra (`pip install -e '.[langfuse]'`) — and the prod pipeline image bakes it.
   Langfuse is an optional o11y extension — see
   [OBSERVABILITY_EXTENSIONS.md](OBSERVABILITY_EXTENSIONS.md).
2. **Control plane reads** — the `traces` probe / `prod_recent_traces` MCP tool / Ops-view card
   query the same account back (Basic auth, `httpx` only — no SDK in the light control plane).

Hosting is **decided at enable-time**: point `LANGFUSE_BASE_URL` at Langfuse Cloud or a self-hosted
instance. Spans never block a run — every tracing entrypoint is wrapped, so a tracing failure is at
most a missing span plus a debug log.

## Validation

- Unit tests: `tests/unit/podcast_obs/` (config, every source, summary, MCP wiring).
- Live E2E: `tests/e2e_observability/` — asserts **shape + invariants** (not values, since live data
  changes) and **self-skips** when a source is unconfigured/unreachable. Run with real network:

  ```bash
  .venv/bin/python -m pytest tests/e2e_observability/ -q --no-cov -p no:cacheprovider
  ```

  GitHub runs live via your `gh` token; the rest light up as you provide their read-scoped tokens.
