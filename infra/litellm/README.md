# LiteLLM — the prod VPS LLM gateway

One OpenAI-compatible endpoint on the prod box (`http://127.0.0.1:4001/v1`) in front of
every LLM provider — **providers are config, consumers never change**. This is prod's
**own** instance (not the homelab one): *one gateway per failure domain* — prod inference
must not depend on a residential ISP + Mac mini (#1357 / ADR-142). Consumption guidance
follows #1356, with the base URL = this local instance.

Shared lineage with `agentic-ai-homelab/infra/litellm/` — keep the two configs from
drifting; the per-instance difference is **env only** (keys, callbacks, bind).

## Status: live (#1357 §1)

Deployed + operational on the prod box (`deploy-litellm.yml`): gateway healthy, the
`proj-podcast-prod` virtual key minted (budget-walled), and all telemetry flowing
(Langfuse / GlitchTip / VictoriaLogs / VictoriaMetrics spend). The aliases in `config.yaml`
are still a **placeholder** copied from homelab so the gateway boots + is testable; the real
prod alias set is decided in #1356 (provider integration) and will replace them.

## Layout

| File | What |
| --- | --- |
| `docker-compose.litellm.yml` | `litellm` (litellm-database) + `postgres:16`; own `-p litellm` project |
| `config.yaml` | model aliases (placeholder) + Langfuse/GlitchTip callbacks + master-key |
| `.env.example` | the secret slots (delivered from GH Actions secrets → staged `.env` at deploy, ADR-115 Option A; never commit `.env`) |
| `spend-to-vm.sh` | per-key spend → homelab VictoriaMetrics; run by the `litellm-spend-push` compose sidecar (rootless, no host systemd) |
| `../deploy/deploy-litellm.sh` | deploy: stage env, resolve homelab IP, compose up, health-gate |

## Two prod-specific differences from the homelab reference

1. **Loopback + tailnet, never public.** API on `127.0.0.1:4001` (the app, always) **and** on
   the box's tailnet IP `:4001` (admin UI from a laptop/phone — added by
   `docker-compose.litellm-tailnet.yml` when `deploy-litellm.sh` resolves the self IP; loopback
   stays so the gateway never depends on tailscale being up). Postgres on `127.0.0.1:5433`
   (spend pusher). No Caddy vhost, no public bind; the tailnet side is gated by the ACL
   (`autogroup:admin → tag:prod:4001`).
2. **Telemetry is remote.** Homelab runs Langfuse/GlitchTip; on prod the container reaches
   them by the `homelab` name over the tailnet. Docker's bridge can't resolve MagicDNS, so
   `deploy-litellm.sh` resolves homelab's tailnet IP fresh and pins it in compose
   `extra_hosts` (mirrors `deploy.sh`'s `HOMELAB_TAILNET_IP`). Use `homelab`, **not**
   `host.docker.internal` (that's the prod host — it runs neither service).

The tailnet ACL already permits `tag:prod → homelab-host:4000,8090,8428` (Langfuse /
GlitchTip / VictoriaMetrics) — no ACL change needed.

## Deploy (once the LITELLM_* GH secrets are set)

```sh
# on the box, from the checkout — or via deploy-litellm.yml
bash /srv/podcast-scraper/infra/deploy/deploy-litellm.sh
docker exec litellm sh -c 'curl -fsS http://127.0.0.1:4000/health/liveliness'   # -> I'm alive!
```

## The app's virtual key (per-project budget — the money backstop)

Minted with the master key; the app NEVER gets the master key:

```sh
curl -s http://127.0.0.1:4001/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' \
  -d '{"key_alias":"proj-podcast-prod","max_budget":25.0}'
```

`max_budget` is a hard cap; `/key/info` shows spend. Raising it is a deliberate, auditable
top-up.

## Budget-wall test (do this once — it's a success criterion)

1. Mint a scratch key with a tiny budget (e.g. `max_budget: 0.01`).
2. Make a call with it → expect a refusal once the cap is hit.
3. Delete the scratch key; mint the real `proj-podcast-prod` with the real budget.

## Observability — four planes, each to the right tool

- **Traces** (per-request: prompt/completion/tokens/cost/latency) → **Langfuse**, project
  `litellm-vps` (`success/failure_callback: langfuse`). This is the LLM-call detail.
- **Errors** (gateway exceptions) → **GlitchTip**, project `litellm-vps` (`failure_callback:
  sentry`). Both `litellm-vps` projects are kept separate from homelab's `litellm-gateway`.
- **Metrics** (per-key spend/budget/burn) → **VictoriaMetrics**: the `litellm-spend-push`
  compose sidecar runs `spend-to-vm.sh` every 30 min, reading the gateway Postgres and pushing
  `litellm_key_spend_usd` / `_max_budget_usd` / `_budget_burn_ratio` (`box="prod"`) to
  `homelab:8428`. Viewed in **Grafana** ("Prod LLM Gateway", see `grafana/`) + the homepage.
- **Logs** (gateway container stdout — startup, config reloads, provider failures, budget
  refusals, Postgres) → **VictoriaLogs**: `litellm.alloy` (a node-Alloy drop-in, installed by
  `deploy-litellm.sh` into `config.d/`) scrapes the `litellm` + `litellm-postgres` containers
  (`app=litellm`), drops health-poll noise, extracts `trace_id`. Langfuse carries the *calls*;
  this carries everything else.

## Rotating a key

- **Provider key** (e.g. OpenRouter): update the `LITELLM_OPENROUTER_API_KEY` GH secret +
  re-run `deploy-litellm.yml` (or `deploy-litellm.sh` on the box). The app never holds a
  provider key.
- **Virtual key**: regenerate via `/key/generate`, update the app's secret, delete the old.
