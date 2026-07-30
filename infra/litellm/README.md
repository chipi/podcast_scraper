# LiteLLM — the prod VPS LLM gateway

One OpenAI-compatible endpoint on the prod box (`http://127.0.0.1:4001/v1`) in front of
every LLM provider — **providers are config, consumers never change**. This is prod's
**own** instance (not the homelab one): *one gateway per failure domain* — prod inference
must not depend on a residential ISP + Mac mini (#1357 / ADR-142). Consumption guidance
follows #1356, with the base URL = this local instance.

Shared lineage with `agentic-ai-homelab/infra/litellm/` — keep the two configs from
drifting; the per-instance difference is **env only** (keys, callbacks, bind).

## Status: scaffolding (#1357 §1)

Built, not yet deployed. The aliases in `config.yaml` are a **placeholder** copied from
homelab so the gateway boots + is testable; the real prod alias set is decided in #1356
(provider integration) and will replace them.

## Layout

| File | What |
| --- | --- |
| `docker-compose.litellm.yml` | `litellm` (litellm-database) + `postgres:16`; own `-p litellm` project |
| `config.yaml` | model aliases (placeholder) + Langfuse/GlitchTip callbacks + master-key |
| `.env.example` | the secret slots (sops-delivered on prod; never commit `.env`) |
| `spend-to-vm.sh` + `litellm-spend-push.{service,timer}` | daily-ish per-key spend → homelab VictoriaMetrics |
| `../deploy/deploy-litellm.sh` | deploy: stage env, resolve homelab IP, compose up, health-gate |

## Two prod-specific differences from the homelab reference

1. **Loopback-bound, never public.** API on `127.0.0.1:4001`, Postgres on `127.0.0.1:5433`
   (for the spend pusher). No Caddy vhost. Admin UI via an SSH/tailscale tunnel; a
   tailnet-only admin bind is a follow-up, not day one.
2. **Telemetry is remote.** Homelab runs Langfuse/GlitchTip; on prod the container reaches
   them by the `homelab` name over the tailnet. Docker's bridge can't resolve MagicDNS, so
   `deploy-litellm.sh` resolves homelab's tailnet IP fresh and pins it in compose
   `extra_hosts` (mirrors `deploy.sh`'s `HOMELAB_TAILNET_IP`). Use `homelab`, **not**
   `host.docker.internal` (that's the prod host — it runs neither service).

The tailnet ACL already permits `tag:prod → homelab-host:4000,8090,8428` (Langfuse /
GlitchTip / VictoriaMetrics) — no ACL change needed.

## Deploy (once secrets are in via sops)

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

## Spend → homelab observability

- **VictoriaMetrics:** `litellm-spend-push.timer` runs `spend-to-vm.sh` every ~30 min,
  reading per-key spend from the gateway Postgres and pushing `litellm_key_spend_usd`,
  `litellm_key_max_budget_usd`, `litellm_key_budget_burn_ratio` (`box="prod"`) to
  `homelab:8428`. Installed on the box by cloud-init (durable across rebuilds).
- **Grafana:** a "Prod LLM Gateway" dashboard (spend/tokens/requests/budget-burn) lives on
  the homelab Grafana — see `grafana/` for the dashboard JSON + the homelab-side handover.
- **Langfuse / GlitchTip:** per-request traces + gateway errors ship to the `litellm-vps`
  projects on homelab (own projects, kept separate from homelab's `litellm-gateway`).

## Rotating a key

- **Provider key** (e.g. OpenRouter): one sops edit + `deploy-litellm.sh` (or
  `docker compose -p litellm up -d litellm`). The app never holds a provider key.
- **Virtual key**: regenerate via `/key/generate`, update the app's secret, delete the old.
