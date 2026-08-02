# Homelab handover — Prod LLM Gateway dashboard

Grafana is homelab-owned (`agentic-ai-homelab`), so this side ships the **dashboard JSON +
the metric contract**; the homelab agent wires it into Grafana. Nothing here touches your
Grafana directly.

## The metric contract (already flowing once prod deploys)

The prod box pushes these to VictoriaMetrics every 30 min (the `litellm-spend-push` compose
sidecar — a postgres-client container in the `-p litellm` project reading the gateway Postgres
over the compose network; the gateway's Prometheus endpoint is enterprise-gated, so we push
metered truth). All carry `box="prod"`:

- `litellm_key_spend_usd{box,key_alias}` — lifetime spend on each virtual key
- `litellm_key_max_budget_usd{box,key_alias}` — the hard budget wall (0 = unset)
- `litellm_key_budget_burn_ratio{box,key_alias}` — spend / budget (0 when no budget)

Push target `homelab:8428` — ACL `tag:prod → homelab-host:8428` already open, no change.

## To wire it (pick one)

1. **Import as its own board:** import `prod-llm-gateway.dashboard.json` against the
   VictoriaMetrics/Prometheus datasource. It has: total spend (stat), budget burn per key
   (bargauge, amber ≥70% / red ≥90%), spend-per-key over time (timeseries), and a per-key
   table.
2. **Fold into the existing "LLM Gateway — LiteLLM" board** (issue §5's preference — one
   pane for both gateways): copy the four panels in and add a `box=~"prod|homelab"` split, or
   keep prod as its own row. The queries above are datasource-agnostic PromQL.

## Not included yet (needs a decision your side or a later push)

- **Tokens / requests / per-stage tags** — the v1 pusher only carries spend + budget. If you
  want tokens/requests on this pane, either (a) extend `spend-to-vm.sh` to also read
  `LiteLLM_SpendLogs` / `LiteLLM_DailyTagSpend`, or (b) point a Langfuse panel at the
  `litellm-vps` project. Say which and prod side will add the push.
- **Langfuse traces / GlitchTip errors** for the `litellm-vps` projects — those land in your
  existing Langfuse/GlitchTip; just confirm the projects exist so the gateway callbacks have
  somewhere to write.

## What prod needs FROM homelab to finish the loop

- A **Langfuse project** `litellm-vps` (public + secret keys) and a **GlitchTip project** for
  `litellm-vps` (DSN) — these go into the prod gateway's sops secrets. Kept separate from
  homelab's own `litellm-gateway` project so the two gateways' telemetry doesn't merge.
