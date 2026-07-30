# ADR-142: LiteLLM gateway on the prod VPS (one gateway per failure domain)

- **Status**: Accepted
- **Date**: 2026-07-30
- **Authors**: Marko Dragoljevic, Claude (Opus 4.8)
- **Related**: [#1357](https://github.com/chipi/podcast_scraper/issues/1357) (epic),
  [#1356](https://github.com/chipi/podcast_scraper/issues/1356) (provider integration),
  [ADR-114](ADR-114-shared-multi-tenant-public-edge-caddy.md) (the box's edge ownership),
  [ADR-128](ADR-128-decouple-tailnet-acl-from-hetzner-tofu.md) (tailnet ACL / GitOps),
  agentic-ai-homelab `infra/litellm/` + its RFC-0001 / ADR-0008 (the gateway concept)

## Context

The homelab runs LiteLLM as the estate's LLM gateway (mini, `:4001`) — one
OpenAI-compatible endpoint, providers-as-config, per-consumer budgets, spend
observability. podcast_scraper's prod pipeline could simply consume it over the tailnet
(#1356). But that couples **prod inference to a residential ISP + a Mac mini**: a homelab
outage would stop prod from summarizing/transcribing, not just stop its telemetry.

## Decision

**Prod runs its own LiteLLM instance on the VPS** — *one gateway per failure domain*.
Zero-hop from the app (loopback), so inference survives any homelab/ISP outage. #1356's
consumption model is unchanged; only the base URL becomes the local instance
(`http://127.0.0.1:4001/v1`).

Concrete choices:

- **Deployment** mirrors homelab's `infra/litellm/` (litellm-database + postgres:16), as its
  own `-p litellm` compose project (isolated, like operator/player). **Config-as-code, shared
  lineage** so the two gateways don't drift — per-instance difference is env only. Image tag
  pinned after first boot.
- **Loopback + tailnet, never public.** API bound to `127.0.0.1:4001` (the app, always) and
  the box's tailnet IP `:4001` (admin UI from a laptop/phone, ACL `autogroup:admin →
  tag:prod:4001`); Postgres to `127.0.0.1:5433`. No Caddy vhost, no public bind. Loopback stays
  regardless, so the gateway never depends on tailscale being up.
- **Provider keys move INTO the gateway** (via the sops secrets flow) and only there — the
  app never holds a provider key again (#1357 §2, delivered with #1356). The VPS gets its
  **own** upstream OpenRouter key so provider-side billing separates prod from homelab spend.
- **One budgeted virtual key** for the app (`proj-podcast-prod`, hard `max_budget`); the hard
  budget wall is tested before go-live.
- **Inference local, telemetry home.** Losing telemetry in a homelab outage is acceptable;
  losing inference is not. Four planes ship to the homelab pane (own `litellm-vps` projects):
  **traces** (Langfuse) + **errors** (GlitchTip) via the gateway callbacks; **logs** (container
  stdout) via a `litellm.alloy` node-Alloy drop-in → VictoriaLogs; **metrics** (per-key spend)
  pushed daily-ish to VictoriaMetrics by a host systemd timer reading the gateway Postgres (the
  gateway's own Prometheus endpoint is enterprise-gated, so we push the metered truth — option
  **a**, mirrors the container-metrics collector). A homelab Grafana "Prod LLM Gateway"
  dashboard + a homepage card summarize spend.

**The container reaches the remote homelab pane by the `homelab` name over the tailnet.**
Docker's bridge can't resolve MagicDNS, so `deploy-litellm.sh` resolves homelab's tailnet IP
fresh and pins it in compose `extra_hosts` (the same pattern `deploy.sh` uses for the app's
OTEL). This is the one place a naïve copy of the homelab config — which uses
`host.docker.internal` because Langfuse is same-host there — would silently drop all telemetry.

The tailnet ACL already permits `tag:prod → homelab-host:4000,8090,8428`, so **no ACL change**
is needed. Postgres stays unexposed: spend leaves the box only as the daily VM push
(option **a**), avoiding a new `homelab-host → prod:<pg>` ACL line (option b).

## Consequences

**Positive:** prod inference is decoupled from the homelab failure domain; per-project spend
attribution + a hard budget wall; provider swaps become gateway-side config, not app code;
full observability (traces/errors/spend) with no ACL change; provider keys leave app config.

**Negative:** a second gateway to keep in lineage with homelab (mitigated: shared config,
env-only per-instance delta); one more prod service to run + patch. The spend path is a
custom pusher (the Prometheus endpoint is enterprise-gated) — a small script to own.

**Neutral:** the day-one aliases are a homelab-copied placeholder; the real prod alias set
lands with the #1356 provider integration.

## Alternatives considered

- **Consume the homelab gateway directly** (no prod instance). Rejected — couples prod
  inference to the home ISP + mini; a homelab outage stops prod pipelines.
- **Grafana reads prod's Postgres over the tailnet** (spend option b). Rejected for v1 —
  fewer moving parts but needs a new ACL line + a tailnet-exposed RO Postgres port; the push
  (option a) keeps prod's DB unexposed and needs no ACL change.
- **Expose the gateway/admin UI on the shared edge.** Rejected — the gateway is never a
  public surface; admin access is tailnet/tunnel-only.

## References

- `infra/litellm/` (compose · config · env · spend pusher · README) ·
  `infra/deploy/deploy-litellm.sh` · `infra/litellm/grafana/` (dashboard + homelab handover)
- agentic-ai-homelab `infra/litellm/` (reference deployment), `tailscale/policy.hujson`
  (`tag:prod → homelab-host` telemetry ports already open)
