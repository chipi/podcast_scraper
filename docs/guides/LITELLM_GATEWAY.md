# LiteLLM gateway — how to think about it + how to integrate (guide for #1356)

This is the orientation guide for wiring podcast_scraper's LLM calls through the **prod
LiteLLM gateway** (#1356 integration). Read it before touching provider code. The gateway
itself is built + its telemetry projects exist (#1357 / ADR-142); this guide is about
*consuming* it.

## The one idea: aliases are the contract, providers are config

The app never names a provider again. It asks the gateway for an **alias** (e.g.
`podcast-pro`) with **one virtual key**, over an OpenAI-compatible endpoint. Which vendor
serves that alias — OpenRouter, direct DeepSeek, Gemini, whatever — is decided *gateway-side*
in `config.yaml`, swappable with no app deploy. So: eval stamps, rate tables, and stage
configs survive a provider swap, and a bad provider is a one-line gateway change, not a
code change.

Three properties you get for free by routing through it:

1. **Per-project spend attribution + a hard budget wall.** One virtual key
   (`proj-podcast-prod`) with a `max_budget`. The gateway meters every call; when the budget
   is hit, calls are refused. (Gateway metering has already caught self-reporting bugs ~4×
   estate-wide — trust the gateway's number over the app's.)
2. **Provider swaps become config.** Change the route behind an alias, not the app.
3. **Free observability** — traces, errors, spend (see "Where the data goes").

## Architecture (why prod has its OWN gateway)

**One gateway per failure domain** (ADR-142). Prod runs its *own* LiteLLM instance on the
VPS, loopback-bound — so prod inference does NOT depend on the home ISP + Mac mini. A homelab
outage costs telemetry, never inference.

```text
podcast prod app ──(OpenAI SDK, localhost)──▶ prod LiteLLM gateway ──▶ providers (OpenRouter, …)
                                                     │
                                       (telemetry, over the tailnet, best-effort)
                                                     ▼
                              homelab: Langfuse (traces) · GlitchTip (errors) · VictoriaMetrics (spend)
```

## How to wire the app (the #1356 work)

Any OpenAI-compatible client — base URL + the virtual key + an alias:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:4001/v1", api_key=PROJ_PODCAST_PROD_KEY)
resp = client.chat.completions.create(model="podcast-pro", messages=[...])
```

- **Base URL** = the local gateway, `http://127.0.0.1:4001/v1` (loopback — the app and the
  gateway are on the same box). NOT `homelab:4001` (that's the homelab gateway; prod uses its
  own for failure isolation).
- **Key** = the single `proj-podcast-prod` virtual key, delivered as a prod secret (sops / GH
  secret → gateway-minted with the master key). The app holds THIS, never a provider key.
- **Model** = an alias. The day-one aliases are placeholders copied from homelab; **the real
  prod alias set is yours to define in #1356** once you know which stage calls what. Add an
  alias = one block in `infra/litellm/config.yaml` + restart the gateway.

**Per-stage cost breakdown without more keys** — pass request tags:

```python
resp = client.chat.completions.create(
    model="podcast-pro", messages=[...],
    extra_body={"metadata": {"tags": ["transcribe-cleanup"]}},
)
```

Tags aggregate into the gateway DB (`LiteLLM_DailyTagSpend`) and are queryable per stage.

## Migrating existing provider calls (the actual integration)

The pipeline currently calls providers directly (OpenRouter, Gemini, DeepSeek, …). The move:

1. **Keys move INTO the gateway, out of the app.** Every provider key leaves app
   config/secrets and lands in the gateway's env (sops). Verify with a secrets-scan in CI
   that no provider key remains in app config — that's a success criterion (#1357 §6.1).
2. **Point each call site at the gateway** — base URL + `proj-podcast-prod` key + an alias.
   Decide the alias set first (map each pipeline stage → an alias).
3. **Gemini decision:** LiteLLM speaks Gemini natively, so the cloud-fallback path *can*
   route through the gateway too — but that means the Gemini key goes into the gateway. Decide
   whether the proxy indirection is worth it for the fallback path, or scope the first pass to
   the OpenRouter/Chinese-provider calls and keep direct-Gemini as a documented emergency path.
4. **Prove the swap once:** flip an alias's route gateway-side and confirm the app picks it up
   with no deploy. That's the whole point — demonstrate it works.

## Where the data goes (and how to read it)

All tailnet-only — reach from any Tailscale device (laptop / phone), MagicDNS `homelab`:

- **Traces** (prompt/completion/tokens/cost/latency, per request): **Langfuse**
  `http://homelab:4000`, project **`litellm-vps`**.
- **Errors** (gateway exceptions): **GlitchTip** `http://homelab:8090`, project **`litellm-vps`**.
- **Spend** (per-key $ / budget / burn): pushed to **VictoriaMetrics** as
  `litellm_key_spend_usd{box="prod",key_alias="proj-podcast-prod"}` (+ `_max_budget_usd`,
  `_budget_burn_ratio`) every ~30 min by the `litellm-spend-push` systemd timer. Viewed in
  **Grafana** (`http://homelab:3000`, "Prod LLM Gateway" board) and summarized on the homelab
  homepage (`http://homelab:8888`, the Prod LLM card).

The gateway's own admin UI (keys, budgets, request logs) is on the prod tailnet IP:
`http://<prod-tailnet-ip>:4001/ui` from any Tailscale device (laptop/phone), auth = the master
key (ACL `autogroup:admin → tag:prod:4001`). It's also linked with creds from the homelab
homepage's prod section.

## Operating the gateway (runbook)

- **Restart** (config change, provider swap): `docker compose -p litellm up -d litellm` on the
  box (or re-run `infra/deploy/deploy-litellm.sh`). Boring by design; the app should
  retry/backoff over the blip.
- **Mint the app key** (once): `curl -s http://127.0.0.1:4001/key/generate -H "Authorization:
  Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' -d
  '{"key_alias":"proj-podcast-prod","max_budget":25.0}'`. The master key never leaves the box.
- **Budget wall test** (do once): mint a scratch key with `max_budget: 0.01`, make a call,
  watch it refuse, delete the scratch key.
- **Rotate a provider key:** one sops edit + one gateway restart. The app is untouched.
- **Rotate the virtual key:** regenerate via `/key/generate`, update the app's secret, delete
  the old key.
- **Reconcile monthly:** gateway-metered spend for the VPS's own OpenRouter key vs OpenRouter's
  billing — the check that catches self-reporting drift.

## References

- `infra/litellm/` (compose · config · spend pusher · README) · `infra/deploy/deploy-litellm.sh`
- [ADR-142](../adr/ADR-142-litellm-prod-gateway.md) · issues #1357 (deploy) / #1356 (this integration)
- Homelab reference: `agentic-ai-homelab/infra/litellm/` · RFC-0001 / ADR-0008 (the gateway concept)
