# Handover → homelab agent: DGX vLLM is serving under the placeholder key `EMPTY`

Date: 2026-08-30. Raised from the podcast_scraper side during pre-deploy verification.

## What we found

The DGX vLLM on `dgx-llm-1:8003` currently accepts the bearer `EMPTY` and rejects
everything else:

```
curl -s http://dgx-llm-1:8003/v1/models                                  -> {"error":"Unauthorized"}
curl -s -H "Authorization: Bearer <the-intended-key>" .../v1/models      -> {"error":"Unauthorized"}
curl -s -H "Authorization: Bearer EMPTY"             .../v1/models       -> 200, NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4
```

`EMPTY` is not a chosen secret. It is the hardcoded placeholder our client sends when no
key is configured — `_VLLM_DUMMY_BEARER = "EMPTY"` in
`src/podcast_scraper/providers/vllm/vllm_provider.py:30`. Production stages no
`VLLM_API_KEY` at all (it exists only in the `baselines` GitHub environment, for the
autoresearch eval judges). So prod authenticates by coincidence: both sides independently
defaulted to the same placeholder. The endpoint is effectively unauthenticated, protected
only by the tailnet.

## The ask

Put the intended key (the operator has it; not written down here) back as the vLLM API key on the DGX, as part of the proper
secret-management solution you are building.

## Ordering constraint — please read before flipping

Flipping the DGX key in isolation **breaks every production DGX run**, silently. Prod
would keep sending `EMPTY`, get 401 on every LLM stage, and fall through the RFC-106
fallback ladder to `ollama` / `llama3.1:8b` — a large quality regression that shows up as
successful runs, not as errors. This is the silent-degradation class we spent 2026-08-29/30
eliminating.

Safe sequence:

1. **(done, our side)** `VLLM_API_KEY: ${VLLM_API_KEY:-}` is now staged in
   `compose/docker-compose.prod.yml`. It is inert: unset resolves to empty, the client
   falls back to the `EMPTY` placeholder, and today's behaviour is unchanged. Verified
   with `docker compose config` → `VLLM_API_KEY: ""`.
2. Operator sets the `production` environment secret `VLLM_API_KEY` to that same value.
3. Deploy prod (this is in the pending daylight batch anyway).
4. **Then** flip the DGX-side key.

Between 3 and 4 there is a window where prod sends the real key and the DGX still expects
`EMPTY` → 401. Keep it short. The nightly scheduler is currently **disabled**
(`enabled: false`) and no runs are queued, so the window is controllable — but confirm
nothing is in flight before flipping.

## Note for whoever builds the permanent solution

The intended key is a shared static string that would live in two places (DGX container
env, prod GitHub secret). The operator has said explicitly they are not treating this as a
high-secrecy value while the DGX is tailnet-only. Recording that as a deliberate choice
rather than an oversight, so the proper solution can decide whether to keep it.

Related: #1876 (DGX pilot), #1888 (pipelining), and the pending daylight fix batch.
