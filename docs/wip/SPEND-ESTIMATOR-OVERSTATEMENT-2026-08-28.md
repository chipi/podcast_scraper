# Run-reported LLM cost overstates the gateway's actual spend ~3.5×

**Status:** root-caused, NOT fixed. Measured 2026-08-28 during the nightly-scheduler
acceptance. No money is lost — this is a *reporting* defect — but the run-cost cap and the
`$12/day` alert both fire on the inflated number, so runs halt earlier than the budget
intends.

## The measurement

One sweep window (`b5f885ca`, 2026-08-28 16:57–19:07 UTC), 14 feeds, all LLM traffic through
the prod LiteLLM gateway (`deepseek-v4-flash`):

| Source | LLM cost | How obtained |
| --- | --- | --- |
| Run-reported (`Run cost — LLM:` lines, summed) | **$2.2685** | pipeline's own estimate |
| Gateway SpendLogs (billing truth) | **$0.6518** over 824 calls | `litellm-postgres` → `LiteLLM_SpendLogs` |

Ratio **3.48×**. Cache-hit ratio in the same window was `0.00`, so caching is not the
explanation.

## Cause

`_openai_response_cost_usd` (`providers/openai/openai_provider.py`) prefers the upstream's
real cost and falls back to the local pricing table only when it finds none. It looks in two
places — `response.usage.cost` (OpenRouter style) and `_hidden_params["response_cost"]`
(LiteLLM's own SDK).

Probed the prod gateway directly from `compose-api-1`:

```text
usage: None                       # the gateway strips usage from the body
x-litellm-response-cost: 0        # the real cost is a RESPONSE HEADER
```

The gateway returns cost in an HTTP header, and the plain `openai` SDK client the provider
uses never surfaces headers on the parsed response object. So the upstream-cost lookup always
returns `None`, every call falls back to `pricing_assumptions`, and that table prices
`deepseek-v4-flash` at direct-DeepSeek list rates — not the gateway's negotiated/routed rate.
Hence a consistent multiple rather than noise.

## Fix options (not yet chosen)

1. **Read the header** — use `client.chat.completions.with_raw_response.create(...)`, take
   `x-litellm-response-cost`, fall back to the table when absent. Most faithful: it is the
   gateway's own number, per call, no table maintenance. Cost: touches the shared
   OpenAI-compatible transport, so it needs care across all providers built on it.
2. **Reconcile after the run** — leave per-call estimates alone and correct the run total from
   SpendLogs at run end. Accurate for reporting; does NOT fix the in-run cap, which is the
   part that actually changes behaviour.
3. **Recalibrate the table** — cheapest, but a static multiplier rots the moment routing or
   rates change, and it hides the real defect (upstream cost is available and ignored).

Recommendation: **option 1**, with option 2's reconciliation as a cross-check in the run
summary. Do not do option 3 alone.

## Why it matters operationally

The `$10` soft cap halted two of the three catch-up sweeps on 2026-08-28. With true costs
those runs had spent well under a third of the budget — i.e. the cap is currently ~3.5× more
conservative than intended, which turns one nightly into several manual re-fires.
