# ADR-145 — One in-place re-roll for invalid structured LLM responses before provider fallover

- **Status:** Proposed
- **Date:** 2026-08-03
- **Authors:** Podcast Scraper Team
- **Amends:** [ADR-099](ADR-099-lancedb-first-single-index-search.md) is unrelated;
  this amends **ADR-100** (response-shape guardrails) — see below.
- **Related:** `providers/guardrails` (`check_chat_response`), `utils/provider_metrics`
  (`retry_with_metrics`, circuit breaker #697), `utils/retryable_errors` (`is_retryable_error`),
  `summarization/fallback.py` (`FallbackAwareSummarizationProvider`), `workflow/llm_call_fuse`.

## Context & problem

A v2.5 reprocess over a 9-feed mock corpus failed 1/9 episodes: the vLLM
(`NVFP4/Qwen3-30B-A3B`) returned **truncated/invalid structured-summary JSON**.
`triggered_guardrail=false`, `completion_tokens=262` — not a max-tokens cut, a one-off bad
response. Re-running the same episode produced **valid** JSON (vLLM is not bit-deterministic even
at `temperature=0`). So the failure is a **transient invalid response**, not systematic.

Two defects combined:

1. **Summary validates in the wrong layer.** GI and KG validate their structured response **at the
   call site** (`check_chat_response(..., expect_json=True)`). Summary's real check —
   `parse_summary_output` (schema, stricter than `json.loads`) — runs **one to two layers above the
   call**, in `workflow/metadata_generation.py`, *after* the call returned successfully. So the
   provider-fallover wrapper, the guardrail, and `retry_with_metrics` never see a bad summary; it
   surfaces as a bare `RuntimeError` at the episode level.

2. **Response-shape violations are non-retryable in place (ADR-100).** `is_retryable_error` returns
   `False` for `GuardrailViolation`, with the rationale *"retrying the exact same request yields the
   same bad content."* The p04 evidence disproves that premise for `temperature > 0` (and for a
   non-deterministic server even at 0): a re-roll is a **fresh sample** and transient truncations
   self-heal.

The interim fix (commit `57ee206a`) made the episode-level retry catch the summary `RuntimeError` —
wrong layer (re-runs transcribe/diarize/GI/KG to fix one LLM call) and summary-only. This ADR
replaces it.

## Decision

1. **Co-locate structured validation at the call** for summary, matching GI/KG. The summary
   provider validates its own response (schema) at the call site, not downstream.

2. **One bounded in-place re-roll before fallover.** When a structured response fails validation,
   re-issue the *same* request on the *same* endpoint **once** (`temperature > 0` / non-deterministic
   → a genuinely different sample). If it still fails, fall over to the next provider (the existing
   ADR-100 `FallbackAware` chain), then fail the episode. Cheap free re-roll first, fallover second,
   fail last.

3. **Content-retry lives at exactly one layer — the call.** Remove content-parse retry from the
   episode layer (revert `57ee206a`); episode-retry keeps only genuinely episode-scoped transients
   (download/transcode/connection).

## Consequences

**Positive:** transient bad responses recover for free on the same endpoint (the common case),
across all structured call types (summary/GI/KG/labeling/quotes/entailment) and all providers
(vLLM/Gemini/OpenAI/…) — it is a general capability, not a vLLM/summary patch. A persistent-bad
response still fallovers then fail-fasts, so quality is unchanged.

**Bounded / safe:** the in-place re-roll is capped at **1** — each attempt ticks the per-episode
LLM-call fuse (the ~3500-call incident), so the blast radius stays tiny. Invalid-response carries no
HTTP status, so it does **not** trip the #697 circuit breaker (a healthy endpoint returning bad
content must not be parked). Content-retry at one layer only avoids the multiplicative
`in_place × fallback_chain × episode_retry` waste.

**Negative / neutral:** re-roll costs one extra generation on the rare bad response. For a
near-deterministic call shape (bundled summary `temperature=0`) the re-roll helps less than a
fallover — acceptable because the re-roll is bounded to 1 and fallover still follows.

## Alternatives considered

- **Episode-level retry (the `57ee206a` band-aid):** wrong layer (whole-episode re-run for one bad
  call), summary-only, and string-matched (fragile). Rejected.
- **Pure fallover, no in-place re-roll (ADR-100 status quo):** wastes a provider swap on a transient
  the *same* endpoint would have recovered for free; and it never fired for summary because summary
  validated downstream. Rejected as the sole mechanism; retained as the second stage.
- **Unbounded in-place retry:** storms the call-budget fuse. Rejected — bound = 1.
