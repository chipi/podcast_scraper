# ADR-148 — One in-place re-roll for invalid structured LLM responses before provider fallover

- **Status:** Accepted
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

## Shared seam

`providers/guardrails/reroll.py::structured_call_with_reroll(make_call, validate, *, service,
max_reroll=1)` is the one reusable mechanism: call → validate → on any validator exception re-roll
up to `max_reroll` on the same endpoint → then raise `GuardrailViolation` so the ADR-100
`FallbackAware` chain engages. `validate` raising anything (a `GuardrailViolation` from
`check_chat_response`, a `json`/schema error) counts as "invalid response, re-roll". It is
provider- and call-type-agnostic.

## What is wired (this PR) — and what is NOT

Each structured call already has a recovery posture; the re-roll is added where a transient bad
response would otherwise fail-up with no cheaper recovery:

| Call | Validator | Recovery before this PR | Now |
| --- | --- | --- | --- |
| **Summary** (staged/mega/bundled) | `parse_summary_output` (schema) — runs in `workflow/metadata_generation` | none in place; validated a layer above the call → episode fail (the p04 bug) | one in-place re-roll at the orchestrator (re-issues the same summary call), then the existing hard fail |
| **`complete_text`** (ADR-110 resolver) | `check_chat_response(expect_json)` | raise `GuardrailViolation` → fallover | seam re-roll → fallover |
| **`classify_insights`** (GI value-gate) | `check_chat_response(expect_json)` + `json.loads` | raise → gate fails open | seam re-roll → then raise (gate fails open) |
| **`generate_insights`** (GI line-list) | `check_chat_response(finish_reason)` | truncated-line **salvage**, else fallover | unchanged — salvage already recovers the common truncation; re-roll not added |
| **`extract_kg_graph`** (KG) | `parse_kg_graph_response` | degrades to `None` (no episode fail) | unchanged — degrade-to-None is already safe |

The summary re-roll is at the **orchestrator** (not inside the provider) because summary's real
validator is `parse_summary_output`, which lives a layer above the provider call; re-issuing there
keeps content-retry at exactly one layer for summary. The JSON provider calls use the seam directly
at the call site. GI line-list and KG keep their existing (different, appropriate) recovery, so this
is not deferred wiring — it is per-shape recovery, documented here so the boundary is explicit.

## Consequences

**Positive:** transient bad responses recover for free on the same endpoint (the common case). The
seam is a general capability, not a vLLM/summary patch — it is provider-agnostic and wired today at
summary (orchestrator) + `complete_text` + `classify_insights` (see the wiring table for the exact
set, and for the calls that keep their own per-shape recovery). A persistent-bad response still
fallovers then fail-fasts, so quality is unchanged.

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
