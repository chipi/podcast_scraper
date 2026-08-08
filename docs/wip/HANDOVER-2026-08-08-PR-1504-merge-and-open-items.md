# Handover 2026-08-08 — PR #1504 status

## TL;DR

PR #1504 lands the v2.5 provider + corpus-prep arc. Doc-number collision with `main` has
been RESOLVED (renumbered); rebase + open-items in progress.

- PR: [chipi/podcast_scraper#1504](https://github.com/chipi/podcast_scraper/pull/1504)
- Branch: `feat/naming-arc-and-corpus-prep`
- Closes: #1482 (transcript-prefix caching RFC), #1499 (ADR-147 providers), #1500 (v2.5
  finale winners), #1501 (ADR-148 re-roll), #1502 (ADR-149 reprocess), #1503 (hardening)
- Part of: #1356, #1357 (LiteLLM-gateway epic — prod-deploy scope remains)

## Doc-number collision — RESOLVED

`main` had reused ADR-144/145/146 + RFC-111 for different docs. This branch's docs were
renumbered to the next-free numbers (all confirmed free on main):

- ADR-144 → **ADR-147** — first-class vLLM/Qwen/LiteLLM provider
- ADR-145 → **ADR-148** — in-place re-roll for invalid structured responses
- ADR-146 → **ADR-149** — corpus reprocess methodology
- RFC-111 → **RFC-115** — transcript-prefix caching for LLM stages

~70 files updated (docs, code comments, config, the reroll test filename, mkdocs nav +
indexes); `make docs` green; 222 affected tests pass. (Issue #1482's title still says
"RFC-111" — the doc is now RFC-115; harmless, the issue number is what Closes uses.)

## What's in the PR (the arcs)

- **ADR-147** first-class vLLM / native-Qwen / LiteLLM providers (namespaced cost/telemetry,
  fail-closed served-model checks, airgapped DGX on real HF ids).
- **RFC-115** transcript-prefix caching across all 10 providers.
- **v2.5 finale** to cloud winners `cloud_openrouter` / `cloud_qwen`, promoted to governed
  registry presets (plus `no_diarization` StageOption, `provider_chunked_gated_v25` GI config).
- **ADR-148** in-place re-roll and **ADR-149** corpus reprocess methodology.
- **Hardening:** guardrail fence-strip, gateway real-cost telemetry, podcast_obs zero-config,
  and cleared pre-existing ci-fast debt (E501, 3 codespell false-positives, 2 stale tests).

## Finale ranking (results are git-ignored .test_outputs)

Quality (blind Opus-4.8 vs 2.4 baseline, avg sum/ins/top), cost $/9ep:

- deepseek-native **8.41** ($0.740) — quality winner, clean 9-0
- anthropic-haiku **8.29** ($1.346) — was n=8 (2/9 episodes hard-failed on the fence bug, now fixed; re-run pending)
- gemini **8.04** ($0.971); deepseek-or **8.04** ($0.68 telemetry / ~$0.20 real)
- qwen-native **7.85** ($0.560); qwen-or **7.70** ($0.53 telemetry / ~$0.05 real)

## Open items (in progress)

- **anthropic finale re-run** — validate the fence fix (expect 9/9) and get anthropic's true rank.
- **Precise OR $/arm** — token-attributed SpendLogs query, or accept telemetry (fixed for future runs).
- **Scoped-out, now in scope** — govern litellm/qwen speaker/cleaning wire models; thread
  `response` into the GI-evidence (non-bundled) cost path.
