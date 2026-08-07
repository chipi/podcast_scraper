# Handover 2026-08-08 — PR #1504 merge + open items

## TL;DR

**PR #1504 is OPEN, ready, and green** (Snyk pass; `make ci-fast` green locally — this
repo has no GitHub-Actions test-CI on PRs, validation is local ci-fast). The **only**
thing between it and merge is a **doc-number collision with `main`** (below). Nothing
else blocks it.

- PR: https://github.com/chipi/podcast_scraper/pull/1504
- Branch: `feat/naming-arc-and-corpus-prep` (84 commits ahead of main, 16 behind)
- Closes: #1482 (RFC-111) · #1499 (ADR-144 providers) · #1500 (v2.5 finale winners) ·
  #1501 (ADR-145 re-roll) · #1502 (ADR-146 reprocess) · #1503 (hardening)
- Part of: #1356 / #1357 (LiteLLM-gateway epic — prod-deploy scope remains)

## THE ONE MERGE BLOCKER — doc-number collision (needs your decision)

`main` allocated its OWN ADR-144/145/146 (delivery-queue / outbox / week-in-app) **and**
RFC-111 (curation-surfaces) while this branch used the same numbers for different docs
(vllm-provider / re-roll / reprocess / transcript-caching). On merge they collide
(duplicate numbers + `mkdocs.yml` conflict). `mergeable = CONFLICTING`.

**Why I did NOT auto-fix it overnight:** it's ~**70 files**, not just docs — the ADR
numbers are embedded in **code comments** (`ADR-144 B2` across ~30 provider files),
a **test filename** (`test_structured_response_reroll_adr145.py`), config, and docs. And
the *direction* (branch yields to main vs a different scheme) is a doc-policy call that's
yours. Too large + code-adjacent to sed blind with no CI safety net.

**Recommended fix (≈30 min, supervised):**
1. Renumber branch docs to next-free: **ADR-144→147, ADR-145→148, ADR-146→149, RFC-111→115**
   (all free on main). `git mv` the 4 files; then across `docs/ src/ config/ tests/
   mkdocs.yml tailscale/`:
   `sed -i '' -E 's/ADR-144/ADR-147/g; s/ADR-145/ADR-148/g; s/ADR-146/ADR-149/g; s/RFC-111/RFC-115/g'`
   on the ~70 files (`grep -rIlE 'ADR-14[456]|RFC-111'`). Rename the reroll test file too.
   `mkdocs.yml` nav is HAND-listed (update the 4 nav entries + paths).
2. `make docs` (mkdocs strict catches missed internal links) + `make ci-fast`.
3. `git fetch origin main && git rebase origin/main` — resolve the `mkdocs.yml` +
   `docs/wip/WIP_README.md` conflicts (take BOTH nav/entry sets). No other overlap expected
   (main's 16 commits are player/delivery/MCP/infra; this branch is providers/registry/profiles).
4. `git push --force-with-lease`. Then merge.

(Or tell me to do it — I have the exact plan staged.)

## What's in the PR (the arcs)

- **ADR-144** first-class vLLM / native-Qwen / LiteLLM providers (namespaced cost/telemetry,
  fail-closed served-model checks, airgapped DGX on real HF ids).
- **RFC-111** transcript-prefix caching across all 10 providers.
- **v2.5 finale** → cloud winners `cloud_openrouter` / `cloud_qwen` promoted to governed
  registry presets (+ `no_diarization` StageOption, `provider_chunked_gated_v25` GI config).
- **ADR-145** in-place re-roll · **ADR-146** corpus reprocess methodology.
- **Hardening (this session, 5 commits):** guardrail fence-strip, gateway real-cost telemetry,
  podcast_obs zero-config; + cleared pre-existing ci-fast debt (E501, 3 codespell
  false-positives, 2 stale tests).

## Finale ranking (for context — results are git-ignored .test_outputs)

Quality (blind Opus-4.8 vs 2.4 baseline, avg sum/ins/top) · cost $/9ep:
1. deepseek-native **8.41** ($0.740) — quality winner, clean 9-0
2. anthropic-haiku **8.29** ($1.346) — ⚠️ n=8, 2/9 episodes hard-failed (the fence bug now fixed)
3. gemini **8.04** ($0.971) · deepseek-or **8.04** ($0.68 telem / ~$0.20 real)
4. qwen-native **7.85** ($0.560) · qwen-or **7.70** ($0.53 telem / ~$0.05 real)

All arms beat 2.4 (avg 6.33). OR routes are the value story pending real-cost confirmation.

## Open items (the "last stabilizing round" — post-merge or on your say-so)

1. **anthropic finale re-run** — the fence fix is in; re-run the anthropic arm to confirm
   9/9 (no more fenced-JSON episode failures) and get its TRUE finale rank (currently n=8).
2. **Precise OR $/arm** — SpendLogs lacks per-model tokens; pin the real per-arm OR cost
   with a token-attributed query, or accept telemetry (now corrected for future runs).
3. **Scoped-out items to pull in:** govern litellm/qwen **speaker/cleaning** wire models
   (only summary_model governed so far); thread `response` into the GI-**evidence** cost
   path (non-bundled only; bundled already covered).
