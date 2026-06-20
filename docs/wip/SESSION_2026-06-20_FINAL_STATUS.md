# Session final status — 2026-06-20

Branch: `feat/autoresearch-followups-2026-06-18`. **15 commits unpushed.**

## Issues closed / delivered

| Issue | Status | Commits |
|---|---|---|
| **#1033** — eval/prod code-path divergence audit | local-done (close awaiting op) | `c1109272` + `01863d03` |
| **#1034** — delete `summary_bullets` path entirely | local-done (close awaiting op) | `7135b618` + `1750dcbf` + `6b03d945` |
| **#1035** — NER pre-pass for KG entity extraction | **closed in GH** | `a236ef4d` + `86d01e3c` + `d8df114d` + `6e1a0f6a` |
| **#116** — Cell C re-baseline against Cell F NVFP4 | local-done | (Cell C re-baseline commit) |
| **#113** — small-model standoff (9B / safe / top dog) | local-done | (small-model standoff commit) |

## Headline results

### #1033 — corrected scoreboard
Surfaced eval/prod code-path divergence (eval used `summary_bullets`,
lossy; prod used `provider`). Flipped 40 configs + 2 profiles. Reran
7-candidate cohort. Filed #1034 + #1035 as follow-ups.

### #1034 — `summary_bullets` deletion (3 chunks)
- Chunk 1: Literal value removed, default flipped to `provider`
- Chunk 2: `_try_bullet_derived_extraction` + 4 dispatch arms deleted
- Chunk 3: 7 provider methods + helpers + Jinja template + metrics
  tracking deleted. **994 lines deleted, 47 added.**

### #1035 — NER pre-pass (4 phases, closed in GH)
- Phase 1: design spec
- Phase 2: `kg/ner_prepass.py` + v5 Jinja + Config flag + 7 providers
  + 26 new unit tests
- Phase 3: Validation on Cell F + Qwen3.5-35B-A3B. Surfaced + fixed
  scorer bug (KG-artifact shape). Both candidates: **0% → 100% entity
  coverage**, **+20–27pp topic coverage**, 0 false-positives
- Phase 4: Config default flipped True. Verdict doc + GH close

### #116 — Cell C re-baseline
Cell C (Ollama 35B): 66% topic / 83% entity. Cell F (vLLM NVFP4):
65% topic / **100% entity**. **Cell F dominates on the dimension #1035
engineered for.** Cell C demoted from autoresearch daily-driver
consideration; retained as Ollama fallback.

### #113 — Small-model standoff
- Qwen3.5:9b Ollama: 66% topic / **97% entity** at 6.6 GB footprint
- Moonlight-16B-A3B (was: safe pick): 54% topic / 93% entity
- Cell F NVFP4 (top dog): 65% topic / 100% entity
- Qwen3.5-35B-A3B vLLM (top quality): 77% topic / 100% entity

**NER pre-pass closes the model-size gap.** 9B model rides spaCy hints
to 97% entity recall — competitive with 30B+ models. Edge / local /
airgapped profiles can drop to Qwen3.5:9b without material KG quality
loss. Moonlight loses safe-pick status (highest topic + entity errors
in the sweep).

## Cross-cutting infrastructure / fix

- **Scorer bug fix** (`d8df114d`): `extract_entities_from_prediction`
  was reading `output.entities` (flat) instead of `output.kg.nodes
  [type=Entity]` (current production shape). All KG eval runs
  (#1016, #1022, #1033) silently reported 0% entity coverage. Fixed
  in #1035 phase 3.

## DGX state

- vLLM autoresearch slot: **Cell F NVFP4** (canonical daily-driver
  state, restored after every swap cycle)
- Ollama daemon: running (systemd-managed)
- No uncommitted homelab compose changes against `main`
- Total wall-clock spent on DGX swaps this session: ~1.5 hours
  (Qwen3.5 vLLM + Cell C Ollama + Moonlight vLLM + Qwen3.5:9b Ollama,
  plus Cell F restoration cycles)

## Tests / CI

- 4325 unit + integration tests pass post-#1035
- 1 skip (lancedb absent, pre-existing)
- 2 Gemini Client mock failures (pre-existing on parent commit
  `01863d03`; unrelated to this session's work)
- Audit test (`tests/integration/eval/test_autoresearch_config_source_audit.py`):
  50/50

## Open / not started this session

- **Push branch** — 15 commits unpushed (per `feedback_never_push_early`)
- **#1033 close** — local-done; operator authorization needed
- **#1034 close** — local-done; operator authorization needed
- **model_registry.py docstring updates** flagged by #113 — bundle with
  next functional registry touch (not blocking, doc-only)
- **benchmark_v2 held-out validation for Qwen3.5:9b** — recommended
  before flipping `airgapped_thin` / `local_dgx_balanced` to 9B; not
  started

## Recommendations for next session

1. Push the branch (operator authorization)
2. Close #1033 + #1034 with summary comments
3. Decide whether to bundle the registry docstring updates + 9B
   benchmark_v2 validation into this branch or a follow-up
4. Audit wave (Groups A–I in `AUDIO-WAVES-HARDENING-AUDIT.md`) still
   awaiting your go — separate from this autoresearch thread

## Commit log this session (15 commits)

```
6e1a0f6a feat(#1035) phase 4 — flip kg_extraction_use_ner_prepass default to True + verdict doc
d8df114d fix(eval): score_kg_topic_coverage reads entities from KG-artifact shape + runtime opts
86d01e3c feat(#1035) phase 2 — NER pre-pass implementation
a236ef4d docs(wip): #1035 phase 1 — NER pre-pass design spec
fb8e9b7c docs(wip): session status — #1033/#1034/#1035 done locally
6b03d945 feat(#1034) chunk 3 — delete provider bullets methods + corpus-side artifacts
1750dcbf feat(#1034) chunk 2 — delete bullet-derived KG/GI dispatch arms
7135b618 feat(#1034) chunk 1 — remove summary_bullets value from Literal
01863d03 docs(eval): #1033 — correct #1016 + #1022 addenda + registry headline_metric
c1109272 feat(eval): #1033 step 2 — corrected cohort scoreboard
+ Cell C re-baseline (#116)
+ Small-model standoff (#113)
+ This final status doc
```
