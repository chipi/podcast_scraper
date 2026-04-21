# Backend 2.6 Close-Out

Summary of what shipped in the 2.6 epic and what's deferred to 2.7.
This is a handoff doc — not a plan. For active work, see
`PLAN_NEXT_SESSION.md` and `POST_REINGESTION_PLAN.md`.

## PRs that shipped 2.6

| PR | Focus | Key outcomes |
| --- | --- | --- |
| **#624** | Speaker flow validation + CI fixes + acceptance assertions (#598, #622) | Speaker-detection pipeline stabilised; acceptance tests catch drift. |
| **#628** | Eval baseline + KG v3 prompt + SummLlama (#590, #571, #625) | Cross-dataset baseline locked; KG noun-phrase prompt is default. |
| **#642** | Whisper cost autoresearch + audio preprocessing presets + meta-tensor fix (#577, #634) | `speech_optimal_v1`: 32 kbps + -30dB/0.5s silence. 75 % smaller files, <1% WER impact. |
| **#646** | Mega-bundle pipeline modes + corpus layout + JSON truncation (#643, #644, #645) | All 6 cloud providers support `mega_bundled` / `extraction_bundled`. `cloud_balanced` default now 72 % cheaper than staged at same artifact counts. |

## Features that reached production

- **Single-call mega-bundle summarisation.** `llm_pipeline_mode=mega_bundled`
  returns summary + bullets + insights + topics + entities in one LLM call.
  All 6 cloud providers (Anthropic, DeepSeek, OpenAI, Gemini, Mistral, Grok).
  Real-episode validation at `scripts/validate/validate_phase3c.py`.
- **Two-call extraction bundle.** `llm_pipeline_mode=extraction_bundled`
  keeps the staged summary prompt but collapses GIL+KG into one call.
- **Cloud-LLM JSON truncation floor.** `cloud_llm_structured_min_output_tokens`
  (default 4096) prevents silent empty summaries on long transcripts.
  Applied to every cloud provider.
- **DeepSeek HTTP timeout knob.** `deepseek_timeout` (default 600 s) —
  mega-bundle on DeepSeek needs more than the stock 120 s.
- **Single-feed corpus layout.** `single_feed_uses_corpus_layout=True`
  wraps single-feed output under `feeds/<slug>/run_*/` matching multi-feed.
  Migration helper at `scripts/tools/migrate_single_feed_to_corpus.py`.
- **Audio preprocessing profile.** `audio_preprocessing_profile: speech_optimal_v1`
  is the shipped recipe. Tuned on real 5-episode sweep; 75 % file-size drop at <1 % WER cost.
- **KG v3 prompt.** Noun-phrase topics instead of sentence slugs; default everywhere.
- **Insight clustering + multi-quote GIL** (from #599–#601 into #611).
  Grounded-quote-per-insight, cross-episode cluster browse.

## Profile defaults (as of main @ 00e87a2d + followups branch)

- `cloud_balanced`: Gemini flash-lite **mega_bundled**, whisper-1,
  speech_optimal_v1 audio, GI provider + KG provider.
- `cloud_quality`: Anthropic haiku-4.5 **mega_bundled**, whisper-1,
  speech_optimal_v1 audio.
- `local`: Ollama `qwen3.5:9b` bundled (held-out 0.529/0.509).
- `airgapped`: Transformers + local Whisper.
- `dev`: Cheap + fast for iteration.

## Validation harnesses shipped

- `scripts/validate/validate_phase3c.py` — dispatch + cost + artifact counts
  for mega/extraction bundled across all 6 cloud providers.
- `scripts/validate/validate_layout_644.py` — real-RSS CLI invocations
  covering single-feed / multi-feed / flag-off / flag-on directory layouts.
- `scripts/validate/validate_post_reingestion.py` — runs the 6 explore-expansion
  CLI commands on a reingested corpus and applies soft gates from
  `POST_REINGESTION_PLAN.md`.
- `docs/guides/REAL_EPISODE_VALIDATION.md` codifies the pattern.

## Deferred to 2.7

- **#286 — gpt-4o-transcribe chunking.** Needs upstream chunking because
  the model has a 1400 s duration cap + tight payload limits. Not in 2.6
  due to preprocessing edge cases (RFC / design pending).
- **Voxtral hallucination investigation.** #641 closed as "not a win"; if
  we want to revisit in 2.7 we need deeper research on anti-loop mitigations.
- **Qdrant vector backend (RFC-070).** Removed from `vector_backend`
  Literal in #646 until wired. Re-add when the backend lands.
- **Real-episode validation for `local` / `airgapped` / `dev` profiles.**
  Config-layer check passed (parametrized test in
  `tests/unit/podcast_scraper/test_cli_profile_routing.py`), but no
  real-audio end-to-end run on those profiles. Cheap to add.
- **KG/GI per-stage provider routing.** `cloud_quality` currently picks one
  provider for both summary + GIL + KG; the "best provider per stage" idea
  is still open work (referenced in `cloud_quality.yaml` comments).

## Post-merge follow-ups (not scheduled)

Noted inline in `docs/wip/PLAN_NEXT_SESSION.md`:

- Real-episode E2E for `single_feed_uses_corpus_layout` (#644 follow-up).
  **Done in #646 followups branch via `validate_layout_644.py`.**
- Real-episode E2E for `audio_preprocessing_profile` (#634 / #642 follow-up).
  **Done in #646 followups branch via `test_audio_profile_wiring.py`.**

## Known guardrails that prevent regressions

- `@model_validator` in Config for `single_feed_uses_corpus_layout` —
  every construction path wraps consistently, not just `service.run()`.
- `_build_config` forwards `llm_pipeline_mode`, `cloud_llm_structured_min_output_tokens`,
  `deepseek_timeout`, `audio_preprocessing_profile`, `ml_preprocessing_profile`
  so `--config` YAML doesn't silently drop them.
- `--profile NAME` routes through `_load_and_merge_config` so argparse
  defaults don't clobber profile values.
- Parametrized CLI round-trip test locks every YAML field in every shipped
  profile.
- Provider-dispatch completeness tests for `transcription_provider` and
  `summary_provider` Literals — new values can't ship without wiring.
- Real-episode validation rule in `CLAUDE.md` — production code changes
  require a run against real audio/transcripts before push.

## Quick links

- `docs/guides/AI_PROVIDER_COMPARISON_GUIDE.md` — which provider to pick
  for which scenario; real-episode numbers per provider.
- `docs/guides/REAL_EPISODE_VALIDATION.md` — when/how to build a harness.
- `docs/wip/PLAN_NEXT_SESSION.md` — what's queued after 2.6 close-out.
- `docs/wip/POST_REINGESTION_PLAN.md` — 6-step plan for validating a
  re-ingested production corpus.
