# ADR-124: Model governance — only registry-sanctioned models may run (opt-in)

- **Status**: Accepted
- **Date**: 2026-07-22
- **Authors**: Podcast Scraper Team
- **Tracking issue**: [#1258](https://github.com/chipi/podcast_scraper/issues/1258)
- **Related ADRs**: [ADR-123](ADR-123-quality-gate-transcription-failover.md) (coverage failover —
  its failover model is gated too), the registry-governance pattern in
  [ADR-122](ADR-122-self-hosted-model-resilience-policy.md).

## Context & Problem Statement

The model registry's StageOptions are the source of truth for *vetted* models — each carries the
eval/research that justified it. But nothing stopped an **unvetted** model from actually running: a
CLI `--dgx-whisper-model <typo>`, a hand-edited profile, or a copy-paste of an un-benchmarked model
would silently transcribe/summarise. For the v2→v3 reprocess this is unacceptable — the
reprocess-once economics do not survive discovering, after 1000 episodes, that a typo'd or
un-benchmarked model produced the corpus. (Adopting turbo made this concrete: turbo was the model we
*wanted*, yet it was not even in the registry — see below.)

## Decision

A **model-governance gate**, opt-in via `enforce_model_governance`. When on, every **active** model
— the model the CONFIGURED provider for each stage (transcription incl. the ADR-123 coverage
failover, summary, diarization) will actually run — must appear in that stage's **sanctioned set**
(the `.model` values of the stage's registry StageOptions). An unsanctioned model raises
`UnsanctionedModelError` with a stable code `MODEL_NOT_SANCTIONED`, naming the stage, config field,
offending model, and the allow-list.

Design points:

- **Opt-in.** `enforce_model_governance` defaults **off**, so the thousands of tests and
  experiment-mode runs (base.en, tiny, ad-hoc eval models) are unaffected. The **reprocess profiles
  turn it on** (`reprocess_dgx_{turbo,no_llm}`); serving/eval presets stay off until their summary
  sanctioned set is verified complete. The flag is a `REGISTRY_GOVERNED_FIELD`, materialized into
  every profile and drift-checked.
- **Active models only.** A Config carries `{provider}_*_model` fields for *every* provider, but only
  the selected provider's model runs. Gating all of them would falsely reject a config that merely
  carries a default for an unused provider, so the gate resolves the active model per stage from the
  configured `*_provider`.
- **Typed, catchable error.** `UnsanctionedModelError` is a `RuntimeError` (not `ValueError`) so that,
  raised from the pydantic `model_validator`, it **propagates as itself** — pydantic wraps
  `ValueError`/`AssertionError` into a generic `ValidationError`, which would erase the type + code.
- **Turbo is now sanctioned.** Enforcing governance surfaced that `large-v3-turbo` — the #1178
  reprocess pick — was not a registry StageOption. It was added (`tailnet_dgx_whisper_turbo`), so the
  reprocess profiles pass. This is the intended loop: to *use* a model you must first *sanction* it.

## Alternatives Considered

- **Always-on hard reject.** Rejected: breaks the test suite + experiment mode unless every test
  model is added to the registry — huge surface, high churn, no benefit for non-corpus runs.
- **Gate every `*_model` field.** Rejected: false-rejects defaults for unused providers.
- **`UnsanctionedModelError(ValueError)`.** Rejected: pydantic would wrap it, erasing the specific
  type/code the requirement asked for.

## Consequences

- **Positive**: a reprocess can only run vetted models; an unvetted model fails fast at config build
  with a specific, machine-identifiable error. Sanctioning a model is a deliberate registry act with
  its research attached.
- **Negative**: adopting a new model now requires adding a StageOption first (a feature, not a bug —
  it forces the vetting record). The active-model resolver is a provider→field map that must track
  new providers (transcription/summary/diarization covered now; GI/KG/NER models are a follow-up).
- **Neutral**: off by default → serving/experiment unchanged.

## Implementation Notes

- `providers/ml/model_governance.py`: `sanctioned_models(stage)`, `active_models(cfg)`,
  `assert_models_sanctioned(cfg)`, `UnsanctionedModelError` (+ `MODEL_NOT_SANCTIONED`).
- `Config._enforce_model_governance` (`model_validator(mode="after")`) calls it when the flag is on.
- Registry: `enforce_model_governance` added to `ProfilePreset` + `REGISTRY_GOVERNED_FIELDS`,
  emitted by the resolver, materialized (`make profiles-materialize`), on for the reprocess profiles.
- `tailnet_dgx_whisper_turbo` StageOption added to `_TRANSCRIPTION_OPTIONS`.

## References

- Issue [#1258](https://github.com/chipi/podcast_scraper/issues/1258) — scope + acceptance.
- [ADR-123](ADR-123-quality-gate-transcription-failover.md), [ADR-122](ADR-122-self-hosted-model-resilience-policy.md).
