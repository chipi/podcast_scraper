# ADR-112: The registry is the source of truth for every tuned parameter

**Status:** Accepted
**Date:** 2026-07-15
**Deciders:** Marko
**Related:** ADR-111 (eval and pipeline are one set of bricks), #907 (registry-layered profiles), #939 (same-vendor judge bias)

## Context

The model registry already governed *which model* runs each stage, materialized that into the
profile YAMLs, and enforced agreement with a drift test (the diarization promotion, #1170, is the
worked example). But the **tuned parameters** — `max_insights`, temperature, dedupe threshold,
prompt version, the QA/NLI floors, the value-gate switch — were governed by nobody. A value could
come from four places at once, and they disagreed silently:

| where | example |
| --- | --- |
| registry `StageOption.extra_settings` | `max_insights: 12` |
| profile YAML | `gi_max_insights: 50` |
| `Config` field default | `default=20` |
| `config_constants.py` | a hardcoded literal |

`gi_max_insights` really was **12 / 50 / 20** across those three doors at the same time, and
production ran whichever door the caller came in through. Worse:

* `provider_chunked_gated_v3` — the entire researched v3 tuning, temperature pin and all — was an
  **orphan**: no preset pointed at it, so the measured configuration reached zero production
  profiles. The profiles hand-copied their way to something else.
* `gi_insight_temperature` was recorded in the registry and plumbed to no `Config` field, so every
  scored bake-off arm sampled at **0.3** while its YAML said **0.0** — and a model that disagrees
  with itself between runs was about to be credited with "finding knowledge the other model missed".
* `gi_value_gate_enabled` defaulted to `False`, so a profile that forgot the key shipped **ungated**
  and the run looked fine.

Every one of these is ADR-111's allowlist bug, one layer along: a key nobody copies does not error,
it takes a default, and the default is usually "off".

## Decision

**The registry owns every tuned parameter. Profiles are a generated VIEW. The default is the
registry's value. Nothing is hardcoded. An integrity check makes forgetting a red build.**

1. **`REGISTRY_GOVERNED_FIELDS`** — one list, in the registry, of every field the registry owns.
   Read by both the materializer and the integrity test, so there is no second allowlist to drift.

2. **`make profiles-materialize`** regenerates the governed fields of every profile YAML from the
   registry, line-surgically (comments preserved — the profiles are ~57% comments and they carry
   the reasoning). `--check` is the CI gate.

3. **Config defaults equal the registry.** A caller that loads no profile must not run a different
   pipeline than the one we measured. Bound by `test_the_config_default_is_not_a_trap`.

4. **The value-gate judge is DERIVED, never pinned.** It must be vendor-disjoint from the summariser
   (#939): pin one literal judge across a multi-vendor bake-off and that vendor's own arm grades
   itself and gets a free pass while every rival is held to a stricter bar. `resolve_value_gate`
   returns the first policy vendor that is not the defendant.

5. **The gate is an LLM asking a question, so it does not exist on the pure-ML path.** For a
   summariser that is not an LLM (transformers, summllama), the gate is *inapplicable* — not off by
   preference. Enabling it there would reach for a hosted judge, which for `airgapped` is a network
   call (the one thing airgapped forbids) and for `dev` is a paid LLM call from CI. Enforced both in
   the resolver and at the point of use in `value_gate.py`.

6. **The integrity check is CLOSED-WORLD.** `test_every_gi_tunable_is_either_governed_or_explicitly_exempt`
   fails the build for any `gi_*` / `gil_*` field that is neither registry-governed nor on a short,
   argued exempt list. Every other guard checks a list someone remembered to update; this one checks
   the list itself. You cannot forget to register a param, because forgetting is a red test.

## The loop this creates

An eval finds a better value → it goes in the `StageOption` (with a `research_ref` naming the
report) → `make profiles-materialize` → every profile inherits it. That is the new default,
everywhere, in one commit. Superseded options are marked `tier="deprecated"` with a
`deprecation_reason`, kept as history — knowing *why* n=12 was wrong is worth more than the 12 was.

## Alternatives considered

**A. Keep the drift test, add the missing fields to it.** That is what existed. It only compared
fields *present* in the YAML, so a profile that omitted a knob was never checked and silently
inherited a code default — absence was indistinguishable from agreement. And it is a list someone
must remember to extend. **Rejected:** it already failed, in exactly this way.

**B. Make profiles the source of truth, registry advisory.** Backwards. The registry is where a
value earns a `research_ref`; the YAML is where it gets hand-copied wrong. **Rejected.**

## Consequences

**Good.** A tuned value has exactly one home. A new param cannot hide on a code default. The eval and
production cannot diverge on a knob, because both resolve from the registry. The bake-off measures
the model, not our config drift.

**Cost.** The exempt list (`NOT_REGISTRY_GOVERNED`) is a judgement call per field and must stay
short and argued; a lazy addition there is the one remaining way to opt out of governance. The
integrity test is the place that pressure surfaces.

**Not addressed here.** The KG and NER stages have the same shape and were not audited. They should
adopt `REGISTRY_GOVERNED_FIELDS` the same way rather than growing a parallel mechanism.
