# ADR-138: Versioned labeling profiles (a knob-bundle, not a code fork)

- **Status**: Proposed
- **Date**: 2026-07-29
- **Authors**: Marko Dragoljevic
- **Related**: ADR-017 (Registered Preprocessing Profiles — the cleaning precedent), ADR-128
  (ASR-mangled name recovery), ADR-135 (additive LLM role resolution), ADR-137 (text-normalization
  contract). `METHOD_VERSIONS["naming"]` in `workflow/processing_manifest.py`.

## Context & Problem Statement

**Cleaning** is versioned (ADR-017): `preprocessing/profiles.py` holds a `_PROFILE_REGISTRY` of
pure `text → text` functions selected by ID (`cleaning_v3`, `cleaning_v4`) via
`ml_preprocessing_profile`. A researcher can swap the whole cleaner and compare models with the
variable isolated. It works because cleaning is a **pure function**.

**Labeling** (the post-diarization speaker-naming "tier 3") has no equivalent, and it is exactly
where we keep innovating — today alone: narrated-desk cue vocabulary, case-blind metadata-anchored
self-intro, nickname/ASR-fuzzy binding, org-form rejection, "my name is" discovery, and Pattern-B
(bounded `unknown`-vs-`unidentified` classification + defect-share alarm). Every one of those was an
**ad-hoc edit to `roster.py`** (~1900 lines) or a scattered module constant
(`CAMEO_MAX_TALK_S`, `UNATTRIBUTED_TALK_ALARM`, `CO_HOST_INTRO_SHARE`, the new Pattern-B bound…).

Two gaps follow:

- **No isolation / comparison.** We cannot run "labeling as of last month" against "labeling today"
  on the same corpus with the variable pinned, the way ADR-017 lets us for cleaning.
- **Under-used provenance.** A version *tag* exists — `METHOD_VERSIONS["naming"]` (bumped to
  `naming-4` for today's work) — and it powers "reprocess every episode below naming-4". But the
  *behaviour* behind the tag is not declared anywhere; the tag is a label on a moving target.

## Decision

Introduce a **versioned `LabelingProfile`: a frozen knob-bundle, not a fork of the pipeline.**
Labeling is `(diarization, transcript, metadata, LLM) → roster` — a stateful pipeline with the LLM
in the loop, **not** a pure function, so ADR-017's swap-the-whole-function registry does not
transfer. What *is* extractable is the **configuration of** that pipeline.

- A `LabelingProfile` frozen dataclass bundles the labeling tunables. **Materialised in naming-4:**
  the boolean feature flags for the ADR-137 fixes (narrator-binding, case-blind self-intro,
  nickname fuzzy binding, first-name-only intro, merged-cluster suppression, Pattern-B bounded
  promotion, defect-share alarm) **and** the defect-alarm threshold — `roster.py` reads these from
  the profile. **Declared but not yet consumed (tier-2 extraction, tracked in
  `docs/wip/LABELING-TIER3-COMPLEXITY.md`):** the scalar floors/windows. `cameo_max_talk_s` is on
  the dataclass but the four cameo sites in `roster.py` still read the module constant
  `CAMEO_MAX_TALK_S`; the tape floor, the intro/co-host windows and the Pattern-B spare-name bound
  are likewise still module constants. Their defaults equal the constants, so a profile that leaves
  them at the default is a validated no-op today; wiring them is the next tier, not a behaviour
  change.
- Profiles are **registered by ID** and **selected via config** (one field, e.g.
  `labeling_profile: "naming-4"`), mirroring `ml_preprocessing_profile`.
- The **active profile ID is recorded in the per-episode sidecar** (next to the `voice_census` from
  the v2.4 sidecar directive) and aligned with `METHOD_VERSIONS["naming"]`, so every episode says
  which labeling behaviour produced it — the "keep an eye on these things" goal.
- `roster.py` reads the **feature flags + alarm threshold** from the profile today; the remaining
  scalar knobs are the tier-2 target. The algorithm stays where it is; only its knobs move behind a
  versioned, declarative boundary, tier by tier.

## Three tiers (what we do, and what we defer)

1. **Tag (exists).** `METHOD_VERSIONS["naming"]` → provenance + reprocess key. Bumped to `naming-4`.
2. **Knob-bundle profile (this ADR).** Centralize the labeling tunables into a versioned
   `LabelingProfile`, selected by ID, recorded in the sidecar. Declarative, reproducible,
   comparable — at a fraction of a full refactor. **This is the recommended work.**
3. **Swappable resolver registry (deferred).** A registry of alternative resolver *implementations*
   (à la ADR-017's function registry) to run two labeling *algorithms* head-to-head. Gated on a
   concrete need for side-by-side A/B; not done reflexively — the pipeline is not a pure function
   and forking it is expensive.

## Alternatives considered

1. **Mirror ADR-017 exactly (a registry of swappable resolver functions).** Rejected for now:
   labeling is stateful and LLM-coupled, not `text → text`; extracting a clean swap boundary is a
   large refactor whose payoff (side-by-side algorithms) we do not yet need. Kept as tier 3.
2. **Status quo (scattered constants, ad-hoc edits).** This is the problem — no isolation, no
   declared behaviour behind the tag.
3. **Config fields only, no profile object.** Adding the knobs as loose `Config` fields works but
   scatters them again; a named, versioned `LabelingProfile` bundle keeps "labeling v4" a single
   greppable, comparable unit.

## Consequences

- **Isolation + comparison:** run `labeling_profile: naming-3` vs `naming-4` on the same corpus.
- **Reproducibility + provenance:** the sidecar declares the labeling profile per episode.
- **Cost:** a mechanical refactor moving ~a dozen `roster.py` constants behind the profile, plus
  wiring the config field and the sidecar record. No algorithm change.
- **Discipline:** future labeling innovation lands as a new profile version + a tag bump, not a
  silent edit to a shared constant.

## Non-Goals

- Not changing any labeling *algorithm* — this is about *where the knobs live and how they are
  versioned*, exactly as ADR-017 was for cleaning.
- Not the swappable resolver-implementation registry (tier 3, deferred).
- Not touching the cleaning profiles (ADR-017 owns those).

## Follow-up

- Fix the stale `See ADR-029` reference in `preprocessing/profiles.py` → ADR-017.
