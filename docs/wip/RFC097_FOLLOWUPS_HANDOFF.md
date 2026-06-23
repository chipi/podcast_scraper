# RFC-097 follow-ups — session handoff (2026-06-23)

Spawned from PR #1061 (merged 2026-06-23T14:28Z). This branch
(`feat/rfc097-followups`, off `c48b01b1`) is the next-session
landing zone for the 4 GH issues #1061 deliberately left open.

## Status (2026-06-23 PM)

- **#1060 — DONE** (commit `36ed9274`). Operator picked D1=B, D2=A, D3=A.
  Doc-only resolution for D1; fresh smoke_v2 benchmarks for D2 + D3
  produced four new `StageOption` entries (`local_whisper_tiny_en`,
  `local_whisper_medium_en`, `summllama_3_2_3b_paragraph`,
  `transformers_bart_small_long_fast_authority`) plus three new
  `ProfilePreset`s (`airgapped`, `airgapped_thin`, `dev`). Drift test
  17/17 green. Eval report:
  `docs/guides/eval-reports/EVAL_DEV_TIER_REGISTRY_2026_06_23.md`.
- **#1048 — DONE** (commit `ceeb0485`, Option 3 path). Person Landing
  rail panel is now the PRD-029 shared shell: "Person Profile" +
  "Position Tracker" tab pair (the latter is a placeholder pointing at
  #1049), identity-header role/episode-count/organization-chips, and an
  `ABOUT∩MENTIONS_PERSON` ranked-topic overview. All previously-shipped
  rail content moved into the Person Profile tab. New parsing helpers:
  `rankedPersonTopicMentions`, `rankedPersonOrganizations`,
  `personRoleFromNode`. Vitest 2227/2227 green; viewer build (vue-tsc
  strict) green. PRD-028 / PRD-029 / RFC-097 closure stanzas now cite
  the actual ticket numbers; surface map + e2e spec updated.
- **#1049 — DONE** (commit `25ab4db9`). Per-(Person, Topic) position arc
  inside the Position Tracker tab: `MENTIONS_PERSON ∩ ABOUT` join, sort
  by `publish_date` then `position_hint`, multi-select `insight_type`
  filter chips, three UXS-009 states. Entry from ranked-Topic rows on
  the Person Profile tab.
- **#1050 — DONE.** Person Profile tab now renders the UXS-010
  sections: Topics discussed / Insights voiced (grouped by Topic) /
  Episodes appeared in (SPOKE_IN list) / Organizations affiliated /
  the existing PRD-033 corpus + stated-positions + attributed-quotes
  blocks. Each Topic-group header reuses #1049's
  `selectTopicForPositionTracker` entry point.

## Issues to close on this branch

| # | Title | Status |
|---|---|---|
| ~~**#1060**~~ | ~~Promote YAML-only profiles to ProfilePreset~~ | **DONE — commit `36ed9274`** + 4 FU commits |
| ~~**#1048**~~ | ~~viewer: Person Landing shared component~~ | **DONE — commit `ceeb0485`** |
| ~~**#1049**~~ | ~~viewer: Position Tracker (Person × Topic over time)~~ | **DONE — commit `25ab4db9`** |
| ~~**#1050**~~ | ~~viewer: Person Profile (everything about a Person)~~ | **DONE — this commit** |

## #1060 — Open decisions BEFORE code

This issue can't move forward without operator input on three things:

### Decision 1: `ProfilePreset` redesign vs accept YAML-only as first-class?

`ProfilePreset` today pins both PROVIDER and MODEL per stage as a 6-tuple
of `StageOption.option_id` keys. `test_default` only pins MODEL choices
across vendors and lets Field defaults pick PROVIDER — a fundamentally
different shape. Two paths:

- **(A) Redesign `ProfilePreset`** to support "model-only" or
  "provider-only" presets. Cleaner long-term but RFC-level discussion
  (drift test, eval reports, registry docs all assume the 6-tuple).
- **(B) Accept structurally YAML-only** as first-class. Document the
  pattern; keep the drift test's "documented YAML-only" check covering
  it. Cheap; matches what we already shipped in PR #1061.

**Recommendation pending operator**: (B) — pragmatic, no RFC overhead.
(A) only if the goal is "every profile lives in `_PROFILE_PRESETS`" as
a hard invariant.

### Decision 2: `summllama` provider — promote or migrate?

`airgapped.yaml` uses `summary_provider: summllama` which has no
matching `StageOption` in the registry. Two paths:

- **(A) Ship a `summllama_*` StageOption** with `research_ref` +
  `headline_metric` + `measured_at` from a real benchmark run. Requires
  someone to actually run the benchmark.
- **(B) Migrate `airgapped` to a different summary provider** that
  already has a registry presence (e.g. transformers + bart-base or
  ollama). Changes the airgapped profile's runtime behavior.

**Recommendation pending operator**: (B) if airgapped's quality
contract is loose; (A) if the SummLlama path's quality is load-bearing
for that profile.

### Decision 3: Empirical benchmark data for dev/airgapped tier?

`StageOption.research_ref` is intended as a non-optional pointer to an
eval report justifying the choice. For dev / airgapped / test tier
StageOptions we don't HAVE benchmark data — the choices are
ergonomic ("smallest / fastest / runs without GPU") not empirical
("highest ROUGE on v2"). Two paths:

- **(A) Actually run the benchmarks** for dev-tier StageOptions
  (tiny.en Whisper, bart-base summary, etc.) so research_ref is honest.
- **(B) Accept placeholder research_refs** like `"ergonomic-default"`
  or `"todo-rebenchmark"`. Signals the gap honestly but compromises
  the registry's "every option is research-justified" property.

**Recommendation pending operator**: (B) with an explicit
`evaluation_status: "ergonomic-only"` field added to `StageOption` so
the gap is structurally visible (not just a string in research_ref).

## What's already in place from PR #1061

- Drift test (`tests/integration/providers/ml/test_profile_yaml_registry_drift.py`)
  with the 3 extensions:
  - `test_every_registry_preset_has_a_matching_yaml` (orphan-preset check)
  - `test_every_yaml_only_profile_documents_its_status` (each YAML-only
    profile must carry a documentation marker)
  - `test_every_profile_pins_model_for_its_summary_provider` (consistency)
- Each YAML-only profile in `config/profiles/` carries a "Registry
  status: YAML-only" comment with its specific gating factor — read
  these before deciding the per-profile path.

## Definition of done for #1060

The drift test's `_PRESET_WITHOUT_YAML_ALLOWLIST` set stays empty AND
every YAML-only profile is either promoted to a `ProfilePreset` OR the
operator's recommendation (B) above is accepted + documented (the
existing YAML-only documentation pattern becomes the operative answer
and #1060 closes "won't promote" rather than "promoted").

## What's NOT in scope

- `profile_freeze.example.yaml` deletion — handled separately when the
  docs-link migration completes (post-v2.7).
- Refactoring `ProfilePreset` dataclass shape itself — only if Decision
  1 (A) is chosen.

## How to resume

```bash
cd /Users/markodragoljevic/Projects/podcast_scraper-FUTURE
git status                  # confirm on feat/rfc097-followups, clean
git fetch origin main && git rebase origin/main  # if main has moved
# read this doc + decide on the 3 #1060 questions
# then ask Claude to start on the chosen path
```

Or skip #1060 and start with #1048 (viewer foundation) — they're
independent. The recommended order above is operator preference, not
a hard dep.
