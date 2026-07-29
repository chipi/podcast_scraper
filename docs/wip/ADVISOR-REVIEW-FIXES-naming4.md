# naming-4 advisor review — findings and resolutions

- **Status**: All actionable findings resolved; two accepted-as-is with rationale. Second advisor
  review pending.
- **Date**: 2026-07-29
- **Context**: Before a full-corpus relabel (to measure the relabel-prompt-parity fix), an advisor
  pass reviewed the naming-4 labeling + cleanup arc (commits 915ea668^..HEAD). Verdict:
  **GO-WITH-FIXES**. It caught a confident-wrong-name bug (F1) that would have fired for **every
  voice of every episode** on the lowercase turbo corpus — i.e. corrupted the relabel we were about
  to run. This records each finding and what was done.

## Fixed (with repro-before-fix tests)

| ID | Finding | Resolution |
| --- | --- | --- |
| **F1** (blocking) | `_metadata_anchored_self_intro` scanned the WHOLE voice text (siblings bound to 5000 chars) and its match-form `this is <Name>` cue accepted a fuzzy surname — "this is sam altman's company" bound **Sam Altman** to the speaker; fires corpus-wide on lowercase turbo. | Head-bound the scan to `[:5000]`; **dropped `this is`** from `_SELF_INTRO_MATCHFORM` (its capitalized sibling `extract_self_introduced_host` is "I'm"-only for this exact reason). Tests: `test_f1_third_person_this_is_never_self_binds`, `test_f1_self_intro_scan_is_head_bounded`. |
| **F2** | First-name-only cue binding ran ungated on every turn and fell through to a bare-first-name bind even when the span carried a contradicting surname — "here with akshat kanaparthy" bound stated "Akshat Bubna". | Gated the first-name-only relaxation to **host-hint introducer turns**; added `_span_has_contradicting_surname` so a real surname after the first name refuses the relaxation. "here with akshat of moto" (affiliation) still binds. Tests: `test_f2_first_name_only_declines_on_contradicting_surname`, `test_f2_first_name_only_declines_from_non_host_turn`. |
| **F3** | `_intro_reader_voice_names` applied the host-ward canonicalization (nickname-as-exact + surname edit ≤ 3) with no role gate — a guest introduced as "Rich Perkins" could be renamed to host "Richard Parker". | Snap toward a known host only when the target voice is itself a host voice (`conv_hosts`) — mirrors the guard `_recover_stated_names` already applies. Guests still canonicalize to the stated **person**. |
| **F5** | `cameo_max_talk_s` is declared on `LabelingProfile` but read nowhere (roster uses the module constant `CAMEO_MAX_TALK_S` at four sites); ADR-138 claimed roster reads all tunables from the profile — doc-vs-code divergence. | Amended ADR-138 to state accurately what is profile-driven now (the ADR-137 feature flags + alarm threshold) vs the tier-2 scalar-extraction follow-up (cameo/tape floors, intro/co-host windows); marked the knob NOT-YET-CONSUMED in-code. Default equals the constant → no behaviour change. Full wiring is the tier-2 target. |
| **F6** | Unknown `labeling_profile` id warned and fell back to naming-4 — a typo'd A/B run would silently produce naming-4 data. | Added a fail-fast Config `field_validator` (`_validate_labeling_profile`) that rejects an unregistered id at construction. The pipeline fallback stays as defense-in-depth. Test: `test_labeling_profile_validator_rejects_unregistered_id`. |
| **Q3** | In `relabel_only`, `feed_hosts` came only from the stored sibling metadata (deterministic parse); when the sibling is missing it returned `[]` — the worst anchor state — while the live `job.feed_hosts` was computed and discarded. | Keep the **freeze** (reproducible relabel), but fall back to live `job.feed_hosts` **only when the sibling is missing/empty**; log any sibling-vs-live divergence to inform a future freeze-vs-live decision. Test: `test_relabel_feed_hosts_falls_back_to_live_when_sibling_missing`. Also fixed the prompt-parity half separately (BUG#1: title/description). |

## Accepted as-is (with rationale)

- **F4 — merged-cluster suppression false-positive (fails SAFE).** A host who *quotes* a stated
  person's self-intro in their first 5000 chars ("…and she said, I'm Katie Martin…") plus their own
  intro maps to 2 stated people, so `_distinct_intros_map_to_multiple_stated` suppresses the host's
  legit name. This is an **under-name**, the safe direction per #876 (no name beats a wrong name).
  No wrong-name path. Left as-is; the `source`/sidecar makes it visible — worth a census after the
  relabel, not a pre-run code change.
- **F7 — old-format corpora only.** Relabel anonymizes by distinct `speaker_label` when `speaker`
  is None; `_recover_stated_names` gives both halves of an over-split the same canonical spelling,
  so on a **v2-format** corpus two clusters could collapse to one under relabel. New-format segments
  carry raw `speaker` ids, so the current corpus is unaffected. Noted, not fixed.

## Measurement guidance carried into the relabel (advisor Q4)

- Gemini resolution temperature is pinned to 0.0, but is not bit-deterministic and `detect_speakers`
  is a second live LLM at relabel time — so run a small **A/A** (relabel 10–20 episodes twice, diff)
  to establish the churn floor **before** attributing the A/B (fixed-vs-prior) delta to the fixes.
- `resolution_attribution` (pure-cue baseline + `llm_delta`) is written per episode — bucket every
  before/after change into **deterministic vs LLM** and report the deterministic diff as the
  headline, so Gemini noise is not mistaken for the fix's effect.
