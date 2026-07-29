# naming-4 review — findings and resolutions (4 adversarial rounds + a self-review)

- **Status**: **GO.** Four adversarial advisor rounds each probed harder and found real
  corpus-wide wrong-name vectors; all closed with probe-tests. A subsequent three-reviewer
  self-review over the whole arc found more (nickname false-friends, a dead-code
  `episode_description`, two dead profile knobs); all fixed. Every gate green.
- **Date**: 2026-07-29
- **Context**: Before a full-corpus relabel (to measure the relabel-prompt-parity fix), the
  naming-4 labeling + cleanup arc was reviewed. **12 confident-wrong-name vectors** were caught and
  fixed across rounds 1-4 — every one reachable corpus-wide on the lowercase-turbo corpus, i.e. each
  would have baked wrong names into the relabel. This records each finding and what was done.

## Round 1 — GO-WITH-FIXES (F1 = the corpus-wide bug)

Fixed (with repro-before-fix tests)

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

## Round 2 — two residual wrong-name vectors (commit 66e3a001)

- **Possessive self-bind:** `"i'm sam altman's biggest fan"` self-bound **Sam Altman** (the
  possessive `altman's` folds to an edit-1 surname through the retained `i'm|i am|my name is` cues,
  which round-1 only closed for `this is`). Drop trailing-`'s` tokens from surname candidacy.
- **2-letter surname hole:** `"here with andrew ng"` + stated `Andrew Chen` bound the wrong Andrew
  (`ng` len-2 escaped the `>=3` contradiction check). `_span_has_contradicting_surname` now uses
  `>=2` plus 2-letter function words in `_INTRO_AFFILIATION_TOKENS`.

## Round 3 — four vectors (commit 2d6d4c81)

- **s-apostrophe possessive:** `"i'm reed hastings' successor"` (bare-apostrophe on an s-ending
  surname escaped the `'s` drop). Also drop trailing `'`.
- **Mid-show recap misattribution:** a past-tense cue `"earlier we spoke with andrew ng"` bound the
  named person to the next voice anywhere in the episode. Split `CUE_FIRST_PAST_BODY` out and gate it
  to head-of-episode + a host introducer.
- **Report-verb topical subject:** `"sam altman explains it best"` bound a bare metadata **subject**.
  Split `NAME_FIRST_REPORT_TAIL` out; on the match-form path it resolves only against **corroborated
  refs** (detected guests + known hosts) threaded from the caller — no fallback to raw metadata.
- **Fuzzy-surname order dependence:** stated `[Chris Smith, Chris Schmidt]` + `"chris schmidt"` bound
  Smith (shared soundex, first-match-wins). `_match_stated_in_span` is now two-pass: exact surname
  across ALL stated first, then fuzzy.

## Round 4 — two more, one self-inflicted (commit 658962e6)

- **Monologue-merge recap:** the round-3 head bound was on merged-turn INDEX, but a host monologue
  merges into turn 0, so a late recap inside it was still "head". Now also text-head-bound
  (first 1500 chars) + reject a temporal-recap-preceded match.
- **Absent-host report-verb (introduced by the round-3 fix-3 threading):** adding `known_hosts` to
  the corroborated set let `"kevin roose explains in his book"` (a topical mention of an ABSENT
  co-host) paint onto a guest. On the report-verb path a HOST name now binds ONLY a host voice.

Round 4 verdict: **GO** once these two land with probe-tests — which they did. No fifth full review
needed.

## Self-review (three read-only reviewers over the whole arc, before the relabel)

The advisor rounds were laser-focused on wrong-name binding in `roster.py`; a breadth review then
covered the rest and found more (all fixed):

- **Nickname false-friends** (correctness): the table merged DISTINCT people —
  `first_names_match("Alexander","Alexandra")` and `("Jonathan","John")` returned True. Split each
  into two groups sharing only the ambiguous short form. Pat/Ted verified surname-safe (the formal
  names never cross-match). *(97fbb2db)*
- **Dead `episode_description`** (correctness): `Episode` had no `description` field, so the ADR-135
  role prompt only ever saw the title — in BOTH the relabel fix and the pre-existing full path. Added
  the field + populated it from the per-item `<description>`. *(97fbb2db)*
- **Two dead profile knobs** (ADR-138 integrity): `nickname_fuzzy_binding` + `cameo_max_talk_s` were
  declared but unread, so a naming-3-legacy A/B would not have isolated them. Both now wired from the
  profile with A/B tests proving the flip; naming-4 unchanged (defaults == prior constants).
  *(2302718f)*
- **Small fixes:** recap-marker lookback widened 25→40 chars; the dead `UNATTRIBUTED_TALK_ALARM`
  constant removed; F3 host-snap role-gate + Q3 divergence-log test gaps closed. *(e383943d)*
- **Doc drift** (this doc + ADR-137/138): ADR-137's match recipe corrected (NFKD, not NFKC), adopter
  scope narrowed to roster (resolution.py is a follow-up), provider-surface declaration marked
  deferred; ADR-138 updated to reflect the two now-consumed knobs.

Post-review follow-ups landed before push:

- **o11y git_sha provenance** (correctness): `update_stage` loaded the frozen manifest but never
  refreshed `git_sha`, so on a reprocess/relabel every `pipeline_stage` event carried the ORIGINAL
  build's sha, not the re-running HEAD. Now `data.update(git_ground_truth())` on every write;
  per-stage code provenance stays carried by each stage's `method_version`. Test:
  `test_update_stage_refreshes_git_sha_on_an_existing_manifest`.
- **Quadratic guest-intro regex** (efficiency): the pre-existing O(n²) in `_NAME`/`_NAMES` (two
  nested unbounded quantifiers over a capitalised run — 60k chars 3.3s, 120k 13s) is fixed by
  bounding the token-run to `{1,5}` and the name-list to `{0,9}` (a real person-name is ≤6 tokens,
  an on-air intro ≤10 people; longer is org/ASR noise the downstream guards reject). Linear now
  (<0.05s at 120k), identical matches on real intros. The existing regression test used lowercase
  text that never hit the case-bound name path — rewritten to a capitalised worst case at a 2s
  budget the old pattern fails.

Accepted / not fixed here: the tape `unknown`→`unidentified` reclassify is already implemented by
Pattern-B bounded promotion (no change).
