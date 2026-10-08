# Naming decision trace — reconstruct the speaker-naming ladder for any episode

| | |
| --- | --- |
| Doc version | 4 |
| Last updated | 2026-10-08 |
| Trace schema | `TRACE_VERSION = 1` (`src/podcast_scraper/providers/ml/diarization/naming_trace.py`) |
| Implemented by | `448151fab` (roster trace), `dbed527d2` (rung tests), `b5923aab4` (LLM answers), `c8bda243d` + `11e75c7ab` (inside the helpers) — #2276. Rules since: `93f49482c` (`introduced` source), `c05cc0273` + this version's commit (one person, one entry) |
| Code content hash | `2c04077820ad` (see below) |
| Status | Phase 1 **deployed**: prod sidecars written since the deploy carry `decision_trace` (114 of 2,451 served episodes with diagnostics on 2026-10-08; older ones were never re-ingested). `roster_replay --trace-out` reconstructs the rest offline |

**How to tell whether this doc matches the code.**

- `tests/unit/podcast_scraper/providers/ml/diarization/test_naming_trace.py` reads the trace schema
  row above. It fails when that row differs from `TRACE_VERSION`. A schema change cannot land
  without this doc.
- The content hash is the first 12 hex digits of the sha256 of `naming_trace.py`, `roster.py` and
  `pipeline.py`, concatenated in that order:

  ```sh
  cat src/podcast_scraper/providers/ml/diarization/{naming_trace,roster,pipeline}.py | shasum -a 256 | cut -c1-12
  ```

  A different value means the ladder or the recorder has changed since this doc was last checked
  against it. Re-read the "Recorded" and "NOT recorded" sections before you rely on them.

**Changelog**

- **v4, 2026-10-08.** In line with the code again: the `host_naming_pair` rung and the
  `address_pair` seat naming were missing; the `introduced` source (`93f49482c`) is no longer
  pending; the trace is deployed. The one-person-one-entry rules (`c05cc0273` and the follow-up)
  are described at the rungs where they act ("Same-person rules" below), and the record-level
  check and m0025, which happen outside the trace, are listed under NOT recorded.
- **v3, 2026-10-04.** v2 called phase 1 "Done"; it was not — the helpers' decisions (host-seat
  steps, forced names, guest variants, one-name groups, voice-type reasons) were listed as phase 1
  in v1 and not recorded. Now recorded, plus both LLM answers in full (moved forward from phase 3),
  a test per rung, an end-to-end run with the real LLM, and an old-vs-new behaviour gate.
- **v2, 2026-10-04.** Phase 1 implemented and reviewed. The design text is replaced by what the
  code records, the real sidecar shape, what is NOT recorded, and the purity-gate result.
- **v1, 2026-10-04.** Design, written from the 31-rung inventory.

## Goal

The operator set this goal on 2026-10-04. Every episode's sidecars must hold enough to reconstruct
how each voice got its name and role: which rung proposed what, what won, what was overridden, and
why. With that we can:

- measure the priority order instead of arguing about it;
- find patterns across the corpus;
- design new solutions from data.

## Why: what the inventory found

The ladder has **31 rungs**, from feed host detection to the published roster (inventory of
2026-10-04). Before phase 1, the diagnostics sidecar
(`<transcript>.speakers.diagnostics.json`) recorded only three things:

- the **inputs** (`tried`);
- the **final** per-voice result (`voices[]`);
- `resolution_attribution`.

The `source` field is **lossy**. `self_intro` is the catch-all for every intro entry that is not
from the publisher or the LLM. Downstream code (the metadata show-name filter, m0014, the
copresence prior) reads `self_intro` as "the voice said it", which is not always true.

Worked example: Conversations with Tyler, "Julia Ioffe on…", gold dev c087.

1. The rules named the 2,735 s voice "Julia Ioffe" from her own self-introduction. This is correct.
2. The LLM then added "Julia Ioffe" as HOST on Tyler's 380 s voice. The show states no host.
3. Corroboration had already dropped her from the guest list, because "X on …" in a title is not an
   accepted cue.

The old sidecar showed none of this chain.

## Phase 1: what is recorded (code as of this doc version)

### Mechanism

`resolve_speaker_roster(..., trace: Optional[NamingTrace] = None)`. The helpers that make the
decisions take the same optional `trace`; `resolve_speaker_roster` passes its own.

- **Observer only.** Recorders are plain calls on a `NamingTrace`; with none given, a `NullTrace`
  (`enabled = False`) records nothing, so the ladder carries no `if trace` branches.
  `_select_host_voices` sits at the complexity cap (mccabe 25), so its recorders are straight-line
  calls only.
- **One decision-code refactor**, in `_name_host_voices`: the forced-name condition was one `and`
  chain; it is now named booleans (`one_name_one_seat`, `seat_owns_talk`, `presenter_elsewhere`,
  `addressed`) evaluated in the same order with the same short-circuits, so the trace can say which
  gate held. Gated by the old-vs-new replay below.
- **Failure isolation.** A recorder's exception is swallowed and sets `degraded: true`;
  `to_dict()` never raises and returns a JSON-safe copy (sets → sorted lists); an unserialisable
  trace (NaN) becomes `{version, degraded: true, error}` so the sidecar is still written. NOT
  guarded: arguments the roster builds at a call site (`sorted(...)`, `dict(...)`).
- **The talkative-host retry runs untraced** (`_name_host_voices` a second time, one seat's answer
  taken); its `talkative_host` diff records the result.
- **Pipeline.** Traces the SHIPPED pass. Adds `inputs.baseline_without_llm` (rules-only roster per
  voice), `non_regression` / `restored` steps, `inputs.llm_resolution` and `inputs.detection`.
  Written as `speaker_diagnostics.decision_trace`.
- **Offline replay.** `scripts/measure/roster_replay.py --trace-out PATH` replays every stored
  episode plain and traced; summary counts `traced_roster_differs`, `traced_only_error`,
  `trace_degraded`, `trace_bytes_max`; exits 1 if any of the first three is non-zero; refuses a
  roster variant without a `trace` parameter.

### The two LLM answers, in full

- **`inputs.detection`** — ingest-time speaker detection, recorded on the `Episode`
  (`speaker_detection_report`) by `_detect_speakers_for_episode` and carried to every
  `apply_diarization_to_result` call site: `detector`, `raw` (the model's answer; each LLM
  provider's parser keeps its last one, cleared before every call; capped at 8k chars, `raw_chars`
  keeps the length), `returned` (speakers / hosts / succeeded), `dropped_placeholders`,
  `dropped_non_persons`, `hosts_from_description`, `hosts`, `proposed_guests`,
  `corroborated_guests`, `corroboration_rejected` (`{name, reason}`, reason `is_a_host` /
  `no_interview_cue`), and `outcome` / `reason` (a skipped detection says so). `null` when the call
  site had no record (e.g. a reprocess that skipped detection).
- **`inputs.llm_resolution`** — the post-diarization resolver: `provider`, `stated_names`,
  `prompt_chars`, `raw` (capped at 16k, `raw_chars` uncut), `verdicts` (per voice the model named:
  `said_voice`, mapped `voice`, `name`, `role`, `matched` stated name, `outcome` = `accepted` /
  `invented` / `third_person` / `duplicate` / `unmapped_voice` / `role_only` / `abstained`),
  `complement` (`two_voice` / `swap`), `error`, or `skipped` with the reason it was never asked.

### Step semantics

- `name` is the name a step SET (or refused at the publish gate / intro reader); a proposal not
  taken is `proposed`.
- **Set a name:** `named` / `renamed` (name diffs), `set` / `changed` (roster diffs; `changed`
  carries `previous`, `cleared` lists fields that became null), `accepted` (LLM merge),
  `named_from_earlier_rung` (host or guest naming kept a name an earlier rung put on the voice —
  that rung's own step says which), `forced_pool_name`, `forced_name`, `prefix_stripped`, `restored`.
- **Other:** `seated`, `stated`, `heard`, `typed`, `unnamed`, `forced_name_vetoes`,
  `bare_first_name_joins`; did-not-take: `skipped`, `refused`, `refused_spelling`, `excluded`,
  `never_names`, `removed`, `dropped`.

### Rungs recorded, in ladder order

| Rung (trace key) | Recorded as | What it captures |
| --- | --- | --- |
| `inputs.detection`, `inputs.llm_resolution` | input | both LLM answers in full (above) |
| `inputs.*` | input | `known_hosts`, `detected_guests`, `metadata_named`, `stated_voice_names`, `llm_voice_names`, `llm_voice_roles`, `ad_voices`, `conversation_roles`, `host_pool` |
| `guest_host_pool` | episode note | a guest host the episode text added to the pool |
| `publisher_label` | per voice `stated`, then name diff | every person label the publisher states; where it replaced a name |
| `self_intro`, `show_mononym_filter` | name diff | names voices gave themselves; show-mononyms removed |
| `intro_reader` | per voice `heard` (`heard`, `spelling`) / `skipped` (`ad_voice` / `named_itself`) / `refused_spelling` (`name`, `resembles`), plus name diff | every introduction read, what it became, what was refused and whom it resembled |
| `llm_merge` | per voice `accepted` (`llm_said`) / `skipped` (`proposed`, `reason`, `kept`) | every surviving LLM name and what happened to it |
| `cohost_formula` | name diff | names from the co-host formula |
| `presenter_evidence` | episode note | branded / introducer / copresenter voices; conversation guests |
| `host_pool` | input, plus per name `never_names` | the pool with each name's source |
| `host_seat_guards` | episode note | every veto set the seat steps consult, and the cap |
| `host_seat_step` | per voice `seated` (`step`) | WHICH step seated it: `1_named_as_a_stated_host`, `1b_presents_on_own_evidence`, `1c_unnamed_introducer`, `2_performs_host_role`, `3_opener`, `4_count_fill`, `5_llm_host_no_pool`, `6_stand_in_interviewer` |
| `host_seat_opener`, `host_seat_step_4` | episode notes | the opener; step 4's arithmetic (empty seats, unclaimed / said-absent hosts, fillable, candidates, guest present, third-party excess, co-host said present) |
| `host_seats` / `host_seat` | episode note, per voice `order` | final seats in order |
| `host_naming_forced_gates` | episode note | spare names before/after the feed-history and said-absent filters, unnamed seats, guest-hosted episode, one-name-one-seat, seat owns the talk, better presenter elsewhere |
| `host_naming_pair` | episode note (`performs_guest_act`, `named`) | two unnamed seats and two unclaimed pool names, each seat addressing the other's host by name (`_pair_by_address`, The Rest Is History) |
| `host_naming` | per seat `named_from_earlier_rung` / `forced_pool_name` (`by: address_pair` when the pair rule named it) / `unnamed` (`intro_name_taken`), plus `forced_name_vetoes` (guest act, greeted by that name, rescued from bleed, forced) | how each seat was named or why not |
| `host_introduction_harvest`, `two_voice_interview` | per name `added` / `refused` | guest names harvested from the host's introduction |
| `guest_pool` | per name `excluded` (reason), episode note | the guest pool and why declared names left it |
| `guest_naming_forced` | episode note | spare names, candidates above the cameo floor, forced name, `forced_by` = `one_name_one_voice` / `only_voice_left_below_cameo_floor` / `dominant_unassigned_voice` / `host_elimination`, forced voice |
| `guest_naming` | per voice `named_from_earlier_rung` / `forced_name` (`forced_by`) / `unnamed` (`role`, `intro_name_refused`, `forced_name_refused`, `role_evidence`) | how each remaining voice was named or why not |
| `talkative_host`, `ad_voice_placeholder`, `stated_spelling_snap`, `recover_stated_names` | role diff | changes only |
| `publish_gate` | per voice `prefix_stripped` / `refused`, per name `refused` | the final plausibility gate |
| `one_name_per_person` | role diff, per voice `bare_first_name_joins`, episode note per group (`unified`: names, roles, kept name + `kept_because`, role + `role_reason`; or `kept_apart`) | one person, one name, one role |
| `nameable` | episode note | nameable voices, leftover / unbound names |
| `voice_types` | per voice `typed` (`voice_type`, `reason` = `edge_ad` (+`name_demoted`) / `named` / `mostly_inside_ads` / `brief` / `no_source_names_them` / `a_name_existed_and_we_failed`, `talk_s`, `classified_by`) | every voice's type and why |
| `non_regression`, `inputs.baseline_without_llm` | pipeline | names the additive contract restored; the rules-only roster |

**Sources a voice's name can carry** (`voices[].source`). `_NAME_EVIDENCE_RANK` ranks four of them,
used when one person's voices disagree: `self_intro`, `introduced`, `publisher_transcript` = 3,
`llm_resolution` = 2.

- `publisher_transcript` — the publisher's own speaker label.
- `llm_resolution` — the post-diarization LLM's closed-list match.
- `introduced` — the HOST's words, read by the introduction reader (`93f49482c`). Counts as
  host-copresence evidence, not as the voice's own words.
- `self_intro` — every other name in the intro map: the voice's own introduction (opening,
  sign-off, or a weak one the metadata vouches for or anchors) and the co-host formula's names.
- A host seat named from the pool keeps the pool entry's source (`feed`, `known_hosts`, …).
- `forced` — placed by arithmetic. `raw` — unnamed.

### Same-person rules, and the rungs they act at

One person is one entry (2026-10-08, `c05cc0273` and its follow-up). Two predicates, by scope:

- **Across sources** — `_same_person`: same surname, and a matching given name or initial (or a
  title + surname). A missing generation ("Jr.") is not a disagreement.
- **Within one episode** — `_same_person_on_one_episode`: titles stripped, whole-name similarity
  >= 0.91, or one differing token with similarity >= 0.70 (`_TOKEN_RESPELLING_SIMILARITY`); a
  generation difference IS a different person.

Where each acts:

| Rung | Rule |
| --- | --- |
| `self_intro`, `host_seat_guards` (`stated_non_host`) | A leading title is not the given name (`_core_name_tokens` drops it while a given name and a surname remain). `_snap_near_identical_host` maps a self-introduction to the stated host when they differ only by a title, by one letter (5+ letter surname), or by two adjacent swapped letters ("Wiesenthal" / "Weisenthal"). A voice that says "I'm Professor Hannah Fry" on Hannah Fry's show is therefore not a stated non-host. |
| `recover_stated_names` | A guest never takes a stated HOST's spelling (N1, "Kevin Ross" / "Kevin Roose") — unless the episode also names that person as a participant (`detected_guests` / `metadata_named`): then the host pool is the polluted source ("Lucas Kaiser" -> Lukasz Kaiser, guest, The a16z Show). When the fuzzy canonicaliser finds nothing, `_stated_participant_spelling` gives the voice the ONE stated participant it is a respelling of by the one-episode rule, if no other voice holds that name ("Scholastic Gatobu" -> "Schola Gatobu", "Charming Lai" -> "Chiamin Lai"). A name the episode already states is never re-spelt. |
| `guest_pool` (`same_person_as_a_named_voice`) | A declared guest name is not spare when it is the same person as a voice's name by `_same_spoken_person` OR the one-episode rule ("Andy Ratcliffe" beside "Andy Rachleff", The Long Run). |
| `one_name_per_person` | Voices grouped by the one-episode rule take one name and one role. |
| record (`metadata_generation._unplaced_speakers`) | An unplaced entry is dropped when it is the same person as a placed or earlier entry by EITHER predicate. Not a roster rung: not in the trace. |

## What is NOT recorded yet

Do not read an absence in a trace as "did not happen".

- **"Considered and declined" for diff-recorded rungs** (`talkative_host`, `stated_spelling_snap`,
  `recover_stated_names`, `ad_voice_placeholder`): no step when nothing changed.
- **Host-seat vetoes per voice.** The veto SETS are recorded once (`host_seat_guards`); which veto
  stopped a given voice at a given step is derivable from them, not recorded per voice. Step 1b's
  presence test (share / span / co-presenter partner) is not recorded.
- **Inside `_voice_named_by_the_introduction`** (complexity 22): its per-cue rules (narrator cue,
  first-name-only, report-verb corroborated resolution) — only what it returned (`heard`).
- **`_guest_voice_by_host_elimination`**: only whether it picked (`forced_by=host_elimination`),
  not why it declined.
- **LLM name reuse** is not flagged; it is visible as an `llm_merge` `accepted` step on a second
  voice plus the first voice's earlier-rung step.
- **Host-pool sub-source** beyond the single `source` tag (feed statement / author tag /
  description / config / recurrence).
- **Detection on non-LLM detectors** (spaCy) has no `raw`; `returned` is its answer.
- **`post_hoc`.** Migrations m0012–m0020 never write the diagnostics sidecar. m0022 and m0025 make
  `voices[]` follow the repaired segments (name, `named`, `source: raw` when unnamed; m0025 also
  `role`), but no migration writes `decision_trace`: a migrated episode's trace describes the
  roster as ingested, not as published. m0025's own record is its receipts
  (`one_person_one_entry.jsonl`: each episode's renames, voice-by-voice swaps, unnamings, dropped
  entries). `scripts/measure/m0025_parity.py` checks, on a copy, that m0025 publishes per voice
  what the fixed roster publishes wherever the stored record is today's code's product.
- **The record's de-duplication** (`_unplaced_speakers`, above) runs in metadata generation, after
  the roster; the trace does not show which unplaced name it dropped as a twin.
- **`source` is less lossy than before**: the introduction reader's names are `introduced`
  (`93f49482c`), but `self_intro` still mixes a voice's own words with metadata-vouched weak
  intros and the co-host formula (see "Sources" above).
- **Offline replays stay lossy for the LLM rung**: `roster_replay` rebuilds LLM inputs only from
  voices whose final `source` is `llm_resolution`. Full LLM answers exist only in traces written
  at ingest (after the deploy).

## Gates and evidence

All runs 2026-10-04. The prod runs are read-only, inside `compose-api-1`, over `/app/output`, with
staged files whose md5s matched the committed ones.

**1. Behaviour gate — the helpers' instrumentation and the `_name_host_voices` refactor change no
decision.** Old roster (`6117aedb5`) vs new (`c8bda243d`, pre-rebase `095a2954d`), every field of every voice (name, role,
named, source, voice_type), same inputs (`--repool --signatures` equivalents):

```text
run_equal.py /app/output   ->  {"episodes": 2319}   EXIT=0   (differ 0; old_error 0; new_error 0)
```

`11e75c7ab` after it renames one step label (string only), covered by the unit tests.

**2. Purity gate — a trace never changes the roster.** New code, traced vs untraced:

```text
run_trace_replay.py --corpus /app/output --repool --signatures --new roster=... --trace-out ...
  -> {"summary": {"episodes": 2319, "trace_bytes_max": 12181}}   EXIT=0   tracebacks 0
```

`traced_roster_differs`, `traced_only_error`, `trace_degraded`, `replay_error` all 0 (absent).
Also, locally, every `resolve_speaker_roster` call in the unit suite (184) run traced and untraced
through a pytest plugin: 0 differ, 0 degraded.

**3. Completeness.** In those 2,319 traces: 4,171 voices published by name, every one with a step
that SET exactly that name (0 without). Trace JSON: p50 2,876 B, p99 7,245 B, max 12,181 B.

**4. End to end with the real LLM.** Laptop, worktree code at `c8bda243d` (pre-rebase `095a2954d`), profile
`prod_dgx_full` (DGX Whisper, pyannote, vLLM Qwen3-30B), No Priors, newest episode:
`EXIT=0`; the written `.speakers.diagnostics.json` holds `decision_trace` v1, `degraded: false`,
with `inputs.detection` (`detector: VLLMProvider`, the model's raw JSON, corroboration result) and
`inputs.llm_resolution` (raw answer; two verdicts, one written `SPEAKER_1` and mapped to
`SPEAKER_01`), seat steps, forced-name gates and voice types; 4.6 KB.
A 5-episode run of one feed (`prod_dgx_full`) was still running at the cut-off push; its result is added when it lands.

**5. Tests.** `test_naming_trace.py` (15), `test_naming_trace_rungs.py` (32: one per rung, mostly
reusing an existing roster test's scenario with a trace injected), `test_llm_answers_are_recorded.py`
(13). Mutation-checked: deleting any of 12 sampled recorders fails a test. Full unit suite with
`PYTHONPATH=src`: 13,139 passed, 55 skipped.

**What the gates do NOT prove.**

- That the decisions are RIGHT: gates 1–2 compare code with code.
- Prod sidecars: nothing is deployed; the first ingest after the deploy is the first prod check.
- Order-dependent unit flakes seen three times in this work (`test_run_manifest`,
  `test_summary_poison_guard`, `test_metadata_generation::test_generate_metadata_dry_run`): each
  failed once inside the parallel full suite and passed alone and on re-run. Not investigated to
  cause.
- `test_cli_profile_subprocess` and two other subprocess tests fail in this worktree unless
  `PYTHONPATH=src` is set (the borrowed venv's editable install points at a removed worktree);
  with it they pass.

## Phases

1. **Trace (roster, helpers, both LLM answers) + `roster_replay --trace-out`.** Code on main;
   deployed: no. Remaining gaps are listed above.
2. **Analysis on existing data.** Per-rung precision against the gold labels; failure chains per
   show across ~2,300 prod episodes. Runs on `roster_replay --trace-out` now; the LLM rung needs
   ingest-time traces.
3. **Ingest-side records** — detection and the resolver are now done (pulled into phase 1); left:
   host-pool sub-source, `post_hoc` migration records.
4. **Then priority and ordering changes**, each gated on dev + validation with the trace explaining
   every voice that moved. Candidates already visible: title/description guest phrases ("X on …",
   "— X", "with X") as corroboration; an LLM name may not take a second voice once the episode bound
   it by self-introduction; an LLM-inferred host name should not count as step 1's "named as a
   stated host" the way a spoken self-introduction does (seen on the first DGX run); the rest of
   the `self_intro` source split (the reader's half is done, `93f49482c`).
