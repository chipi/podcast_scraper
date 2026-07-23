# roster-golden fixture schema (v4 Phase 0 — #1189)

Golden diarization fixtures for the speaker roster (`resolve_speaker_roster`,
`src/podcast_scraper/providers/ml/diarization/roster.py`). Tracked by #1189;
sequencing lives in `docs/wip/1000-EPISODES-REPROCESS-PLAN.md`; the case
catalogue and rationale live in `docs/wip/CORPUS-V4-FIXTURE-LADDER.md` (§G,
"The anatomy of a real episode").

## The one rule

The runner (`tests/unit/podcast_scraper/providers/ml/diarization/
test_roster_golden_fixtures.py`) DRIVES `resolve_speaker_roster`. It never
re-implements any part of the roster's decision logic. A fixture is inputs in,
real function call, assert on the real output. This is the §F0 lesson: a
diagnostic that restates the rule in its own words measures its own words, not
the shipped code.

## Friendly shorthand vs. the dataclass — read this before writing a fixture

The fixture-ladder spec (`CORPUS-V4-FIXTURE-LADDER.md` §G) writes examples like:

```yaml
expected_roster:
  SPEAKER_04: {role: advertisement, named: false}
```

**`role: advertisement` does not exist.** `SpeakerRole.role` is one of exactly
`"host" | "guest" | "unknown"` (see `roster.py:762` docstring and
`SpeakerRole` dataclass). An advertisement is a `voice_type`, not a role:
`SpeakerRole.voice_type` is one of `"person" | "cameo" | "commercial" |
"unidentified" | "unknown"` (constants `VOICE_*`, `roster.py:121-142`). This
schema uses the real fields; the spec's shorthand is translated below.

### Translation table (spec shorthand → real fields)

| spec says | this fixture's `expected_roster` entry |
| --- | --- |
| `role: advertisement` | `role: unknown, voice_type: commercial` (an ad narrator is never a host or a guest, and never named — see `_classify_voice_types`, roster.py) |
| `role: host` (named) | `role: host, voice_type: person, named: true, name: "<Name>"` |
| `role: host` (unnamed, show-centric) | `role: host, voice_type: person, named: false, name: "<raw voice id>"` |
| `role: guest` (named) | `role: guest, voice_type: person, named: true, name: "<Name>"` |
| a brief interjection (<20s) | `voice_type: cameo` |
| substantive, nobody ever names it | `voice_type: unidentified` |
| substantive, a name existed and we failed to use it | `voice_type: unknown` (the raw `SPEAKER_NN` id IS the defect marker — §E2) |

Every fixture's `expected_roster` block MUST use only the real field values
above — never the spec's shorthand directly.

## Fixture shape

```yaml
id: <string>                    # unique fixture id, matches the filename stem
source: reconstructed_from_spec | frozen_real_run   # provenance (see below)
show: <string>                  # show slug, for grouping

# §G metadata half of the contract — the WHO/HOW MANY half, read from the feed.
# Phase 0 does not feed this text through detect_hosts_from_feed (see "Out of
# scope" below); it is carried for documentation and for the future
# integration-tier harness that will exercise that stack.
feed:
  title: <string>
  description: <string>
  authors: [<string>, ...]
expected_hosts_from_feed: [<string>, ...]     # documentation only, Phase 0

episode:
  description: <string>
expected_guests_after_gate: [<string>, ...]   # documentation only, Phase 0

# The ACTUAL diarization turns (start_s, end_s, speaker). Real pyannote output
# when source: frozen_real_run; reconstructed timing/proportions from the
# documented episode structure when source: reconstructed_from_spec.
diarization:
  - {speaker: SPEAKER_04, start: 0.0, end: 15.0}
  - ...

# Each voice's OWN words, concatenated across its turns. This is what
# resolve_speaker_roster's ``voice_texts`` parameter receives directly.
voice_texts:
  SPEAKER_04: "I'm Paul Tenorio. I cover soccer for The Athletic."
  ...

# The PRE-RESOLVED inputs Phase 0 hands to resolve_speaker_roster. These
# stand in for what upstream detection (feed NER, corroboration, ad-region
# detection) would have produced; Phase 0 does not run that upstream code —
# see "Out of scope" below.
resolver_inputs:
  host_candidates: [<string>, ...]     # feed-author-derived host candidates
  known_hosts: [<string>, ...]         # trusted/confirmed host names (config or feed-stated)
  detected_guests: [<string>, ...]     # corroborated guest names
  metadata_named: [<string>, ...]      # every person the episode metadata stated, pre-corroboration
  ad_intervals: [[<start_s>, <end_s>], ...]   # explicit ad regions (sponsor-pattern hits)

# The VERIFIED real output of resolve_speaker_roster on the inputs above.
# Every value here MUST come from an actual pytest run against the shipped
# roster.py — never hand-derived. See "Honesty workflow" below.
expected_roster:
  SPEAKER_04: {role: unknown, voice_type: commercial, name: SPEAKER_04, named: false}
  SPEAKER_07: {role: host, voice_type: person, named: true, name: "Kevin Roose"}
  ...
```

`transcript_text` and `ordered_turns` are optional fixture fields
(`resolve_speaker_roster` accepts both) — omitted here because the seed case
does not need them (guest naming is resolved via the forced-match path, not
the host-introduces-guest-by-name path). A future fixture that needs to
exercise `_voice_named_by_the_introduction` should add `ordered_turns` as a
list of `[speaker, text]` pairs in chronological order.

## Honesty workflow — non-negotiable

1. Write the diarization turns and `voice_texts` from the documented episode
   structure (or a frozen real run).
2. Build the fixture's `resolver_inputs`.
3. Call `resolve_speaker_roster` with those inputs in a throwaway script (or a
   temporary `print(roster.by_voice)` in the test), and RUN it.
4. Copy the ACTUAL output into `expected_roster`. Never hand-derive it.
5. If the real output contradicts the spec's stated intent (an ad voice
   gets a role other than `unknown`, a host's name lands on the wrong voice,
   a name appears that has no source) — that is a FINDING, not a fixture bug
   to paper over. Do not encode the wrong value as "expected." Keep the case
   (or the specific perturbation) out of the asserted/green suite and report
   it for triage. See `test_roster_golden_fixtures.py`'s per-perturbation
   PENDING notes for the standing findings from this seed.

## Out of scope for Phase 0 (documented, not silent)

`resolve_speaker_roster` is the back half of speaker resolution — it resolves
a WHICH-VOICE question given already-detected names. The front half —
`detect_hosts_from_feed` (feed-text NER), `corroborate_guests` (the
description-vs-transcript gate), and the LLM voice-resolution path
(`resolve_voices_from_conversation`) — all require spaCy (and, for the LLM
path, a completion provider) and are NOT driven by this harness.

This keeps `test_roster_golden_fixtures.py` a `tests/unit` test with only
`[dev]` dependencies (PyYAML), matching the 3-tier test policy (`U1`: no
`importorskip` gymnastics to pull in `[ml]` for a unit test).

Driving the full stack — feed text in, roster out, through
`detect_hosts_from_feed`/`corroborate_guests` — is a documented FOLLOW-ON:
an integration-tier harness (`tests/integration/...`, `[ml]`-gated) that
exercises `feed.description` / `episode.description` for real instead of
treating `expected_hosts_from_feed` / `expected_guests_after_gate` as
documentation-only fields. Until that harness exists, those two fields are
carried in every fixture for the humans reading it and for that future
harness to consume — they are not asserted by `test_roster_golden_fixtures.py`.

## Perturbations

A perturbation is a one-line transform of a fixture dict (see the
`_perturbations` mapping in `test_roster_golden_fixtures.py`). Each is
verified empirically against the real roster before being wired up:

- Passes green → parametrized into the asserted perturbation suite.
- Fails → left defined (the transform exists and can be inspected/re-run) but
  NOT parametrized into the asserted suite, with a `# PENDING:` comment
  stating exactly what the roster does vs. what the spec expects, and a
  reference to the relevant CORPUS-V4-FIXTURE-LADDER.md case number or issue.
  Never `xfail`ed — an `xfail` still runs the assertion and a suite full of
  expected failures stops being read (§F, "a red suite is not a nuisance").
