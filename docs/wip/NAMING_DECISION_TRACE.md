# Naming decision trace — reconstruct the speaker-naming ladder for any episode

| | |
|---|---|
| Doc version | 2 |
| Last updated | 2026-10-04 |
| Trace schema | `TRACE_VERSION = 1` (`src/podcast_scraper/providers/ml/diarization/naming_trace.py`) |
| Implemented by | `448151fab` (phase 1, #2276) |
| Code content hash | `8a795c303601` (see below) |
| Status | Phase 1 shipped to main, **not yet deployed**: prod sidecars get `decision_trace` only after the next deploy. `roster_replay --trace-out` works offline today |

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

`resolve_speaker_roster(..., trace: Optional[NamingTrace] = None)`.

- **Observer only.** The roster calls recorders on a `NamingTrace`. With no trace it calls a
  `NullTrace` (`enabled = False`), so the roster code carries plain calls and no `if trace`
  branches.
- **Failure isolation.** Every recorder is wrapped:
  - an exception inside a recorder is swallowed and sets `degraded: true`;
  - it never propagates into the roster;
  - `to_dict()` never raises, and returns a JSON-safe **copy** (sets become sorted lists).
  - A trace that still cannot be serialised (for example NaN) becomes
    `{version, degraded: true, error}`, so the sidecar is still written.
  - What is NOT guarded: the arguments the roster builds at a call site (`sorted(...)`, `dict(...)`)
    are evaluated by the roster itself. They use the same operations the roster already performs on
    the same values.
- **Pipeline.** `pipeline.py` traces the **shipped** pass only. It adds to the trace:
  - the rules-only baseline roster per voice, as `inputs.baseline_without_llm`;
  - every voice `_reconcile_non_regression` put back, as a `non_regression` / `restored` step.

  The result is written as `speaker_diagnostics.decision_trace`. The baseline pass itself is
  untraced.
- **Offline replay.** `scripts/measure/roster_replay.py --trace-out PATH` replays every stored
  episode twice, once plain and once traced, and writes JSONL:
  `{meta_path, feed, episode, roster, decision_trace}`. Its summary counts:
  - `traced_roster_differs`;
  - `traced_only_error` (the traced pass raised where the plain pass did not);
  - `trace_degraded`;
  - `trace_bytes_max`.

  It exits 1 if any of the first three is non-zero. It refuses to run against a roster variant
  without a `trace` parameter, because that gate would pass vacuously.

### Sidecar shape (as emitted)

```json
"decision_trace": {
  "version": 1,
  "degraded": false,
  "inputs": {
    "known_hosts": ["…"], "detected_guests": ["…"], "metadata_named": ["…"],
    "stated_voice_names": {"SPEAKER_00": "…"},
    "llm_voice_names": {"SPEAKER_01": "…"}, "llm_voice_roles": {"SPEAKER_01": "guest"},
    "ad_voices": ["SPEAKER_04"], "conversation_roles": {"SPEAKER_00": "host"},
    "host_pool": [["Name", "source"]],
    "baseline_without_llm": {"SPEAKER_00": {"name": "…", "role": "…", "named": true, "source": "…"}}
  },
  "voices": {
    "SPEAKER_00": [
      {"rung": "self_intro", "decision": "named", "name": "Julia Ioffe"},
      {"rung": "llm_merge", "decision": "skipped", "proposed": "…", "reason": "already_named", "kept": "Julia Ioffe"},
      {"rung": "host_seat", "decision": "seated", "order": 0},
      {"rung": "host_naming", "decision": "set", "name": "…", "role": "host", "named": true, "source": "…"},
      {"rung": "voice_types", "decision": "changed", "voice_type": "person", "previous": {"voice_type": null}}
    ]
  },
  "names": {
    "Before Gene": [{"rung": "publish_gate", "decision": "refused", "reason": "not_publishable", "voice": "SPEAKER_02"}]
  },
  "episode": [
    {"rung": "presenter_evidence", "branded": [], "introducers": [], "copresenters": [], "conversation_guests": []},
    {"rung": "host_seats", "voices": ["SPEAKER_00"], "cohost_said_present": false},
    {"rung": "guest_pool", "declared": ["…"], "guest_names": ["…"]},
    {"rung": "nameable", "voices": [], "leftover_guest_names": [], "stated_unbound": [], "spare_name_count": 0}
  ]
}
```

### Step semantics

- `name` is always the name the step **set** (or refused, at the publish gate).
- A proposal the step did not take is recorded as `proposed`, never as `name`.
- **Set-a-name decisions:**
  - `named`, `renamed`, `removed` come from voice-intro name diffs;
  - `set`, `changed`, `dropped` come from roster-entry diffs;
  - `changed` carries `previous` holding the old value of every changed field;
  - `cleared` lists the fields that became null;
  - the others are `accepted`, `seated` and `restored`.
- `stated`: a source states this name. It is recorded even when it changes nothing.
- **Did-not-take decisions:** `skipped`, `refused`, `refused_spelling`, `excluded`, `never_names`.

### Rungs recorded, in ladder order

| Rung (trace key) | Recorded as | What it captures |
|---|---|---|
| `inputs.*` | input | `known_hosts`, `detected_guests`, `metadata_named`, `stated_voice_names`, `llm_voice_names`, `llm_voice_roles`, as received |
| `guest_host_pool` | episode note | names the episode text added to the host pool (a guest host) |
| `inputs.ad_voices`, `inputs.conversation_roles` | input | the edge-ad voices; the speech-act roles per voice |
| `publisher_label` | per voice `stated` | every person label the publisher states (after the spelling snap) |
| `self_intro` | name diff | names the voices gave themselves |
| `show_mononym_filter` | name diff | self-intro names removed because they are the show's mononym |
| `publisher_label` | name diff | where the publisher label replaced or added a name |
| `intro_reader` | name diff, plus per voice `refused_spelling` | names read from a host's introduction; refused when the introduced person is unstated |
| `llm_merge` | per voice `accepted` (`llm_said`) / `skipped` (`proposed`, `reason` = `names_the_show` / `ad_voice` / `already_named`, `kept`) | every LLM name and what happened to it |
| `cohost_formula` | name diff | names from the co-host formula |
| `presenter_evidence` | episode note | branded / introducer / copresenter voices; conversation guests after presenters are removed |
| `host_pool` | input, plus per name `never_names` | the pool with each name's source; entries that may seat a voice but never name it |
| `host_seats` / `host_seat` | episode note, plus per voice `seated` (`order`) | which voices took host seats, in order; whether a co-host was said to be present |
| `host_naming` | role diff against `{}` | every seat's resulting entry |
| `host_introduction_harvest` | per name `added` (`heard`) / `refused` (`resembles_an_unbound_stated_name`) | guest names harvested from the host's introduction |
| `two_voice_interview` | per name `added` | the one stated guest a two-voice host hands over to |
| `guest_pool` | per name `excluded` (`host_pool_name` / `already_named_on_a_voice` / `same_person_as_a_named_voice`), plus episode note | the guest pool and why declared names left it |
| `guest_naming` | role diff | guest names placed |
| `talkative_host` | role diff | a host named once the guests are placed |
| `ad_voice_placeholder` | role diff | ad voices given their placeholder |
| `stated_spelling_snap` | role diff | hosts snapped to the episode's stated spelling |
| `recover_stated_names` | role diff | stated names recovered onto unnamed voices |
| `publish_gate` | per voice `prefix_stripped` (`previous`) / `refused` (`not_publishable` / `show_mononym`), plus per name `refused` | the final plausibility gate |
| `one_name_per_person` | role diff | duplicate-person resolution |
| `nameable` | episode note | nameable voices, leftover guest names, stated-but-unbound names, spare name count |
| `voice_types` | role diff | voice-type classification and edge-ad demotion |
| `non_regression` | per voice `restored` (pipeline) | names the additive contract put back over the LLM pass |
| `inputs.baseline_without_llm` | input (pipeline) | the rules-only roster per voice |

## Phase 1: what is NOT recorded yet

These are known gaps. Do not read their absence from a trace as "did not happen".

**Inside the roster (phase 1 follow-ups, replayable):**

- **Host seats, steps 1 / 1b / 1c / 2–6.** Only the final seats and their order are recorded. Not
  recorded: which step seated each voice, and which veto refused a voice.
- **Host naming (`_name_host_voices`).** Not recorded: intro vs forced pool name, which veto blocked
  a forced name, and the ownership guard. Only the end state and the coarse `source` are recorded.
- **Guest naming (`_name_guest_voices`).** Not recorded: which variant fired (forced / cameo floor /
  dominant / host elimination), the third-person guard, and the refused-intro guards. Only the
  end-state diff is recorded.
- **Intro reader.** Only the name diff and `refused_spelling` are recorded. Not recorded: per-rung
  proposals it rejected, and the report-verb corroborated resolution.
- **One name per person.** Only the diff is recorded. Not recorded: the groups, the losers, and the
  role unification.
- **Voice types.** Only the diff is recorded. Not recorded: why each voice got its type.
- **No "considered and declined" for diff-recorded stages.** A diff-based rung (`talkative_host`,
  `stated_spelling_snap`, `recover_stated_names`, …) leaves no step when it changes nothing. So an
  absent step means "no change", never "not eligible".
- **LLM role verdicts.** `llm_voice_roles` appears only in `inputs`. There is no per-voice step for
  a role the LLM changed.
- **LLM name reuse.** The worked example (an LLM name on a second voice after the episode bound it
  by self-intro) shows up only as an `accepted` step on the second voice. Reuse has to be detected
  in analysis by joining against other voices' `self_intro` steps. The trace has no explicit flag.

**Ingest side (phase 3, forward only, not replayable):**

- the raw `detect_speakers` answer;
- names dropped by the placeholder and organisation filters;
- corroboration rejects with a reason (for example `no_interview_cue`; today this is only a log
  line);
- the host-pool sub-source beyond the single `source` tag (feed statement, author tag, description,
  config, recurrence);
- the LLM resolution's raw per-voice verdict;
- names the LLM's closed list, third-person rule or one-per-voice rule refused;
- whether the two-voice complement or swap fired.

**After ingest:**

- **`post_hoc`.** Migrations m0012–m0020 rewrite metadata, segments and graphs but never the
  diagnostics sidecar. A migrated episode's trace can disagree with its published roster. Planned:
  migrations that touch a speaker surface append `{migration, change}` to
  `decision_trace.post_hoc`.

**Not changed by phase 1:**

- `source` is still lossy. Only the trace's rung name tells `self_intro` apart from `intro_reader`
  and `cohost_formula`. The `source` split is a behaviour change, gated in phase 4.

**Not measured:**

- Sidecar size on prod. The replay summary's `trace_bytes_max` bounds the trace's JSON size, but
  not the `indent=2` file growth.

## Phase 1 gate result

**Run:** 2026-10-04, 09:14–09:52 UTC, on prod (`compose-api-1`, read-only, `/app/output`).
The code was staged into `/tmp/rp9`, with md5s matching the committed files (content hash
`8a795c303601`).

```sh
python run_trace_replay.py --corpus /app/output --repool --signatures \
  --new roster=/tmp/rp9/roster_new.py --trace-out /tmp/rp9/traces.jsonl
# {"summary": {"episodes": 2311, "trace_bytes_max": 9652}}   EXIT=0   tracebacks: 0
```

**Gate counters.** The summary prints only non-zero counters. All of these were absent, so 0:

| Counter | Value |
|---|---|
| `traced_roster_differs` | 0 |
| `traced_only_error` | 0 |
| `trace_degraded` | 0 |
| `replay_error` | 0 |

**Completeness check over the 2,311 traces:**

- 4,162 voices were published by name.
- Every one of them (0 without) has a step that SET exactly that name. A set step is `named`,
  `renamed`, `set`, `changed`, `accepted`, `added`, `restored` or `prefix_stripped`.
- Trace JSON size: p50 1,773 B, p99 6,084 B, max 9,652 B.

**What this gate does NOT prove:**

- **It compares the traced roster with the untraced roster of the SAME code.** It proves the
  recorder changes no decision. It says nothing about whether the decisions are right.
- **The replay runs `resolve_speaker_roster` only.** The pipeline-side additions are not exercised
  by the replay: `baseline_without_llm`, the `non_regression` step and the sidecar write. They are
  covered only by the unit suite (`pytest tests/unit`: 13094 passed, 55 skipped).
- **No prod sidecar has a `decision_trace` yet.** That needs a deploy. The first ingest after it is
  the first end-to-end check.
- **The replay's LLM inputs are lossy.** The sidecar does not store the LLM's raw answer.
  `roster_replay` rebuilds `llm_voice_names` / `llm_voice_roles` only from voices whose FINAL
  `source` is `llm_resolution`. An LLM proposal that lost (skipped as `already_named`, refused at
  the publish gate, overridden later) is not in a replayed trace. Offline traces therefore
  undercount `llm_merge` `skipped` steps and LLM-proposed junk. Phase-2 analysis of the LLM rung
  needs ingest-time traces (after the deploy), or the phase-3 raw-verdict record.

## Phases

1. **Roster trace + `roster_replay --trace-out`.** Done. See the gate result above.
   - Gate: the traced roster equals the untraced roster on every replayed episode, with no
     traced-only errors and no degraded traces.
   - Unit tests: `test_naming_trace.py` (14).
2. **Analysis on existing data.** Per-rung precision against the gold labels: how often each rung's
   decision matches the label, and how often it overrode an earlier rung that was right. Failure
   chains across ~2,300 prod episodes, per show. This sets the priority order. It runs on
   `roster_replay --trace-out`; no deploy is needed.
3. **Ingest-side records** (see "NOT recorded"). Forward only; ships with a deploy.
4. **Then priority and ordering changes.** Each is gated on the dev and validation sets, with the
   trace explaining every voice that moved. Candidates already visible:
   - title and description guest phrases ("X on …", "— X", "with X", "In conversation with X") as
     corroboration;
   - an LLM name may not be reused on a second voice once the episode bound it by
     self-introduction;
   - `self_intro` split into self_intro / intro_reader / cohost_formula.
