# Speaker-naming defects found by V.6b — handover (#2187, 2026-10-08)

V.6b ran the production profile on six real non-English episodes (one each from es, it, fr, de,
pt-PT, pt-BR shows). Checking who is named what, against the transcript text and the speaker
decision traces, found the defects below. One is fixed on `feat/2187-a2-gap-recovery`; the rest
are handed to the speaker-naming work. No transcript text is quoted here: evidence is counts,
decision-trace fields and file paths in the private eval repo.

Run artifacts (private): `podcast-scraper-eval-data/cache/v6b_real/<run_id>/<feed>/run_*/transcripts/`
— `*.speakers.diagnostics.json` holds the decision trace. **Trap:** after a successful translation
`<stem>.segments.json` is the ENGLISH text with no `speaker` field; the source-language segments
are `<stem>.<lang>.segments.json`. A roster replay that loads `segments.json` sees no voices and
reports a vacuous "fidelity OK".

## Fixed here: a refused "<name>, guest" answer left its guest role on the host (pt-PT)

`resolution.resolve_voices_and_roles`: the LLM answered `SPEAKER_03 = <guest name>, guest`; the
third-person guard refused the name (the voice introduces the guest), but the role "guest" stayed
on SPEAKER_03, and ADR-137 lets an LLM "guest" demote a positional host. Result: the host the feed
names was unseated and the episode had no named voice. Now `_role_after_third_person_refusal`
drops "guest" (and only "guest") with a third-person-refused name. Replay with the stored LLM
answers: one voice changes on the six feeds (that host restored); on prod (snapshot 2026-10-07,
the 30 episodes carrying verdict traces) the two affected English episodes do not change. Text
check: the restored voice opens the episode, asks the guest's view by name, and closes thanking
them for coming.

## Open 1: hosts who call each other by first name or nickname are named swapped (es)

- **Symptom**: two-host show; both voices `source: llm_resolution`, names swapped.
- **Evidence** (counts from the source segments): the voice published as host A addresses host A's
  nickname/first name ~30 times; the voice published as host B addresses host B's first name 4
  times. People address the other person, so the labels are inverted. A few counter-examples sit
  where diarization bleeds between turns.
- **Cause, from the code comments, not yet traced end to end**: `refuted_by_third_person` matches
  the full name or the surname only (`roster.py` ~3926: "co-hosts call each other by first
  name"), so first-name and nickname address never refutes a wrong binding.
- **Direction**: count vocative first-name / nickname address per voice as evidence against that
  voice being the named person (nicknames need a per-language prefix rule or a small alias list).
  Replay against English two-host shows first: they use first-name address too.

## Open 2: near-identical names merge two different people (de)

- **Symptom**: two-host show; the co-host is published unnamed, role guest, 79% of talk.
- **Evidence** (decision trace, `decision_trace.voices.SPEAKER_00`): `llm_merge` accepted
  `name: <host>` with `llm_said: <co-host>` — the LLM named the co-host correctly, and the merge
  snapped it onto the stated host, whose surname differs by two letters. Guest naming then refused
  it as `name_already_on_another_voice`; `summary.unbound_names` lists the co-host.
- **Cause**: the near-identical-host snap (`_snap_near_identical_host`) exists for ASR misspellings
  of one host's name, and cannot tell that from two different people with similar names.
- **Direction**: do not snap a name that is itself in the episode's stated names
  (`metadata_named` holds the co-host here) — a stated name is a person, not a misspelling.

## Open 3 (verify first): the fr host and guest may be swapped

The self-introduction ("I am <host>") was hidden under invented subtitle-credit lines until
ADR-160/159 recovered it on this branch. Recovered, it is spoken by SPEAKER_02, while production
published the host's name on SPEAKER_01 (66% of talk; the guest is the long-answer voice in this
format). Recovery runs before speaker alignment and naming, so the end-to-end V.6b re-run on this
branch shows whether the naming now resolves on its own; check that run before working on it.

## Not a naming defect (checked)

- de, 79/21 talk split: not merged voices. The 21% voice self-introduces and asks the questions;
  the 79% voice is the analyst co-host. The defect is Open 2.
- it: one host named by self-introduction; the two other main voices stay unnamed. Not examined
  further.
