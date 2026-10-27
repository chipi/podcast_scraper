---
id: known_host_third_person_veto
class: full
status: rejected      # 2026-10-27: failed acceptance 6/17 on its corpus reach; code reverted
issue: "#2094 (D5)"
touches: []           # rejected: guards nothing
---

# A feed's known host is never forced onto a voice that speaks of that host in the third person

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md). Operator, 2026-10-27: land
the fix as a hypothesis now; every re-measure, gate and review of the main scope runs once, at the
end ("park the whole validation complexity where we rerun ... then we go and validate"). Steps 2
and 4 therefore carry the numbers read from stored evidence today, and say which numbers the final
pass measures. This record covers the PIPELINE fix only; the served episode is repaired by a
separate text-only migration with its own record.

## 1. Intake

- **#2094 D5**, measured by prod on 2026-10-27 over 3006 served episodes: one episode with its two
  names swapped, `substack:post:165726942` (run `run_23b4222d-6325-4100-94dc-6ef228c1f611_20260904-084827`,
  naming profile `naming-4`). Evidence on the mini, `~markodragoljevic/prod-naming-20261027/d5/`
  (`files.sha256`).
- **The diagnostics** (`*.speakers.diagnostics.json`): `tried.known_hosts` = the feed's one host;
  `SPEAKER_01` (1145.8 s) holds that host's name with `source: known_hosts`; `SPEAKER_00`
  (1942.4 s) holds the metadata-named guest with `source: llm_resolution`. No self-introduction on
  either voice. In this episode the guest interviews the feed's host.
- **The guards that exist** (read 2026-10-27 on this checkout):
  - `pipeline._apply_llm_third_person_guard` (v54): only `source == "llm_resolution"` names.
  - `roster`'s guest check (`refuted_by_third_person` on a forced guest, ~line 3225, 3701).
  - the talk floor (`_apply_talk_floor`): only names under the floor.
  - the forced host seat's three vetoes (`roster.py`, `forced_name_vetoes`): performs a guest act
    (#2276), greeted by that name at the open (#2078), speaks of the host by given name (#2276),
    the last one exempt when the seat keeps addressing the co-host (bleed).
  None reads the resolver's third-person refusal on the forced host seat.
- **Related records:** v53 / v54 (`talks_about_in_third_person`), #2078 (the greeting veto),
  #2276 (the given-name veto and its co-host-address exemption).

## 2. Outcome and baseline

- **The outcome:** a feed's known host's name is not forced onto a voice whose own text refers to
  that host in the third person.
- **Baseline from stored evidence, 2026-10-27** (scratch `rm27/d5_veto.py`, `d5_tp.py`, run with
  this checkout's `src` on the D5 segments):
  - `SPEAKER_01`, the seated voice: `refuted_by_third_person(text, host)` = **True**,
    `talks_about_in_third_person` = **True**; the three existing vetoes = **False, False, False**.
  - `SPEAKER_00`: all of the above **False** for both names.
  - So today's code still seats the host on `SPEAKER_01` (no veto fires), reading statically; the
    episode was not re-run.
- **Served loss (prod, 3006 served):** D5 = **1** episode. Its GI holds **86** insights; quote
  attributions: **91** to the guest's person id, **3** to the host's (scratch `rm27/d5_gi.py`),
  nearly all of them on the wrong person.
- **Measured in the final pass, not now:** how many episodes of the snapshot clone the veto
  reaches (a replay of the roster over stored segments), and what each reached seat becomes.

## 3. Diagnosis by loss

1. **The forced host seat has no third-person veto** (this change): the only cause in D5. The seat
   is the first voice by start, which in an interview of the host by a guest is the guest.
2. Not this change: once the host seat is wrong, the LLM's closed list leaves the guest's name for
   the other voice. Whether, with the seat refused, the LLM names the host's real voice is unknown
   (the run's LLM answer is not stored in these diagnostics); with the veto alone the expected
   result is the seated voice unnamed. A wrong name is worse than a hidden one (operator,
   2026-10-19), so refusing the seat is the right first step.
Samples read: the 16 sentences on `SPEAKER_01` naming the host (ids only here; text in the evidence).

## 4. Prediction

Dated: 2026-10-27

- On the D5 episode, replayed through the roster: the veto fires on **1** voice (`SPEAKER_01`);
  forced host names placed: **1 → 0**; that voice ends unnamed (**0** new wrong names).
- On the unit scenarios: every existing seat test (the greeting, given-name, guest-act, co-host
  address and pair cases) keeps its result: **0** changes.
- A seat whose text keeps addressing the co-host (the bleed exemption) is **not** vetoed by the new
  check: **0** such seats change.
- What must not move: a seat that introduces itself as the host (the resolver's refusal already
  excludes self-introduction); seats named from an earlier rung (`named_from_earlier_rung`) and
  address pairs (the veto applies to the forced pool name only).
- Corpus reach on the snapshot clone: **not predicted as an integer yet**; it is measured in the
  final pass, and a reach above **5** episodes is read episode by episode before anything lands
  on prod.

## 5. Acceptance

Dated: 2026-10-27
Written by: a fresh reviewer (sonnet subagent), from sections 1-4 only, without code.

Floor: the CORPUS_CHANGE_METHOD §3 scoreboard rows apply as well. S1n and S7-wrong never rise. S1,
S2 and S3 rise only by insights listed by identity under the trade rule. The criteria below add
the change's own numbers.

1. **D5 outcome** (`substack:post:165726942`, run `run_23b4222d-6325-4100-94dc-6ef228c1f611_20260904-084827`,
   profile `naming-4`). Replay the roster over the stored segments with the changed code and the
   pre-change code. Pass only if all three hold: (a) the veto fires on exactly 1 voice,
   `SPEAKER_01`; (b) forced host names placed go 1 -> 0; (c) `SPEAKER_01` ends unnamed, and the
   host's name is on no voice. A different count, or any name on `SPEAKER_01`, fails. The reviewer
   reruns this with its own code, not the builder's `rm27/d5_*.py`.
2. **No new wrong names.** On the D5 replay every voice's name is unchanged or removed: 0 voices
   gain a name; `SPEAKER_00` (`llm_resolution`) unchanged. For every episode the replay reaches
   (criterion 6), the voice the veto leaves unnamed is the one that speaks of the host in the third
   person. A reached voice that is the host's real voice is a wrong-person refusal and fails.
3. **Existing seat behaviours unchanged.** The existing unit scenarios for the greeting veto (#2078),
   the given-name veto, the guest-act veto (#2276), co-host address and pairs: 0 changes in outcome,
   0 tests edited, weakened or deleted; each also run old code vs new code, voice by voice.
4. **Bleed exemption.** A seat with a third-person reference to the host plus co-host address keeps
   its result (0 seats change), and the test fails if the exemption is removed (mutated at step 8).
5. **What must not move.** A seat introducing itself as the host stays named; seats named from an
   earlier rung and address pairs stay as before; the veto touches only the forced pool name; on the
   clone, 0 changes in voices outside the veto's reach.
6. **Corpus reach on the snapshot clone** (snapshot id stated, joined by `episode_id`): episodes where
   the veto fires and voices unnamed, as integers per path, measured by the builder and re-derived
   by the reviewer. Reach at most 5: every reached episode is read. Above 5: the change stops, and
   nothing proceeds until every reached episode is read against the voice's own text, with a
   verdicts file (episode, voice id, verdict, deciding line). Also read every episode where the veto
   fires on a voice holding a name from a source other than `known_hosts`. One wrong refusal stops
   the change and goes back to step 3.
7. **Derived surfaces and scoreboard.** On the clone S1n and S7-wrong do not rise; any rise of S1, S2
   or S3 is listed by insight identity. A move of 0 voices is reported as "no effect", not a pass.
8. **Reject if:** the D5 replay still seats the host on `SPEAKER_01` or names a voice unnamed before;
   an existing seat test changes result or was edited to pass; the exemption has no test that fails
   when it is removed; reach is not measured, or reach above the threshold is not read; one reached
   voice is a wrong refusal; a number lacks its script path and snapshot id; `make ci-fast` is not
   green on the committed tree with the log attached.

Not decided by these criteria: the served D5 repair (its own migration record); whether the LLM
names the host's real voice once the seat is refused (only "unnamed" is required); the D5 insights
and attributions; whether 5 is the right stop threshold; deploy, the prod plan diff and the apply.

## 6. Build

- **Scenario tests, written first** (`tests/unit/podcast_scraper/providers/ml/diarization/test_known_host_third_person_veto.py`,
  7 tests): before the code, the two D5-shape tests failed (the opener voice seated as the host)
  and the controls passed. One planned control, a show named after the host with the full name in
  the title, turned out to be unseated by an existing rule when the feed title is passed, so it
  could not isolate this veto; it was replaced by a feed title holding only the surname, which
  today's code seats.
- **Code** (`roster.py`, the forced host seat's vetoes): a fourth veto, `named_in_third_person` =
  `refuted_by_third_person(seat text with the feed title removed, host, language)`, with the same
  co-host-address exemption as the given-name veto; recorded in the `forced_name_vetoes` trace.
- **Mutants, all killed** (`replay-proven-fix/scripts/mutate.py`, file restored by sha256): the
  exemption removed (the bleed test fails), the feed-title strip removed (the show-title test
  fails), the veto disabled (the D5-shape test fails).
- **Suite:** `pytest tests/unit/podcast_scraper/providers/ml/diarization/` 1002 passed.
- **The D5 episode's own segments through `resolve_speaker_roster`** (no LLM inputs; scratch
  `rm27/d5_roster.py`): old code `SPEAKER_01` = the host, `source known_hosts` (the served
  name); new code `SPEAKER_01` unnamed, `SPEAKER_00` unnamed. Not the acceptance replay: the
  reviewer reruns criterion 1 with its own code in the final pass.
- **Not built yet:** the corpus-reach replay (step 7) and the served repair (its own record).

## 7. Measure

Corpus reach, 2026-10-27, snapshot-prod-20261027 (11:04Z), scratch `rm27/veto_reach.py`: the
roster with the new code and deterministic inputs only (`tried.known_hosts`, detected guests,
metadata names, feed title, language; no LLM names) over every served episode with known hosts.

- Served 3006; with known hosts 2545; replayed 2545; errors 0.
- The veto fired on **52** voices. On **38** the served name is the host the veto refuses (36
  seated by `known_hosts`, 2 by `voice_identity`): those would be unnamed on re-ingest.
- What matched, read for all 38 (`rm27/veto_ctx.py`): most are the host introducing themselves in
  forms `_introduces_itself_as` does not read: "Hello, it's <host> with <show>", "with me, your
  host, <host>", "<host> here", "<host>, your personal astrophysicist", "Hallo, mein Name ist
  <host>", "my name, though, is <host>", "I'm your host, <co-founder> <host>". The predicted reach
  ("above 5 is read case by case") was exceeded by a factor of 10, and the cases read are wrong
  refusals of real hosts, not D5-shaped ones.

## 8. Verdict

**Rejected, by the record's own acceptance** (criteria 6 and 17: one reached voice that is a wrong
refusal stops the change). Not sent to a reviewer: the builder's reach alone fails it. The code and
its tests are reverted (the revert commit names this record). D5's served episode is repaired by
0037, which does not depend on this veto.

What a second attempt would need (step 3 again, not started): a host-seat signal that tells an
interviewer naming the host ("talking to <host>", "background on <host>") from the host naming
themselves, measured over the same 52 before any code. Not on the current plan.

## 9. Operator review

_TODO_

## 10. Before apply / deploy

_TODO_

## 11. After apply / deploy

_TODO_

## 12. Retrospective

_TODO_

## Not covered

_TODO_
