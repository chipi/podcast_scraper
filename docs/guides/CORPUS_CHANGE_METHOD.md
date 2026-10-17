# Corpus change method — how every change to what the corpus serves is planned, judged and shipped

Status: **DRAFT v0.4** (2026-10-17). Two practice runs done (0030/0031 evaluation, 0030 go). Nothing is built under it yet.

## Why this document exists

On 2026-10-17 migration 0031 passed every check it had and made the corpus worse. It named 48
voices and left all 1,085 of their insights hidden. With 0030 applied first, a clone of the
2026-10-16 prod snapshot went from 28,251 to 28,386 hidden insights; 0031's own effect was +138
insights hidden on unknown voices. It was pushed and handed to prod as good.

Over the 16 days before that, seven speaker migrations and the naming arc ran. The share of
insights hidden by our own naming defect went **up**: 12.4% on 2026-09-30, 13.9% on 2026-10-16.
The work was busy and thoroughly checked, and it did not move the outcome.

The cause is the method, not one bug. Two independent reviews (Fable 5 advisor, 2026-10-17) named
the defects below. This document is the method that replaces them. Every change to naming,
attribution or anything else the corpus serves follows it, and the method itself changes only
through its changelog (section 7).

**Scope.** A change is anything that alters what a listener sees:

- pipeline code: naming, diarization, GI, KG, ad cut, transcription;
- corpus migrations;
- reprocess batches and backfills;
- rule, threshold or profile changes.

## 1. What was wrong, and the gate that now stops each one

A defect counts as removed only when something stops it **without anyone remembering to**. Each row
names that gate. A gate marked *to build* does not exist yet, and strategy item 1 builds it before
any other change is judged.

| # | Defect | Evidence | The gate |
| --- | --- | --- | --- |
| D1 | **The builder writes the check, scoped to the builder's own list.** It passes by construction. | 0031's `verify` asks only "does each listed voice carry the listed name?" It cannot see an insight. `m0025_parity.py` compares names per voice. | The acceptance list is written by the reviewer and attached to the issue **before a branch exists**. At step 8 the reviewer **re-derives** S1, S2 and S7 from the clone with the committed scoreboard command, and picks the sample of changed voices to read. The builder's table is never the evidence. |
| D2 | **Every check measures change (old vs new); none measures the outcome against a target.** An episode broken under both counts as "agree". | `NAMING-COVERAGE-PLAN-2026-10-10.md`, "Why this was missed". The guardrail (#2334 S8) was moved later and never became an acceptance criterion. | The committed scoreboard command (*to build*, strategy 1a). `naming_gate.py` prints S7/S8 by default (*to build*, 1b). |
| D3 | **Mechanics stand in for the outcome.** Undo byte-identity, order equivalence, quote integrity and CI green were reported as proof that the change works. | 0031 handover, 2026-10-17. | Report template (§5): outcome rows first, and a row without a measured value is rejected. Prod refuses a handover without the deploy table (§6; the rule must live in the prod session's own instructions). |
| D4 | **Verdicts come before the measurement that decides them.** | Seven instances on 2026-10-17: "judged", "13× faster", "0031 ready", "expect refusals", "nobody tracks this", "the pipeline keeps producing it", unverified numbers in a review brief. | The step-4 prediction is a dated record on the issue. The template rejects a row with "predicted: —". Step 8 compares against the prediction, not against the builder's account. |
| D5 | **Lessons are memories, not gates.** | 10-10 lesson, 10-16 memory rule, 10-17 repeat. | Migration outcome hook (*to build*, strategy 1d): a migration cannot apply on prod without a receipt holding S1/S2 before and after from its clone run. |
| D6 | **The gold sets and gate measure per-voice accuracy, not insights under the right person.** An 80% guest weighs the same as a 1% clip. A run that reads "0 better, 0 worse" is reported as a pass. | val_v1: 2,010 of 2,382 voices scored. Gates #2362–#2366 read `REGRESSIONS 0 / FIXES 0` on every set. | S7/S8 in `naming_gate.py` on the raw-voice join (*to build*, 1b). A run that moves 0 voices prints "no effect", never "pass", next to the insight-weighted delta. |
| D7 | **Code paths are trusted without being exercised.** m0025's rewrite was only ever used to remove or rename. 0031 was the first to add names, and its gain path never cleared `voice_type` / `surfaceable`. | `m0025_one_person_one_entry.py` L885–990, L1129–1145. | A table test on m0025's `_Episode` covering every transition (gained, lost, renamed) across every type (person, unknown, unidentified, commercial). It asserts segment, insight, quote and diagnostics fields together. Named in #2380's "done means". |
| D8 | **Causes are picked from vivid examples, not ranked by measured loss.** | "Corroboration drops the guest" was wrong (#2200 had measured otherwise on 10-01). The "193 episodes" pattern is looser than K8, and its cause is undiagnosed. | The step-3 artefact is produced by a committed script over stored decision records, not classified by hand. |
| D9 | **Existing work is not read first.** | #2200, `naming_coverage.json`, `quality/attribution.py`, `/metrics` all existed; I claimed none did. Recurred inside v0.1 of this document: "wrong-person attribution is in no issue", when it is K1 #2333, F4a, #2224, #2367 and #2352. | The step-1 reading list is a required section of the issue. The reviewer checks it against `gh issue list` and a docs search before writing acceptance. |
| D10 | **Corrections do not travel.** | "13× faster", "0 → 109 hidden", "worse" for rediarize_only, "I re-measured it" (it was the advisor's measurement). | Every number sent to the operator, prod or an issue carries the script path and snapshot id that produced it. A corrected number goes to every recipient of the original, logged in the report's CORRECTIONS line. |
| D11 | **Partial scope is not stated.** | 0031 reaches 22 of the 835 held voices. | The step-4 prediction states the population reached and not reached, as counts. |
| D12 | **The snapshot does not carry what the scoreboard reads.** | On the 10-16 snapshot, `naming_coverage.py --lang en` gives 379 episodes with no named voice against prod's recorded 267: 122 episodes read "no segments" in the backup (prod: 1). There was no 10-17 snapshot (backup failed, #2378). | A one-time check that the backup carries every file the scoreboard reads (*to build*). Every table carries its snapshot id. Rule when no valid snapshot exists: nothing is judged and nothing ships; the gap is reported. |
| D13 | **"Read" is claimed with no record of what was read.** | "Every stretch read (736)" (0030) and "every one read against the voice's own words" (0031) have no per-item verdicts. The 0031 list holds a wrong-person pair (Hard Fork: Kevin Roose's voice named Jill Lepore and the reverse). Its stored "evidence" is the voice's first words, which were an ad read. | Every read claim attaches a verdicts file (item, verdict, the deciding line) to the issue. A list migration stores the **deciding** evidence per entry (a self-introduction, or the address that identifies the voice), and a test refuses an entry whose evidence holds neither. |
| D14 | **Identity joins by container, not by file.** | "73 of #2332's 91 episodes are in 0030's population" joined by run directory, which holds a whole feed. The per-file answer is 9. It went into the evaluation unchecked. | Every number sent states its join key. Episode identity is `episode_id` or the per-file path, never a directory. |
| D15 | **The clone's population is assumed to be prod's.** | 13 episodes in 0030's snapshot plan were already rewritten on prod (9 by #2332, 4 by #2370); my "~356 recut" was a guess from code. | Before apply, prod produces the per-episode plan (every counter) and it is diffed against the clone's. Every episode in one and not the other is listed with its cause. The migration hook carries this diff. |
| D16 | **Checks that cannot fail.** | 0030's handover spot check "the cold open is in the ad-free text" was true before and after. Its "no Felix by Rogo insights left" was false. | A spot check states what it reads in both states and must be able to fail. Spot checks are chosen by the reviewer. |
| D17 | **Derived surfaces listed from memory.** | 0030's handover omitted the enrichment sidecars (`insight_sentiment`, `insight_density`) and the transcript manifests, which are read at serve time or record counts. | A census of every sidecar suffix under `metadata/` and `transcripts/`, and of every reader of the fields the change touches, is part of the run steps. Each one is ticked or stated as left stale. |

## 2. Principles

1. **Outcome first.** Every change names the outcome a listener sees that it exists to move, and
   reports that number before any other. (D2, D3)
2. **Predict before building.** The expected movement is written down as a number, with its
   population, before the work starts. A result that differs is investigated, not explained away.
   (D4, D11)
3. **The judge is not the builder.** The acceptance criteria are written by an independent reviewer
   from the goal, before the reviewer sees the change. The builder may add checks, never remove or
   weaken them. (D1)
4. **Absolute, corpus-wide, on prod data.** Every outcome is measured over the whole served corpus
   on a prod snapshot clone, before and after, with the snapshot id. A subset or a fixture is never
   the evidence. (D2, D12)
5. **Nothing gets worse, and the trade rule.**
   - S1n (named but hidden) and S7-wrong (wrong person) never rise.
   - S1, S2 and S3 may rise only by insights listed by identity in the prediction as one of:
     - leaving a wrong name: removing a wrong name from a real voice lands it in `unknown`;
     - moving into a correctly typed non-person class (an advert read typed `commercial`);
     - an accepted loss the operator approved.
   - Any other rise fails.
   - Removing a wrong name is an improvement, even though it hides an insight.
   - A deploy carries only passing changes. (Operator rule, 2026-10-17)
6. **Rank by loss.** Work starts on the cause that costs the most outcome, measured by a committed
   script from stored decision records. Examples illustrate a cause; they never select it. (D8)
7. **Read before building.** Every change starts with the existing issues, docs, scoreboards and
   metrics on its topic, listed on the issue. (D9)
8. **Every path exercised.** Each branch a change can take has a test on real-shaped data and a
   count on the clone. (D7)
9. **Gates, not lessons.** A lesson becomes a check in code, the runner, the gate or the handover,
   or it is not learned. (D5)
10. **Corrections travel**, with the script path and snapshot id. (D10)
11. **"Not verified" is a complete answer. "Done" without the deciding measurement is not.** (D4)

## 3. The scoreboard: what each number measures, and where prod is today

**The goal it serves.** A listener sees each insight, under the person who actually said it. Two
things go wrong, and they need different instruments:

- **Hidden:** a real person's insight is not shown, because we did not name their voice. The
  corpus can count this on its own.
- **Misattributed:** an insight is shown under the wrong person. A wrong name looks exactly like a
  right one, so only labelled truth can measure it (the gold sets), plus one label-free proxy (S11).

**Which numbers gate.**

- S1n, S1c, S7-wrong, S6 and S12 gate.
- S1, S2 and S3 gate under the trade rule (principle 5).
- The rest are cross-checks: they are reported and investigated, but cannot pass or fail a change
  on their own.

**These rows are a floor, not the list.** A change also names its **own** outcome numbers, each
with its definition, population and script. The ad cut, for example, gates on rows A1–A4 below. A
naming-only scoreboard cannot judge a change whose purpose is not naming (both reviews of 0030
found this).

Every row has one definition, one source, and one command. Every measured value carries its
snapshot id or timestamp.

**The committed command** is `scripts/measure/corpus_scoreboard.py`; the definitions are in
`podcast_scraper.quality.corpus_scoreboard`. It covers S1, S1n, S1c, S2, S3, S10a, S10b, S12 and
A1, and writes out every population gap, zeros included. A value in this section that it did not
produce is marked provisional. No number enters this section without its script stored.

### Corpus numbers (whole served corpus, prod data)

| # | Exact definition | Gates? | Source | Snapshot `snapshot-prod-20261016` | Prod live 2026-10-17 ~16:09 |
| --- | --- | --- | --- | --- | --- |
| S1 | Insight nodes with `surfaceable is False` **and** `speaker_voice_type == "unknown"`, over all Insight nodes; served episodes, all languages. "Hidden by our naming failure." | yes | GI files | 16,212 / 116,649 (13.9%) | not measured: no 10-17 snapshot (#2378) |
| S1n | **Named but hidden, must not rise:** Insight nodes whose `speaker` is a real name (not `SPEAKER_NN`, not empty), yet `surfaceable is False` with type `unknown` or `unidentified`. This is exactly how 0031 failed, and S1c cannot see it: the type still says unknown. | yes | GI files | 1,786 (before any 0031). After 0030+0031 on the clone: 2,871 (+1,085 = 0031's named voices) | not measured |
| S1c | Consistency count, must be 0: Insight nodes whose type says person (or no type) but `surfaceable is False`, plus nodes typed `unknown` whose `surfaceable` is not False. | yes (= 0) | GI files | 0 (also 0 after 0030, and after 0030+0031: blind to 0031's failure, hence S1n) | not measured |
| S2 | Insight nodes with `surfaceable is False`, over all Insight nodes, split by `speaker_voice_type`. | trade rule | GI files | 28,251 (24.2%): unknown 16,212, unidentified 11,959, commercial 80 | not measured |
| S3 | Of S2's `unidentified`: insights on a voice holding ≥30% of the episode's talk time (diagnostics `talk_time_s`; the denominator includes ad voices' talk). The diagnostics file is found by the transcript's file name under `transcripts/`, not by the stored absolute `/app/...` path (v0.1 used the stored path and wrongly reported 6,542 and "1,694 with no diagnostics"). **The voice must be found by the raw-voice join (S7's), not by the insight's `speaker`:** after a rename, `speaker` is a name and the lookup fails (0031's apparent −581 was this artefact). | trade rule | GI + `*.speakers.diagnostics.json` | **8,236** (committed command; on the 0030 and 0030+0031 clones too). The 7,710 written here before came from the insight-`speaker` join this row forbids; the difference is 526 named guests still typed `unidentified`, read 12 of 12 correct by the reviewer | not measured |
| S4 | Quote nodes whose `speaker_id` starts with `person:`, over all Quote nodes; English episodes only. | **no**: it improved under the defective 0031 (+2,051 named quotes on the clone) | `quality/naming_coverage.py`; `/metrics` `podcast_corpus_named_quotes` / `podcast_corpus_quotes` | 165,739 / 224,960 (73.7%) | 169,350 / 226,459 (74.8%) |
| S5 | English served episodes with GI where no segment carries a non-`SPEAKER_NN` label. | **no**: improved under the defective 0031 (−12 on the clone); **not reproducible from the backup** (D12) | same; `/metrics` `podcast_corpus_episodes_no_named_voice` | prod's record: 267. From the snapshot: 379 (122 without segments) | 199 |
| S9 | Unattributed talk share: the corpus mean of `unattributed_talk_share` and `unattributed_defect_share` from every speaker-diagnostics file (#1897). It is independent of how many insights the model happened to emit. | cross-check | diagnostics | not measured | not measured |
| S10a | Episodes with zero visible insights on the attributed surfaces: insights exist but `insights_from_gi(gi)` with the default arguments returns none. | cross-check | GI + `server/app_gi_view.py` | 189 (190 after 0030: Ben Horowitz, whose one visible insight was a Ramp read) | not measured |
| S10b | Episodes whose own insights panel is empty: `insights_from_gi(gi, unattributed_fallback=True)` returns none (#2198's definition). | cross-check | same | 2 (2 after 0030) | not measured |
| S12 | **Quote integrity:** Quote nodes whose `char_start:char_end` does not slice to their own text in the file their `transcript_ref` names, corpus-wide. What "Play from" and clips land on. | yes (must not rise) | GI + transcripts | 167 (24 after 0030: 143 off → ok, 0 ok → off). Also 244 quotes whose `transcript_ref` names a missing file. | not measured |
| S11 | Non-person names on shown insights: shown Insight nodes whose `speaker` fails the pipeline's person-name test (`is_person_name` / `plausible_person_names`), e.g. "Latin American Weekly", "Hey Josh". A label-free misattribution proxy. | cross-check | GI | not measured | not measured |

### Ad-cut numbers (the ad cut's own outcome; gating for any change to the cut)

| # | Exact definition | Gates? | Source | `snapshot-prod-20261016` | After 0030 (clone) |
| --- | --- | --- | --- | --- | --- |
| A1 | **Ad copy left:** hits of the committed sponsor markers (`quality.corpus_scoreboard.AD_MARKERS`, between word boundaries; bare "use code" and "sponsored by" excluded after a full read showed talk) in served ad-free text: hits and episodes, plus episodes with no ad-free text. An indicator. | yes (must not rise) | `.adfree.txt` | **569 hits, 343 episodes** (3 without ad-free text). The earlier 644 / 364 came from an unsaved list and cannot be reproduced | **357 hits, 274 episodes** |
| A2 | **Talk cut with an ad:** characters of a dropped ad-free segment outside the ad map's excised ranges (≥20 after the `Name:` marker), measured from old vs new ad-free segments. Pieces, characters, episodes, plus the unmeasurable segments. | trade rule (operator-accepted size) | ad-free segments + ad maps | — | marginal 188 pieces, 22,580 chars, 112 episodes (+58 unmeasurable segments, all read as ad/promo). Upper bound by set difference: 217 / 24,704 / 123 |
| A3 | **Ad-built output removed, by identity:** insights whose every grounding quote is cut ad text; entities whose name is only in cut text; quotes in cut text. | yes (each listed and read) | GI, KG | — | 54 insights (2 real: accepted), 175 entities (3 real: accepted), 98 quotes |
| A4 | **Promo restored:** restored ad-free text that carries a promo marker (the show's own plugs brought back). | cross-check | `.adfree.txt` | — | 56 pieces, 6,543 chars of 130,071 restored |

**Per language.** S1–S3 and S9–S11 are reported in total and per language. S4/S5 are English-only
by default (`DEFAULT_LANGUAGE = "en"`); a per-language row is to be added.

**Counts and shares, always.** Reprocessing and migrations move the denominator: Insight nodes went
116,649 → 116,595 after 0030.

**History.** S4/S5 have a time series in VictoriaMetrics. S1–S3 and S9–S11 have none until they are
added to the snapshot and the `/metrics` gauges (strategy item 5).

### Gold numbers (labelled truth)

The gold sets are hand-labelled voices on real prod episodes. Each label gives role, name (only if
the episode's own text supports it), the evidence quote and a confidence. A case is usable only
while its labelled run is still the served run, because a reprocessed episode can renumber its
voices. Each gold set is pinned to a snapshot release id, so S7 is reproducible.

| # | Exact definition | Gates? | Source | Snapshot `snapshot-prod-20261016` |
| --- | --- | --- | --- | --- |
| S6 | Labelled voices whose published name or role became worse, old code vs new (the gate's severity classes). A run that moves 0 voices is reported as "no effect", never "pass". | yes (= 0) | `scripts/measure/naming_gate.py` | guard; no baseline |
| S7 | **Insight-weighted correctness.** Join: insight → its first grounded quote → `char_start` → the segment containing it → that segment's **raw** `speaker` id (the pipeline's own `_segment_char_spans` path). Among insights on scored, nameable labelled voices: correct (stored name is the labelled person) / **wrong person** / unnamed. Insights with no grounded quote are counted and reported separately. | correct and wrong: yes | to build into `naming_gate.py` | **NOT VALID yet.** The v0.1 prototype joined by stored label, which double-counts when one name sits on two voices (795 insights on val_v1). To be re-derived. |
| S8 | **Undecidable:** insights on voices labelled `confidence: low` or `role: unknown`. Reported as its own number; never relabelled to flatter a rule. | no | same | to be re-derived with S7's join (v0.1 prototype: dev 77, val_v1 1,068, val_v2 694) |

**Known limits of the matcher.** The gate's `same_person` accepts token subsets, so a stored bare
first name ("Josh") matches the label "Josh Elman", and "Sam" matches any Sam. Some wrong-person
cases therefore count as correct. Every S7 report says so, until the matcher is tightened for S7.

### What the gold sets can and cannot measure (measured 2026-10-17)

**They can:**

- Cover 700 served episodes (92 dev_v1, 461 val_v1, 147 val_v2) of 2,685, and about 30,000
  insights.
- Bound the hidden insights in those episodes. The v0.1 prototype put 81% of val_v1's hidden
  insights on scorable voices. That figure comes from the inflated join and is to be re-derived.
- Measure misattribution, which no corpus number can. Those cases are already tracked as the
  gate's `wrong_name` class, K1 swaps (#2333), F4a, #2224, #2367 and #2352 Open 1. What is new is
  weighting them by insights.

**They cannot, as used today:**

- **They count voices, not insights.** Fixed by S7.
- **They have a speakerless gap.** 348 / 1,159 / 160 insights (dev / val_v1 / val_v2) have no
  `speaker` and fell out of the v0.1 denominators unreported.
- **They decay.** 51 of 751 labelled cases (dev 9, val_v1 39, val_v2 3) are no longer the served
  run. #2332 keeps reprocessing, so coverage shrinks.
- **val_v1 is no longer held out.** It was read while tuning (`NAMING_KNOWN_ISSUES.md` caveats) and
  used to choose between variants. It is now a development-tier set. val_v2 has been scored once
  on K8.
- **Their feeds are not matched to the loss.** The top-loss feeds are Rest Is History, EconTalk,
  Rest Is Politics, Conversations with Tyler, ChinaTalk, The Daily, MLST and a16z. The match is not
  measured.
- **The undecidable bucket is unexamined.** It is not known how much of it is "the text cannot
  tell" and how much is "the labeller gave up".

S1–S3, S1c, S9–S11 and S7–S8 are to be one committed command over a snapshot, built from the
existing modules and kept with this document.

## 4. The procedure

Every step leaves a written artefact, attached to the change's issue. A step without its artefact
is not done.

| Step | What happens | Artefact | Who |
| --- | --- | --- | --- |
| **0. Gates exist** | Precondition for judging any change: the scoreboard command, S7/S8 in the gate, the migration outcome hook, and a valid snapshot (D12). If one is missing, building it is the change. | — | builder, reviewer checks |
| **1. Intake** | Read every open and recent issue, doc, scoreboard and metric on the topic. State what is already known and measured. | Reading list on the issue, one line each. | builder |
| **2. Outcome and baseline** | Name the scoreboard numbers the change targets. Take the baseline on a fresh prod snapshot, with its snapshot id. No valid snapshot: stop and report. | Baseline table with command, output and snapshot id. | builder |
| **3. Diagnosis by loss** | Rank causes by outcome lost, with a committed script over stored decision records (decision traces, LLM verdicts, diagnostics). Read samples of each top cause. | Ranked cause table, script path, read samples. | builder |
| **4. Prediction** | For the chosen cause: **integers on a named snapshot** for each gating number and each of the change's own outcome numbers; the population reached and not reached (counts); trade-rule moves listed by identity; what must not move. "~" is not a prediction. "Within prediction" can only be claimed against integers. When the population moves (D15), the prediction is restated before apply, never explained after. | Dated prediction on the issue, before any code. | builder |
| **5. Acceptance** | The reviewer checks step 1's reading list against the tracker, then writes the pass criteria from the goal and the prediction, without seeing code. The criteria name the change's **own** outcome numbers (definition, population, script). The §3 rows are a floor. | Acceptance list on the issue, before a branch exists. | reviewer |
| **6. Build** | Scenario test first (real-shaped data, every path from principle 8), then the change. A list migration stores the deciding evidence per entry (D13). | Tests and code. | builder |
| **7. Measure** | On a snapshot clone: the scoreboard and the change's own numbers before and after; a whole-corpus replay; gold gates; counts per path; **a full-clone undo test** (restored N, refused 0, byte-identical to the snapshot, verify reads pending again); **the per-episode plan** with every counter (and, where cheap, each input file's sha256), for the step-10 diff; **the derived-surface census** (D17). | Before/after table, gate output, undo result, plan file, census. | builder |
| **8. Verdict** | The reviewer mutates each definition against its tests (a mutation that leaves every test green is an unguarded path) and re-derives the gating and own-outcome numbers **with its own code**, or with a line-by-line read of the committed script plus one number recomputed by hand. The committed command is for prod and history, not for independence. It picks and reads its own sample, writes the verdicts file (D13), and compares against the dated prediction and the acceptance list. A miss sends the work back to step 3, never a reworded pass. | Verdict against each criterion; verdicts file. | reviewer |
| **9. Operator review** | The operator sees the table first, the mechanics second, and a not-covered list as long as the covered list. | Report (§5). | operator decides |
| **10. Before apply / deploy** | **Prod produces its own per-episode plan, read-only** (the dry run plus the plan listing), and it is diffed against the clone's. Every difference is listed with its cause, or the run stops (D15). For a deploy: everything it carries (pipeline code plus migrations, in runner order) runs on one fresh snapshot clone, and the gating numbers pass under principle 5. The run steps include every derived surface from the step-7 census. The job and its undo are written for exactly the migrations being run. | Plan diff; deploy table; run steps. | operator decides |
| **11. After apply / deploy** | Re-measure on prod; check the apply's counts against the restated prediction (any other number stops before the derived-surface steps); run the reviewer's spot checks; read prod warnings for the affected window. | Post-run table. | builder, prod |
| **12. Retrospective** | Any miss, surprise or correction becomes a method change (§7). | Changelog entry. | builder, operator |

## 5. Report template

A row with no measured value, or with "predicted: —", is rejected.

```text
OUTCOME (snapshot <release id>, clone)   before → after   predicted   pass?
OWN   the change's own outcome numbers (definition, population, script)  ...
S1n named but hidden                    ...
S1  hidden by our defect (trade-rule moves listed)  ...
S1c consistency (must be 0)             ...
S12 quotes off their text               ...
S7  gold: correct / wrong person / unnamed (insights)  ...
S6  gold regressions (or "no effect")   ...
S2  all hidden, by type (trade-rule moves listed)       ...
S3  principal unidentified              ...
S9  unattributed talk share             ...
S10a / S10b episodes with zero visible insights (attributed / panel)  ...
S11 non-person names on shown insights  ...
S4  named quotes share (cross-check)    ...
S5  episodes with no named voice (cross-check, prod /metrics only)  ...
S8  undecidable                         ...

POPULATION: reached N / not reached M; clone vs prod plan diff
COMMANDS: <each number's definition, join key, script path, command and output fragment>
UNDO: <full-clone undo result>
DERIVED SURFACES: <each from the census: done / when / left stale and why>
PATHS EXERCISED: <counts per path>
READ: <reviewer's sample: how many changed voices, the verdicts>
NOT COVERED / NOT VERIFIED: <at least as long as the above>
CORRECTIONS SENT: <any earlier number this changes, and to whom>
```

## 6. Roles

- **Operator (Marko):** sets goals, approves predictions that trade one number against another,
  and decides deploys.
- **Builder (executing agent):** steps 1–4, 6, 7, 11, 12. Never writes or weakens the acceptance
  criteria.
- **Reviewer:** a **fresh** advisor agent per change, always spun out, never the builder and never
  a resumed reviewer (operator decision, 2026-10-17: "the key thing"). It is bound by this
  checklist:
  - check step 1's reading list against the tracker;
  - write acceptance before seeing code, naming the change's own outcome numbers;
  - re-derive the deciding numbers with its own code;
  - choose and read its own sample, and write the verdicts file;
  - choose the spot checks, each able to fail;
  - compare against the dated prediction;
  - treat earlier reviews as inputs, not evidence.
- **Prod session:** refuses any handover without the outcome table, the plan diff and the commands.
  The prod session loads its instructions from the agentic-ai-homelab repo, not from this one, so
  the rule lives in the homelab `corpus-migration` and `prod-batch-chain` skills and in the global
  `workstation/config/AGENTS.md` (operator approved, 2026-10-17). Until it is merged there, it is a
  reminder, not a gate.

## 7. Changing the method

The method is versioned. Every miss or correction gets an entry: what happened, which defect or
principle it touches, and what changes in the method. A change that only adds a reminder is not
accepted. It must change a check, a gate or a step.

| Version | Date | Change |
| --- | --- | --- |
| 0.1 | 2026-10-17 | First draft, from the m0031 failure and the first Fable 5 review. |
| 0.2 | 2026-10-17 | Second review. v0.1's own defects: S7 baselines from a double-counting join (D6, D10); "wrong-person attribution is in no issue" (D9, recurred); the 193-episode pattern presented as K8 (D8, recurred); S4/S5 shown as able to gate, though both improved under the defective 0031; "nothing gets worse" forbade correct fixes (trade rule added); snapshot gap D12 added; a gate column per defect; reviewer re-derives the numbers; step 0 added; S1 redefined with a consistency count; S9–S11 added. |

| 0.3 | 2026-10-17 | First two practice runs (0030/0031 evaluation, 0030 go). Changes:<br>- S1n added and made gating, because S1c was blind to 0031's failure;<br>- the trade rule now covers S1, since wrong-name removals land in `unknown`;<br>- S12 quote integrity, S10a/S10b, and ad-cut rows A1–A4 added;<br>- the scoreboard is a floor: a change names its own outcome numbers;<br>- D13–D17 added (read verdicts, per-file joins, prod plan diff, falsifiable spot checks, derived-surface census);<br>- integer predictions, restated when the population moves;<br>- a full-clone undo test and a per-episode plan in step 7;<br>- the reviewer uses its own code;<br>- the prod rule's location named. |
| 0.4 | 2026-10-17 | Third practice run: the scoreboard instrument, reviewed by a fresh advisor (FIT WITH CHANGES, all applied). Changes:<br>- S3 corrected 7,710 → 8,236: the earlier value used the join its own row forbids, and the prediction copied it;<br>- A1 re-baselined under the committed, word-bounded list (the earlier list was never saved);<br>- every §3 value must come from a stored script, and is provisional unless the committed command produced it;<br>- at step 8 the reviewer mutates the definitions against their tests (the mutation run found four unguarded paths, now tested);<br>- reading every hit of a marker, not a sample, decides its precision. |

## Operator decisions (2026-10-17)

1. **The scoreboard as the yardstick:** yes.
2. **The reviewer:** yes. A fresh advisor per change, always spun out: the key rule.
3. **Gold sets:** yes, as below.
   - refresh the 51 stale cases as a new labels version, re-cut from the served run. A label
     carries over only where the voice's talk share and first/last seconds match; otherwise
     relabel;
   - draw a new validation set now, half sampled in proportion to hidden insights per feed and
     half uniform over served episodes. Label it; score it once per candidate, logging every
     scoring in the runbook. val_v1 becomes development-tier;
   - audit a sample of about 40 undecidable voices with a stronger model, to split "the text
     cannot tell" from "the labeller gave up". Report both; never relabel to flatter.
4. **Where the method lives:** here in `docs/guides/`, absorbing the acceptance rule of
   `NAMING_GATE_RUNBOOK.md` (whose "Accepting a change" and "Absolute coverage" sections point
   here). Referenced from the repo `AGENTS.md`, in the mkdocs nav, `docs-preflight` green before
   commit.
5. **Where the prod session's refusal rule is written:** the homelab skills and global AGENTS.md
   (§6). Approved.
