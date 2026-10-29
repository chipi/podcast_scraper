---
id: 0037_named_voices_corrected
class: full
status: landed        # applied on prod 2026-10-28 15:46Z (steps 10-11)
issue: "#2094 (D5), naming plan item 13 (night 1)"
touches:
  - src/podcast_scraper/upgrade/migrations/m0037_named_voices_corrected.py
  - src/podcast_scraper/upgrade/registry.py
---

# 0037 — served voices that carry another person's name get their own

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md). Operator, 2026-10-27:
land the fixes as hypotheses, validate everything once at the end. The served half of item A
(D5), with item C (night 1); the pipeline half of D5 was
the change record `known_host_third_person_veto` (rejected, no longer in the tree). Text only: no GPU host, no relabel,
no LLM (the corpus fix ladder, rung 2).

## 1. Intake

- **D5** (#2094; prod's measurement over 3006 served, 2026-10-27): `substack:post:165726942`, two
  voices, each holding the other's name (host seat by `known_hosts`, guest by `llm_resolution`).
  Evidence: the measurement's D5 directory (`files.sha256`), not in this repository.
- **Night 1, plan item 13** (two blind readers over 21 sampled episodes; eval data repository
  (not in this repository), `data/eval/naming-night1-20261027/`: `packets.jsonl`,
  `reader_A.tsv`, `reader_B.tsv`, `key.json`). Both readers agree on every voice below:
  - `substack:post:216929530`: `SPEAKER_00` and `SPEAKER_01` swapped (both `llm_resolution`;
    basis `addressed` / `host_read`, deciding segments 18 and 14).
  - `d99b4720-283b-4f07-89c4-791b12acba2d` `SPEAKER_04`: the feed host's name (`known_hosts`) on
    a voice that is the clip's own speaker (basis `clip_own_person` / `self_intro`).
  - `https://iono.fm/e/1732482` `SPEAKER_07`: a guest name (`llm_resolution`) on the voice the
    readers name as the clip's interviewee (`clip_own_person`).
  - `https://iono.fm/e/1732482` `SPEAKER_00`: both readers say another person, and it is already
    unnamed on the served episode (`talk_floor`): nothing to do.
- **Machinery reused, not new:** 0035's reviewed-list migration (frozen list, verdicts with basis
  columns bound by the list's sha256, a listed voice without a verdict fails `verify`), applied
  through 0029's voice-by-voice rewrite with 0031's derived surfaces; undo from receipts.
- **Not this change:** D4's 43 candidates (classified separately; real wrong names may join this
  list by a revision of this record); the `d99b4720` host-seat cause (a clip voice that never
  mentions the host: the new third-person veto does not reach it, checked 2026-10-27).

## 2. Outcome and baseline

- **The outcome:** each listed voice is published under its own person's name, and the name it
  held is on no voice of the episode that is not that person.
- **Baseline from the stored evidence, 2026-10-27:** wrong-person named voices **6** in **4**
  episodes. Insights credited to them (night 1's `key.json`; D5 from its GI, scratch
  `rm27/d5_gi.py`): `216929530` **10 + 79**, `d99b4720` **14**, `1732482` **3**, D5 **86** in the
  episode (quote attributions 91 to the guest's person id, 3 to the host's).
- **Measured in the final pass:** the scoreboard and per-episode deltas on a clone of
  snapshot-prod-20261027 (taken 11:04Z, after 0034-0036 and the 0030 re-run; the night-1 episodes
  are not in snapshot-prod-20261025).

## 3. Diagnosis by loss

All six voices are wrong-person names, the first rank of the loss order (operator, 2026-10-19:
a wrong name is worse than a hidden one). By insight count: `216929530` (89), D5 (86 in the
episode), `d99b4720` (14), `1732482` (3). By cause: a forced `known_hosts` seat on the wrong voice
(D5, `d99b4720`); the LLM's pair swapped (`216929530`); the LLM's guest name on a clip voice
(`1732482`). Samples read: both readers' rows for every voice; the D5 diagnostics and the 16 host
mentions on its `SPEAKER_01`.

## 4. Prediction

Dated: 2026-10-27

- Listed voices **6**: renames **6** (each to the readers' / evidence's name), unnamings **0**.
- Episodes written **4**; refused **0** on the clone of snapshot-prod-20261027 (a refusal means a
  stored name moved on since the evidence, and is reported, not forced).
- Per episode: `216929530` 2 voices, D5 2 voices, `d99b4720` 1, `1732482` 1; roles: D5 and
  `216929530` swap host and guest with the names; `d99b4720` `SPEAKER_04` host -> guest;
  `1732482` `SPEAKER_07` stays guest.
- Scoreboard: S7-wrong (wrong-person names) falls by **6** on the clone where the gold covers the
  voices, and is **0** elsewhere; S1n does not rise; the episodes' other voices do not move (**0**).
- `upgrade verify` with every applied migration: **0** new failures (0025's 5 known spelling
  flags unchanged).
- D5 has no blind read yet: its two rows enter the verdicts only from a fresh blind reader in the
  final pass; if that reader does not confirm both, D5 leaves the list and the count above drops
  to 4.

## 5. Acceptance

Dated: 2026-10-27

> Superseded 2026-10-27 by method v0.9: the acceptance criteria are section 4's integers plus the
> standing checks; "by the reviewer's own code, not the builder's table" is void; criteria 1-17
> below are kept as history and bind nobody.
Written by: a fresh reviewer (sonnet subagent), from sections 1-4 only, without code.

All measurements on a fresh clone of snapshot-prod-20261027 (id and clone id recorded), by the
reviewer's own code, not the builder's table (D1, D3). "Voice" = (episode, speaker label).

1. **Renames (6/6).** Each listed voice's published name equals its person's name per the frozen
   list and verdicts, read from the clone: `216929530` SPEAKER_00 and SPEAKER_01, D5 (`165726942`)
   both voices, `d99b4720` SPEAKER_04, `1732482` SPEAKER_07. Pass: renames = 6, unnamings = 0,
   none left on the old name.
2. **Roles.** `216929530` and D5 swap host and guest with the names; `d99b4720` SPEAKER_04 host ->
   guest; `1732482` SPEAKER_07 stays guest. Pass: all four match, read per voice.
3. **Every derived surface, per listed voice** (before and after): transcript speaker labels,
   segments, diagnostics (name source and basis), GI insights and quote attributions, KG person
   nodes and edges, bridge, search-index rows. Pass: no surface of an affected episode holds the
   old name on that voice; D5's quote attributions (91 / 3) follow the voices; insights credited per
   person match the corrected voices (10 + 79, 14, 3, 86 in D5). A surface left stale needs a
   census line with a reason, or the criterion fails (D17).
4. **Swap cases.** Per episode, the set of voice names and person ids holds no duplicate person; no
   voice holds both names; the old name sits on no voice that is not that person. Pass: 0
   violations in `216929530` and D5; the two voices exchange exactly.
5. **Wrong-person count.** S7-wrong falls by exactly 6 where the gold covers the voices, 0 outside;
   S1n does not rise; S1 and S2 rise only by insights listed by identity (trade rule).
6. **Refusals.** Episodes written = 4, refused = 0; any refusal listed with its cause, never forced.
   Pass: 0, or the prediction restated before apply (D15).
7. **Already-unnamed voice.** `1732482` SPEAKER_00 (`talk_floor`) is byte-identical after.
8. **What must not move.** Plan diff against the snapshot: every other voice of the 4 episodes 0
   changes (name, role, per-voice insights); every other episode 0 changed files (sha256 per file);
   0 files outside the listed voices' derived surfaces.
9. **Gate floor.** The §3 floor rows before and after (S1n, S1, S1c = 0, S12, S7, S6, S2, S3, S9,
   S10a/b, S11, S4, S5, S8): S1c = 0 and S12 unchanged; no unexplained movement; any gold
   regression listed.
10. **Upgrade verify** over every applied migration on the clone: `verify_regressions` = 0 (0025's 5
    flags unchanged). The migration's own verify fails if a listed voice has no verdict or the
    list's sha256 is not the one the verdicts bind.
11. **Evidence stored.** Each list entry stores its deciding evidence (a self-introduction or an
    address that identifies the voice); a test refuses an entry without it (D13). The verdicts file
    has one row per voice with item, verdict and the deciding line.
12. **Undo on a full clone:** restored N, refused 0, byte-identical to the snapshot by tree hash,
    verify reads the migration as pending again.
13. **Idempotence.** A second apply writes 0 files and changes 0 hashes.
14. **D5 blind-read condition.** D5 stays only if a fresh blind reader who has not seen the proposed
    names confirms both D5 voices (two rows in the verdicts). Otherwise D5 leaves the list: 4
    voices, 3 episodes, renames = 4, restated in the record before any apply; criteria 1-8 re-run.
15. **Verdicts binding.** The night-1 voices need both readers' agreement; the list's sha256 matches
    the verdicts; no voice applied without a bound verdict.
16. **Reviewer's sample:** its own sample of changed voices (D5 and one night-1 voice at least) plus
    untouched voices of the same episodes, with a verdicts file; every sampled voice reads right.
17. **Reject** (back to step 3): a listed voice ends on a wrong or old name; a duplicate person; a
    voice or episode outside the list moves; refusals > 0 without a restated prediction; S7-wrong or
    S1n rises; the undo is not byte-identical; `verify_regressions` > 0; D5 applied without the
    blind confirmation; a report row without a measured value; `make ci-fast` not green on the
    committed tree with the log attached.

Not decided by these criteria: D4's 43 candidates (a revision of this record); the `d99b4720`
host-seat cause; the pipeline half of D5 (future episodes); prod behaviour (steps 10-11); the
readers' judgement beyond their agreement and the sample; the corpus-wide wrong-name rate; the ad
cut, non-English ASR, deploy sequencing.

## 6. Build

- **Scenario tests, written first** (`tests/unit/podcast_scraper/upgrade/test_m0037_named_voices_corrected.py`,
  6 tests): a reviewed swap gives each voice its own name and role on the transcript text,
  segments, diagnostics (source `roster`), metadata speakers and GI persons (one Person per name),
  verify passes and undo restores; a swap with one member rejected writes nothing of the pair; an
  episode whose stored name moved on is refused untouched; the shipped list holds 6 renames in 4
  episodes; an entry without a deciding segment, or without both names, is refused at load; its own
  backups and receipts.
- **Code** (`m0037_named_voices_corrected.py`): 0035's reviewed-list migration with its own list,
  receipts, backup tag and `to_version` 2.7.31; `load_list` refuses an entry without old and new
  names or a `segment N`; `swap_groups` links voices of one episode where one holds the name another
  takes, and `_reviewed` drops a group unless every member is applied; `_Episode` keeps the
  rewrite's own source (`roster`) instead of the rung's. Not registered (verdicts first, final pass).
- **List** (`upgrade/lists/0037_named_voices_corrected.json`): the 6 voices of steps 1 and 4, each
  with its basis and deciding segment (D5: segments 1 and 14 of its `.segments.json`, found by the
  address and the given-name address; night 1: the readers' segments). Like 0035's list it carries
  the served and the new names, which the migration must write.
- **Mutants, all killed:** the swap-group drop disabled, the evidence check disabled, the
  both-names check disabled, the renamed source set to the rung's.
- **Suites:** `tests/unit/podcast_scraper/upgrade/ tests/unit/upgrade/ tests/unit/podcast_scraper/quality/`:
  994 passed, 1 failed, the landing test, on the 3 stale outcomes of 0034-0036 only (rebuilt in the
  final pass).
- **Not built:** the verdicts file (the night-1 rows from both readers; D5 needs a fresh blind
  read), the registry entry, the clone measurement (step 7).

## 7. Measure

Gate (`scripts/measure/corpus_change_gate.py`, report `0037_named_voices_corrected.gate.json`):
clone of snapshot-prod-20261027 (ledger 2.7.30, 0034-0036 applied), code d2b95e39d, clean tree,
15:20-15:49Z. Rebased on origin/main first (68706e872 changed three files inside 0037's code
digest), so the measured code is the code that ships.

| Prediction (step 4) | Measured |
|---|---|
| 6 renames, 0 unnamings | `diagnostics_voices_renamed=6`; plan: `d99b4720` SPEAKER_04, D5 SPEAKER_00 + SPEAKER_01, `216929530` SPEAKER_00 + SPEAKER_01, `1732482` SPEAKER_07 |
| 4 episodes written, 0 refused | rewrote 4, refused 0 |
| 0037's verify | OK |
| full `upgrade verify`, 0 new failures | 35 OK; failing only 0025 (the 5 known episodes, the same 5 on the untouched clone); `verify_regressions` [] |
| S7-wrong -6 where the gold covers the voices; S1n not up; other voices 0 | **no scoreboard total moved** (`gating_moves` {}, per-episode deltas 0): the gold does not cover these 6 voices, so the change is "no effect" on the scoreboard, not a measured fall |

Apply counters: segments relabelled 3868, quotes reattributed 333, quotes moved 386, insights
reattributed 149, transcripts re-rendered 8, turns rebuilt 6, context rebuilt 2, KG persons
removed 2 / added 1, GI persons added 1, bridge identities removed 2, admaps carried 3. 50 files
written: `.segments.json` 8, `.turns.json` 6, `.txt` 4, `.adfree.txt` 4, `.cleaned.txt` 4,
`.metadata.json` 4, `.gi.json` 4, `.bridge.json` 4, `.diagnostics.json` 4, `.kg.json` 3,
`.admap.json` 3, `.context.json` 2.

Idempotence and undo on the gate's copy (scratch `rm27/idem_undo37.py`): a second apply rewrote
0 episodes and changed 0 files; `undo` restored 51, refused 0; afterwards 0 files differ from the
snapshot; verify reads 0037 as pending again.

D5's blind read (criterion 14): a fresh reader given name-stripped copies (reader C,
`d5-blindread-20261027` in the eval data repository) confirmed both voices: SPEAKER_00 the feed's
host (addressed, segment 32), SPEAKER_01 the guest (host_read, segment 1). Open point for the
reviewer: the reader notes SPEAKER_01 speaks as the host of that live session; the list gives the
feed's host the host role.


**Per-voice before/after excerpts** (method v0.9 step 7): the 4 affected runs copied from
snapshot-prod-20261027 to a small corpus, 0037 applied there (the same counters as the gate), each
voice's deciding segment and its label/role read before and after. They hold transcript text, so
they live in the eval data repository (not in this repository),
`data/eval/0037-excerpts-20261027/excerpts37.json` (script beside it). By id:

| Episode | Voice | Before (label, role) | After | Other named voices after |
|---|---|---|---|---|
| `d99b4720-…` | SPEAKER_04 | feed host's name, host | the clip's speaker, guest | none |
| `https://iono.fm/e/1732482` | SPEAKER_07 | another guest's name, guest | the interviewee, guest | 3, unchanged |
| `substack:post:165726942` (D5) | SPEAKER_00 | guest's name, guest | feed host's name, host | SPEAKER_01 holds the guest's name |
| `substack:post:165726942` (D5) | SPEAKER_01 | feed host's name, host | guest's name, guest | SPEAKER_00 holds the host's name |
| `substack:post:216929530` | SPEAKER_00 | guest's name, guest | host's name, host | SPEAKER_01 holds the guest's name |
| `substack:post:216929530` | SPEAKER_01 | host's name, host | guest's name, guest | SPEAKER_00 holds the host's name |

**Open call decided (operator, 2026-10-27):** D5's roles stay as listed: the feed's host keeps the
host role and the guest the guest role. It is a one-off ("a guest asking a lot of questions to the
host"): no new rule. Prod serves the same pairing today, only on the wrong voices; 0037 moves each
name, with its role, to the right voice.

**D5's deciding segments, reconciled:** the list's evidence column is the builder's (segment 1:
the other voice says it is talking to the host; segment 14: the host addresses the guest by given
name); the verdicts are the blind reader C's (SPEAKER_00 addressed, segment 32; SPEAKER_01
host_read, segment 1). Both point the same way for both voices; the verdicts file decides, and the
list is not edited (its sha256 is bound by the verdicts and the code digest).

**Builder's verdict: GO.** Deciding number: 6 of 6 listed voices renamed in 4 episodes, 0 refused,
`verify_regressions` [] (gate report). The swaps exchange exactly; no voice holds both names.

## 8. Verdict

Order (operator, 2026-10-27: no GO while `make ci-fast` is red): the builder's GO below, then the
outcome is built and `make ci-fast` runs green on that committed tree, then the §4a verdict read is
launched with the green log; its verdict is the last line here.

A first §4a read (2026-10-27, sonnet subagent, 55 s, 128,701 tokens against a 40k budget) returned
NO-GO on the red ci-fast log (the only failure was this landing test, waiting for this outcome) and
found three record items, fixed in step 7 and below: D5's segments reconciled; the open D5 role call
decided by the operator; the not-covered list written. It also named a missing test: "wrong-person
voices 6 -> 0" has no gate row when the gold does not cover the voices.

Landing condition met: `make ci-fast` on 33f10ca3d (the outcome commit), all stages through
`=== ci-fast DONE`, `CI_FAST_EXIT=0` (18:05Z; test-fast 0 failed); `make test-integration` on
d2b95e39d (the gated code; since then only this outcome file and docs changed in the tree)
`TEST_INTEGRATION_EXIT=0`, 4628 passed. Decision, 2026-10-27: no second verdict read (CI fast has to be green and D5 was agreed, so there
is nothing for another round of review). 0037 is NOT to be applied before the new deploy method.

Verdict: GO (builder, 2026-10-27): 6 of 6 listed voices renamed in 4 episodes, 0 refused,
`verify_regressions` [] (gate report d2b95e39d). Outcome 15cb660599688a2832950123258923b48c674f50c980ba77cd170d344b7d9cba.

## 9. Operator review

Operator, 2026-10-27: D5's roles kept as listed; no second verdict read (ci-fast green is the
landing condition). Approval, 2026-10-28: the go for the apply, after the
deploy that shipped it.

## 10. Before apply / deploy

Deployed (deploy verify PASS); the deploy
ships 0037 and does not run it. Read-only in the deployment after the deploy (2026-10-28):

- `upgrade status`: 2.7.30, pending exactly `0037_named_voices_corrected`.
- `upgrade verify`: every applied migration OK except 0025 on its 5 known episodes.
- Dry run 15:36:35Z, preflight PASS (76 GB free, nothing busy, nightly 504
  min away, pending exactly 0037): "would rewrite 4 episode(s)" with every counter equal to the
  gate's line on the snapshot-prod-20261027 clone (segments_relabelled 3868, quotes_reattributed
  333, insights_reattributed 149, diagnostics_voices_renamed 6, ...); no `refused=` counter (a
  refusal is always printed as one, `m0029._scan`); "outcome hook: plan diff 0 episode(s)".

## 11. After apply / deploy

Applied 2026-10-28 15:46:14Z on approval:

- "[applied] 0037_named_voices_corrected: rewrote 4 episode(s)", the same counters as the dry run
  and the gate; "Now at version 2.7.31; 0 pending." EXIT=0.
- Pre-upgrade snapshot `/app/.corpus-upgrade-backup-2.7.30-20261028T154633Z` (in the api container's
  writable layer: a container recreate deletes it); undo recorded: `m0037 undo(Path('/app/output'))`.
- Verify 16:18Z: 0037 OK ("every rename the review confirmed carries the voice's own person's
  name"); the only FAIL is 0025's 5 known episodes.
- Ownership: "0 files written since the apply and not owned by the corpus owner", OWNERSHIP PASS.
- Derived surfaces: `enrich --only insight_density,insight_sentiment` (no --force) 16:19Z:
  insight_density 4 ok / 0 failed, insight_sentiment 4 ok / 0 failed, EXIT=0; incremental
  `index` (no --rebuild) 16:20-16:28Z: "scanned=3023 skipped=3019 reindexed=4 vectors=1186", 2
  stale rows of superseded runs pruned, EXIT=0. Corpus-scope enrichers left to the nightly.
- Served outcome, read by the builder through the MCP episode_speaker_roster tool (2026-10-28) for D5: SPEAKER_00 = the feed's host, host, 1942.4 s (62.9%); SPEAKER_01 = the
  guest, guest, 1145.8 s. The other 3 episodes were not read one by one.

## 12. Retrospective

- The forced-host-seat veto built for D5's cause was rejected on its own reach replay (52 fires,
  mostly real hosts introducing themselves); 0037 repaired the served episode without it.
- The first step-8 "review" re-derived the whole gate on its own clone for over an hour; method
  v0.9 section 4a (a review reads, 5 minutes) came out of it.
- A gate row counting a list's wrong-person voices before / after is missing (see Not covered).

## Not covered

- The scoreboard cannot show the fix: the gold covers none of the 6 voices, so "wrong-person voices
  6 -> 0" rests on the plan (`diagnostics_voices_renamed=6`) and the excerpts, not on a gate row. A
  gate row counting a list's wrong-person voices before and after is missing (any change, not only
  0037).
- The causes stay in the pipeline: a forced `known_hosts` seat on the wrong voice (D5, `d99b4720`;
  the veto that addressed D5 was rejected), the LLM's pair swap (`216929530`), the LLM's guest name
  on a clip voice (`1732482`). New episodes can repeat them.
- Other wrong-person voices outside these 4 episodes: the night-1 read sampled 21 episodes; the rest
  of the corpus was not read.
- The search-index rows of the 4 episodes: refreshed by prod's incremental index after the apply,
  not by the migration.
- D5's speakers-diagnostics `resolution_attribution` block still describes the original naming run
  (the LLM's names); 0037 does not rewrite that history block, so it contradicts the served roster.
  Internal diagnostics only; not what the roster serves.
- The served rosters of `d99b4720`, `1732482` and `216929530` were not read one by one after the
  apply (the counts and verify match).
