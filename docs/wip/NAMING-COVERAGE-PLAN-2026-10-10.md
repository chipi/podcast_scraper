# Naming coverage plan — episodes that publish nobody (2026-10-10)

Source: #2304, plus a read-only replay on prod on 2026-10-10. Scope is naming only: Moments and the GI
value gate are out of scope. Every prod-changing step needs the operator's go, one at a time.

## Why this was missed

- **Migrations m0017–m0025 only rewrite names that exist.** They rename, merge, unify or remove. None
  of them adds a name to a voice that never had one.
- **The checks only measured CHANGE.** The voice replay, the m0025 parity check and the dev/val gates
  all compare old vs new. An episode unnamed under both counts as "agree", so it is invisible. Nothing
  measured absolute coverage.

## Baseline (prod, 2026-10-10, English episodes with GI)

There are **296 episodes that publish no named voice** (the issue counted 230 with a narrower test).

| Group | Episodes | What it means |
|---|---|---|
| No speaker diagnostics (older record format) | 96 | Cannot be replayed; only a re-run can tell. |
| Replay names the main voice (host pool rebuilt) | 24 | Today's code already fixes it; data repair only. |
| Replay names some voice, not the main one | 57 | Partial. |
| Replay names nobody, but hosts are known | 69 | Biggest code gap (e.g. The Rest Is History: Tom Holland and Dominic Sandbrook are known, 0 placed in 44 eps). |
| Replay names nobody, no host known | 49 | Host discovery gap (e.g. a16z). |
| No segments | 1 | Ingest. |

**Caveat:** the replay cannot run the LLM naming step (no stored verdicts on these episodes), nor
bylines or per-feed config. The real result under today's code can be better than the replay. The
size of that gap is unknown.

## Segments, in order

Each segment follows the same loop:

1. Set a goal on dev_v1 (the 101-episode development set).
2. Root-cause it from naming traces on real cases.
3. Fix it with unit tests.
4. Gate it on dev, then on val_v1 (500 episodes, scored once), and replay it on the prod corpus. The
   replay must report absolute coverage, not just changes.
5. Commit with the numbers, then push with the operator's OK.

At most 2 loops per segment.

| # | Segment | Size | Done when |
|---|---|---|---|
| S0 | **Coverage metric.** A committed scanner that counts episodes and insight-bearing voices with no name, by group. Wire it into `naming_gate.py` and `roster_replay.py`, so every naming change reports absolute coverage as well as diffs. | tool | Reproduces the table above. Committed. |
| S1 | **Hosts known, never placed.** Why a known host (feed pool / show.json) is not attached to any voice when there is no self-intro or address. Candidates: talk-share / first-voice host prior; recurrence of the same voice signature across a feed's episodes. | 69 (+ part of the 96) | Most of the 69 placed on replay; 0 regressions on labelled hosts in dev/val. |
| S2 | **No host known.** Host discovery: show.json evidence, feed-level recurrence, RSS author tags, episode description patterns ("with host X"). | 49 | Hosts discovered for most; no junk names admitted (name census). |
| S3 | **Guest known, not placed.** A metadata / title guest (`placed:false`) where the transcript addresses them ("Hi, Michael"). Partial-name address matching. | ~62 (issue) | Placed on replay; no wrong placements on dev/val. |
| S4 | **Main voice unnamed while others are named.** | 57 | Main voice named where evidence exists. |
| S5 | **The 96 without diagnostics.** Find out what re-running naming costs on them (`relabel_only`: no audio or ASR, LLM on the DGX). Trial 1–2 episodes, then measure. | 96 | Cost measured; the operator decides. |
| S6 | **Live naming bugs seen on recent ingests.** No Priors guest seated as host; "Alexander Carpe" + "Alexandra Karppi" as two hosts; "Paul Hoffman SC" (title kept); Hard Fork "Max Reed" host; Hard Fork Laporte; the In Our Time interviewer. | ~6 cases | Each fixed or explained; gates green. |
| S7 | **Data repair on prod.** After S1–S4 ship: replay-proven list of episodes whose names change, then a relabel batch. The DGX is needed, so the operator decides. Batches like the #2187 backfill. | all fixed eps | Coverage on prod re-measured with S0. |
| S8 | **Guardrail.** The S0 coverage number goes into the nightly checks and the morning review, with an alert if it rises. | — | Alert live. |

Excluded (not naming): Moments; the GI value gate (all-tier-0); ungrounded insights routing "surface";
the untimed direct_download transcript.

## Working order: easier to harder (operator, 2026-10-10)

1. **S0** coverage metric. Prerequisite: every later step reports against it.
2. **S8** guardrail. Cheap once S0 exists.
3. **S6a** title kept in a name ("Paul Hoffman SC"): a small rule.
4. **S3** guest known, not placed, addressed by name.
5. **S1** hosts known, never placed.
6. **S4** main voice unnamed while others are named.
7. **S6b** the remaining live bugs (Alexander/Alexandra, guest seated as host, Hard Fork, In Our Time).
8. **S2** no host known (discovery).
9. **S5** the 96 without diagnostics (cost trial).
10. **S7** prod data repair (operator decides).

## Progress log

- 2026-10-10: plan written; baseline above from `unnamed_replay*.py` (scratch, to be committed in S0).
- S0 DONE (local commit f87076ca3, not pushed): scripts/measure/naming_coverage.py reproduces the baseline
  exactly on prod (296 / 2,601; 70.8% of quotes named). Runbook section added.
- S8 MOVED LATER: not cheap. scripts/ is not in the api image, so the guardrail has to be a pipeline- or
  API-side metric plus an alert, not a cron around this script.
- S6a DONE (local commit aee17d79e): "Paul Hoffman SC". canonical_person_name strips a capitalised trailing
  credential (+ SC/QC). Prod effect: 1 of 9,185 names changes; that 1 episode needs a data fix after deploy.
- NEXT: S3 (guest known, not placed, addressed by name).
- Pushed f87076ca3 + aee17d79e (operator OK); CI running.
- Trace analysis (prod, read-only; 175 replayable unnamed eps: hosts_known + no_host + names_some), why the
  MAIN voice stays unnamed:
  - guest_naming "none_left": 107. The voice is seated as a guest/unknown and no guest name is known -> S2/S4
    (discovery).
  - host_naming unnamed: 49. The forced host name needs ONE spare name + ONE unnamed seat; abstains otherwise.
    The Rest Is History (31 eps): 2 spare names (Tom Holland, Dominic Sandbrook) for 2 seats -> abstains by
    design. The docstring says the two-name assignment by who-addresses-whom measured 6 right / 3 swapped /
    26 undecidable. Single-host cases (Anish Acharya, Justin Norman, Tim Scarfe x4) still unnamed -> check
    the seat_owns_talk / presenter_elsewhere / feed-history gates per case.
  - guest_naming "says_that_name_in_the_third_person": 15. E.g. the Michael Montgomery ep: the host opens
    "Hi, Michael", and the forced guest name is refused because the guest says "Dr. Montgomery" (his own domain
    name) -> S3 candidate rule: an address of the first name by the other voice at open/close outweighs a
    surname-only third-person mention.
- S3 (third-person refusal of a hailed guest) DONE, local commit (see git log). Replay: 7 gained / 0 lost.
  dev +2 / val +2, 0 regressions. Coverage names_main 24 -> 25.
- S1 findings (trace of all 69 replay_hosts_known eps):
  - 41 The Rest Is History (+4 more with 2 seats): NOT a naming bug. The diarizer MERGED Tom Holland and
    Dominic Sandbrook into ONE voice (80-85% of talk; the merged voice says "so true tom" to itself);
    the other voice is the reading/actor clips. Naming the merged voice either name is half wrong ->
    needs a re-diarization (rediarize_only, GPU) = operator decision, S5/S7 territory.
  - 15 The Flip: 1 name (Justin Norman), 1 seat, refused by `seat_owns_the_talk`. That gate was written
    for the MLST cold-open guest at 75%. The Flip is a documentary where the narrator-host talks 57-68%
    among 3-4 interviewees. A rule that separates the two needs labelled examples. NEXT once the nightly
    is done.
  - 3 a16z: performs_guest_act veto (Angela Strange, Theo Jaffee x2) -> read the cases.
  - 2 presenter_elsewhere (Ben Horowitz, Anish Acharya); 2 with 0 seats (Dwarkesh, David Haber).
- 23:29 UTC: paused prod-container replays while the nightly runs (they are CPU/RAM heavy on the 15 GB box).
- 64654a8c9 (S3) PUSHED 2026-10-10 ~23:35 (operator OK). RESUME after the nightly: 1) nightly review (ingest, errors, speed, dup edges, verify, coverage via naming_coverage.py), 2) S1 The Flip seat_owns_talk, then the a16z/Dwarkesh cases, 3) S4. Backfill continues too (78 broken-to-end + 47 partial), DGX permitting.
- S1 The Flip DONE (local commit, see git log): 8 gained / 0 lost; gates neutral.

## Segment L: the naming ladder as one input/output component (operator, 2026-10-11)

Problem: unit tests pin single rules; the fixture golden pins current output (change, not truth); the
gold sets only cover the episodes they happen to contain. Nothing asserts "an episode like this
must come out named like this".

1. Harness: run the whole naming stage, from guest/host detection out of the feed and episode metadata
   to the roster and the speaker record. In: feed metadata, episode title/description, diarized
   anonymous turns (+ shares), optional LLM verdicts. Out: speaker record + trace.
2. Scenario library, one file per scenario, with EXPECTED-CORRECT output. Seeds:
   - Michael Montgomery, The Flip, Varoufakis, The Rest Is History merged hosts (must stay unnamed),
     Dwarkesh lecture (must not be "Dwarkesh Patel"), Marc Rowan (guest only in the title),
     Paul Hoffman SC;
   - Ken Beck, ChinaTalk "Hey, Jordan", MLST cold open, Hard Fork, In Our Time;
   - format classes: solo, two hosts, panel, documentary, lecture, guest in description only, merged
     cluster, ad voices.
   Known-failing scenarios are marked xfail(strict) until their fix lands.
3. Every naming fix from here is test-first: scenario red -> fix -> green. The library and the dev/val
   gates stay green, and coverage must not drop.
- Segment L harness DONE (local commit 5af569bf6): 13 scenarios, 12 pass + 2 strict xfail (Dwarkesh LLM host verdict; Marc Rowan title guest). NEXT: fix those two test-first.

## Failure inventory (prod, read-only, 2026-10-11) — before any more fixing

| # | Failure class | Count on prod | Example | Scenario |
|---|---|---|---|---|
| F1 | Episode publishes no named voice | 296 (of 2,606 English with GI) | see groups above | coverage metric |
| F2 | Guest name only in the title, never reaches the record | 75 of the 296 have a title name ("Name — topic" 36, "Name on topic" 26, "topic — Name" 8, "topic with Name" 5); only 16 of those names are in the record at all | a16z "Marc Rowan on …" | `guest_only_in_title_on_topic` (xfail) |
| F3 | LLM "feed host" verdict accepted on the titled guest's lecture voice | Dwarkesh (Sarah Paine lectures, 3+); full count to measure | Dwarkesh | `lecture_episode_guest_in_title` (xfail) |
| F4 | Guest seated as HOST (role error): a placed host whose name is in the title and is not a known host of the feed | 41 (heuristic; split 2026-10-11, see F4 replay below) | The Pragmatic Engineer: served record has Kelsey Hightower host from `llm_resolution` (the LLM's role accepted, same root as F3); Peter Mattis host from `self_intro` on a merged cluster (F7) | CORRECTED 2026-10-11: I wrote here that today's code fixes Kelsey and the record was just old. FALSE: the replay with the real record's inputs still seats Kelsey as host on main. `llm_host_role_for_the_titled_guest` passes only on cleaner inputs than the real episode; the real shape is `host_welcome_opens_the_guest_cluster` (xfail) |
| F4 replay | 2026-10-11 replay of the served records with main's roster (`/tmp/nc/f4_replay.json`, `f4_classify.json`; the heuristic now selects 49, not the 41 above: the 41 query was not kept) | 49 = 17 already right on today's code (old records: MLST 3, Mattis, Dwarkesh 2, No Priors Tuhin, a16z 8, Tyler 2) + 32 still host. main == deployed on all 49: S1/S3 do not touch F4 | see sub-rows | — |
| F4a | The LLM put the GUEST's name on the HOST's voice (wrong name AND role; the real guest voice is then refused `name_already_on_another_voice`) | 10 | Studebaker, Tse, Warwick, Verna, Zelikman, Tiwari, Allebest, Besse (voice says "your host, Professor Hannah Fry"), Ioffe ("chatting with ... Julia Yoffe"), Lawrence ("with me luke robert mason"). Cause: `llm_merge` (roster.py ~5222) accepts the model's name with no check of the voice's own words. `refuted_by_third_person` would catch 1 of 10 (exact spelling only): ASR respellings, "Peter C.", second-person address, a pool host's self-intro | to add |
| F4b1 | Guest voice performs a "host act" by IDIOM: `_HOST_SPEECH_ACTS` pattern 0 matches "you're welcome to share", "welcome to the club", "welcome to Moldova", "listeners are welcome to" | 8 | Brook, Berlina, Ian Cameron, Hitchens, Walden Yan, Mark Williams, Maia Sandu, N.T. Wright (pattern 5 "thank you all very much for being here", a lecturer to the audience; found on the roster's NORMALIZED voice text — my first per-case scan used the raw text and missed it) | `guest_saying_welcome_to_my_country_is_not_a_host`, `guest_saying_you_are_welcome_to_share_is_not_a_host` (xfail) |
| F4b2 | Host's line BLED into the guest cluster at a turn boundary ("Welcome to the show. Thank you for having me"); host acts are checked BEFORE guest acts (hosts.py 1682), so the guest's own guest act is ignored | 10 | Dionnet, Wahlforss, Helberg, Zach Dell, Laskin, Ken Burns, Markovits (panel), Carina Hong, Burazin, Kelsey Hightower (91% merged) | to add |
| F4c | Empty pool + LLM "host" verdict seats the guest (`5_llm_host_no_pool`) | 2 | Sharon Blackie ("Thank you for inviting me" is not a guest act), David Shapiro (no speech act either way) | `no_host_pool_llm_host_on_a_thanking_guest` (xfail) |
| F4 ok | Correct as host | 2 | Iza Ding (stated guest host), Alec Hogg (BizNews presenter) | — |
| F4 new | Host lost after the guest is right: Pragmatic Engineer/Mattis, the host voice (28%, reads the bio) is now an unnamed GUEST | 1 seen | stated_non_host covers both voices | to add |
| F11 | One stated name on TWO voices after the non-regression restore: the LLM puts stated guest B on voice 1, the cue path had B on voice 2, the LLM pass drops voice 2's name and the contract restores it (scenario run 2026-10-11, harness log 'restored 1 deterministic name') | to measure; prod has same-name pairs (Odd Lots 'Miles Brundage' llm+self_intro, 'Scott Glenn', 'Darrell Duffie' forced+llm) | synthetic | to add |
| F5 | One person placed twice under two spellings (and the host loses the host role) | 6 | "Casey Newton" + "Casey Noon" (Hard Fork); "Kenneth Rogan" + "Kenneth Rogoff"; "Natalie Kitroeff" + "Kittroweth"; "Nicola" + "Nicky Clayton" | `host_second_cluster_with_asr_spelling` (xfail; also shows the main voice demoted to guest) |
| F6 | Two different people merged by spelling (wrong unification risk) | READ 2026-10-11: 0 confirmed. Both suspects are the REVERSE (F5): the host's own self-intro under an ASR spelling published as a separate GUEST | Odd Lots: "Jill Weisenthal" (Posen, 82 s), "Jill Wisenthal" (Iranian economy 34 s; Lenovo CFO 834 s), "Jill" ("And I'm Jill. Why isn't thal Joe."), "Joeisenthal" (Bostrom); Planet Money "Ciara Gonzalez" ("I'm Ciara Gonzalez", host Sarah Gonzalez). Odd Lots Milanovic: a 69 s voice saying "I'm Joe Wisenthal and I'm Tracy Alloway" is published as forced guest "Professor Milanovitch" | F5 fix must still keep two different people apart: a same-surname, different-given-name pair is not automatically one person (a scenario for that guard is needed with the F5 fix) |
| F7 | Diarizer merged two hosts into one voice | The Rest Is History 45 | RIH | `two_hosts_merged_into_one_voice` (passes: stays unnamed) |
| F8 | Feed host statement not detected | CHECKED 2026-10-11 on all 80 prod feeds: 19 detect no host; 17 of them state no person at all; 2 real wordings, 1 served episode each | People I (Mostly) Admire: 'Freakonomics co-author Steve Levitt tracks down…' (no voice named Levitt); Unexplainable: 'The Unexplainable team — Noam Hassenfeld, Julia Longoria, Byrd Pinkerton, and Meradith Hoddinott — tackles…' (presenter published as ASR 'Bird Pinkerton'). 'A and B take on history': 0 feeds. Latent Space lists INTERVIEWEES in parentheses: must stay non-hosts | `feed_states_its_host_as_title_then_name` (xfail), `feed_states_its_team_as_a_dash_list` (xfail), guard `feed_lists_interviewees_not_hosts` (passes) |
| F10 | All-lowercase transcript | 42 of 2,602 served episodes (15+ feeds: Colombia Calling 6, The Rest Is History 5, Empire 5, FUTURES 4…). Self-intro names per episode 0.45 vs 0.90 on cased transcripts, 'introduced' 0 vs 0.02; any voice named 86% vs 92% | Names STATED somewhere are still read case-blind; a lowercase self-intro with nothing stating the name names nobody (documented limit: case is the only name signal left) | guard `lowercase_transcript_with_stated_names` (passes) |
| F9 | Live cases from the 10-10 nightly | No Priors guest as host; Talk Eastern Europe "Alexander Carpe" + "Alexandra Karppi"; Hard Fork "Max Reed" host | — | to add (F4/F5) |

Harness gaps: CLOSED 2026-10-11. Co-presence history is fed via `feed_history` (scenarios `second_host_seated_when_the_feed_history_shows_both`, `no_phantom_second_host_on_a_one_presenter_feed`). Both earlier 'gap' reasons were WRONG: the address-pair scenario failed because its second host first spoke outside the intro window; the ad scenario because its sponsor read was too weak for the pre-roll detector (3 pattern hits) and its episode too short for the edge-ad share (<3%). All 34 documented rungs are now reached by a scenario (`NOT_YET_EXERCISED` empty); `non_regression` reached only on its `not_restored` branch.
Not a bug after measurement: the ad narrator candidate (0 on prod).

- 2026-10-11 F4b1 CODED (local, not committed until gates): idioms + `over_unstated_introduction`. Corpus replay (2,606 eps): 9 changes, all read, all right. A first guard (refuse every unstated introduced name) lost 59 names, many correct, and was REVERTED.
- 2026-10-11 F4b1 COMMITTED 0b5523630 (local): idioms + `over_unstated_introduction`; replay 9 changes all right; gates 0/0.
- 2026-10-11 F4c COMMITTED 96373d2f9 (local): `_INVITED_REPLY` only in step 5; replay 1 change (Blackie) right; gates dev +1 / 0 regressions. First version (general guest act) reverted: 2 hosts un-named by bleed.
- NEXT: F4b2 (host act vs guest act at a turn boundary, reuse the 200-char bleed pair), then F4a, F5, F8, F3, F2.
- 2026-10-11 F5 measured on prod (published names near an UNPLACED feed host, 18): host variants "Alexander Carpi" x3 / "Alexandra Karpi" (Alexandra Karppi), "Ina Panikova" (Nina), "Brennan Leung"/"Bernard Leung" (Bernard Leong), "John Whelan", "Jill Wisenthal"; different people that must stay apart: James Holland (Tom's brother), Jonathan/Rachel Abrams, Ryan Felton/Knutson. An edit-distance rule (each part <= 2, total <= 3) was tried and REVERTED: it missed Carpi (total 4) and merged "Kevin Ross" into Kevin Roose (the documented guard) and "Daniel Smith" into "Daniela Smyth". Needs another signal (the name said inside the show's own intro formula). F5a (an introduced guest name that snaps to a pool host is not a spare guest, RJ Hanaki) is separate and in test.
- 2026-10-11 branch fix/naming-coverage-2304: F4b1 0b5523630, F4c 96373d2f9, F4b2 c2a77dd33, F5a a4a08af99, F4a 2ca64b59f (all replayed with --signatures, gates 0 regressions except F4b2's RJ case, cancelled by F5a).
- F12 NEW: an undetected mid-roll sponsor read right after "my guest is X" takes the introduced name (Lenny's Lovable episode: "Here's a puzzle for you. What do OpenAI, Cursor... have in common?" named Elena Verna, 4%; her 75% voice unnamed). To measure + fix.
- F4a still open: "My guest today is Peter C.", "chatting with, in person, Julia Yoffe", lowercase self-intro.
- 2026-10-11 ~15:40 branch adds: F8 9805f1637 (+golden d9e11d912), index prune fix 076cf0f2f (NOT naming: the #2158 guard refused 30 correct prunes since 10-03; 28 episodes on prod serve stale insight rows until repaired, needs operator go). F2 in test: "I'm delighted/very happy to be chatting with[, in person,] X" cue (Tyler's opening) — fixes the Ioffe F4a case too. F2 population measured: ~60-70 served episodes where the titled guest owns the talk and is named nowhere (Tyler ~12, a16z ~15, Rest Is Politics ~8, Turkey Book Talk 8, Dwarkesh 4 + 7 F3 "Dwarkesh Patel" on the guest).
- 2026-10-11 F3 attempt REVERTED (not committed): dropping the model's pool-host name on the voice that owns the conversation (`_owns_the_conversation`, no host act, stated guest unplaced) fixed Dwarkesh (Jensen Huang, Sarah Paine) but changed 54 voices, stripping real hosts on host-dominant shows (Eric Topol, Lenny, Robert Armstrong, Shane Parrish, Jordan Schneider, Patrick McKenzie, Luigi Zingales, Jeremy Au, Dwarkesh's solo "Thoughts on AI progress") and reshuffling a Sinica panel. Next design: per-FEED host talk-share history (what share the evidence-named host holds in the feed's other episodes; Dwarkesh is a minority voice, Topol a majority one), served like co-presence from sibling sidecars.
- F3 per-feed history checked and NOT viable: Dwarkesh/Lenny/ChinaTalk/BRAVE/Tyler have no evidence-named host on any served episode (no history, rule would abstain where needed); Ground Truths' median host share 0.35 would still strip Eric Topol. F3 needs voice identity across episodes (diarization speaker embeddings: the host's voice recurs) — compute/DGX design decision for the operator.
- 2026-10-11 20:30 branch fix/naming-coverage-2304 (pushed at 2ca…→ rebased tip before F12; later commits local): + F12 356a4ed93, F2-title + junk names da0b34a1a; in test: harvest junk guard + title helper refactor + F11 one-voice-per-introduced-person. PR #2328 opened by mistake and closed (operator: no PR); its CI runs cancelled.
- Still open: F3 (needs cross-episode voice identity, operator decision), F5 rest (Jill/Carpi/Ina: needs show-intro-formula signal; edit-distance rule reverted), F4a rest ("Peter C.", lowercase), F2 rest (title-only guests the metadata does not state, e.g. Marc Rowan), S7 record repair, S8 alert, stale index rows on 28 episodes (operator go after deploy).
- 2026-10-11 late, branch: F2 description-stated title guest 14a881508, F5 host-formula snap 0e03bfcc4, harvest junk guard c9bd64c99 (all pushed 2026-10-12 with da0b34a1a).

## Status 2026-10-12 — what the branch moved, and a better target

**Branch vs main** (prod replay, 2,604 eps, same snapshot): nameless episodes 236 -> 232, named voices
+36, 108 voices changed (every one read: corrections of wrong names/roles, guests named), 0 episodes lost
every name. Gold gates main vs branch: dev better 6 / worse 0, val better 14 / worse 0. The 296 baseline
and the 236 are different measurements (published records vs a replay of main after 9 backfill batches);
how much of 296 -> 236 is naming vs reprocessing is NOT known.

**Today (2026-10-12), all replay-proven, gates 0 worse, mutants killed, NOT pushed:**
- 55e9bce6f F2c: the guest the title ENDS with ("[Dr. Jeff Beck]", "with Steve McConnell"), vouched by
  metadata or the host's first-name greeting; host seated on evidence, <= half the talk, not a short form
  of a carried name; harvest snaps to the title spelling. 232 -> 226, +17 voices; val +4, dev +1. The first
  version swapped MLST host/guest and put a name on an ad read; those guards came from that replay.
- 4bda9ff85: a guest talking about someone sharing their surname ("Blaise Pascal") keeps their forced
  name. 226 -> 225, +8 voices; val +4, dev +1.
- 47a622b74: an unpublishable self-intro ("I'm Apple just announced") yields to the voice's own
  publishable one ("I'm Ryan Knutson"). +2 voices. A drop-instead-of-replace version was NOT shippable
  (lost Kenny Malone, took the guest role from ~12 real guests) and was not committed.
- REJECTED (not committed): letting a host seat that performs the host role keep its name while it owns
  the conversation — all 8 changed voices were guests or merged clusters.

**Why the nameless-episode count stopped being the target.** Of the 226 nameless (replay, 2026-10-12):
96 have no voices at all (publisher transcripts, never diarized — this is the S5 group); 30 are The Rest
Is History (F7); of the remaining ~100, 42 have one voice at >= 70% (merged clusters), 28 state nobody and
nobody self-introduces, ~30 are reachable by ladder code. Operator 2026-10-12: the 96 and the 30 wait until
everything else is processed.

**New measurement: the failure atlas** (gold labels, per voice, HEAD 47a622b74; `atlas.py`, scratch ->
to commit). val_v1 (500 eps): correct 643, missing 411, wrong 100, spurious 17, role error 18.
- wrong 100: **43 are swaps** (the person's own name sits on another voice) — worst for a listener.
- missing 411: 140 the name is stated or proposed (known host / metadata / LLM / self-intro) but not
  placed (72 hosts, 68 guests); 33 the name is only in the title; 238 the name is nowhere in metadata,
  title or transcript (out of reach for the ladder: S2 discovery or the model).
- dev_v1 (101 eps): missing 74, wrong 26 (12 swaps).

**Working order from here (operator OK 2026-10-12):** 1) the 43 swaps, 2) the 140 stated-but-unplaced,
3) the 33 title-only. Same loop (test-first, replay all served eps, every change read, gates, mutants).
S0 nameless count stays as the guardrail, not the target. Then the 96 (S5) and the 30 (F7).
- 2026-10-12 SWAPS MEASURED, NOT FIXABLE BY RULES (nothing committed). 55 swap voices on dev+val (43 val);
  38 from the model's verdict (`llm_resolution`) on co-hosts / two-guest panels. Address evidence (vocatives,
  `_vocative_count`) over all named pairs in both gold sets (411 correct, 19 swapped): a "swap when address
  says so" rule flips more correct pairs than it fixes at every threshold (>=2/3x: 8 correct broken vs 5
  fixed; >=4/3x: 4 vs 1). Abstaining instead: in every evidence bucket of the 339 model-named voices the
  correct names outnumber the wrong (worst bucket "voice says its own first name, nobody hails it": 26 right
  / 23 wrong). Real fixes are F3-class (cross-episode voice identity) or the model's prompt/input:
  operator decisions. Operator asked about the coin-flip trade (unname that bucket); my advice: leave it.
- NEXT: the 140 stated-but-unplaced voices (72 hosts, 68 guests) on val.
- 2026-10-12 STATED-BUT-UNPLACED (172 dev+val) broken down: ~27 are a host's SECOND cluster (the name is on
  the host's main voice; needs voice identity, F3-class); 23 are 0.2-2.4% edge intro clips typed as ads
  (Capitalisn't recorded opener; no listener value, skipped); 27 metadata-only guests — a first-name
  greeting corroborates only 1 of 27.
- PARKED FOR LATER (operator 2026-10-12): "metadata-stated guest owns the talk". One spare metadata person
  (2+ words, not a host, not placed) and exactly one unnamed non-host voice holding > 50% of the talk ->
  that voice is the person. Gold (dev+val): 27 right / 1 wrong (BRAVE, guest-hosted); at >= 15% share it was
  32 / 14 (the wrong ones are hosts' second clusters). It NARROWS a design that was built and REVERTED
  (roster.py ~3120 "NOT DONE: ... drawing its candidate from metadata_named": admitted 5 people never in
  the room, incl. a man who died in 1956; pinned by test_a_stated_name_we_failed_to_place_is_our_DEFECT,
  Qasar Younis). If picked up: add a corroboration floor (surname spoken in the episode), replay all served
  eps reading every new name for "did this person speak", the Qasar test stays binding, commit message says
  it narrows a reverted decision. Eval script: /tmp/nc/m1_eval.py on prod.
- NEXT: the 33 title-only guests.
- 2026-10-12 TITLE-ONLY (44 dev+val): 2bed7f4c3 F2d (full name said + first name ADDRESSED by an evidence host;
  "ft." tails). Replay: 6 guests named, 0 lost; val +3 / 0 worse. Left: 9 never said, 12 only by first name,
  5 dropped by the address requirement on purpose (Alan Waxman, Josh Kushner, Klarman, Milanovic, Varun Mohan).
- Backfill: operator standing approval 2026-10-12 to auto-approve each next batch (scratchpad/batch_chain.sh).
