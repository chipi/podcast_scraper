# Handover: what is still to fix or think about, after the 0034 / 0035 / 0036 release (2026-10-27)

For the next agent on ingestion and naming work. Each item says what was seen, where the evidence is,
and what is open. None of these is being worked on now. **Updated at the end of 2026-10-27**: items
marked FIXED landed on main that day; section E is what the day's validation pass added; start
there. Ids only. The release records are `docs/wip/changes/0034_talk_floor_names_removed.md`,
`0035_voice_identity_names.md`, `0036_bridge_identities_restored.md`.

## A. Corpus migrations: defects in how they are built or run

1. **A naming migration keeps the stored ad cut, but the cut depends on the labels.** 0034 and
   0035 re-render the raw transcript with new speaker names and carry the stored ad map. The cut
   (`workflow/adfree_transcript.py`, `build_adfree_artifacts`) reads the labelled screenplay:
   `Name:` markers, same-speaker turns coalescing, and the opening host-read cross-promo detector.
   On the deployment after 0035, 0030's verify flagged 2 episodes (`6244ca0c-cdda-11f1-be2a-4fdc8347f8b3`,
   `f25e9982-86ff-42be-8e99-d04396268448`): naming the second voice as the host let today's cut
   find two host-read sponsor blocks the stored cut never removed (one a 4,289-char mid-roll).
   The live pipeline does recompute the cut after a relabel (`episode_processor.py`,
   `_produce_transcript_sidecars` on the relabel path); only migrations skip it. Fix on the deployment
   (2026-10-27, approved per step): re-run 0030's own apply after 0036. **Open:** a naming
   migration that re-renders text should recompute the cut as the pipeline does, or end with 0030's
   resync for the episodes it wrote.
2. **FIXED: clone measurements ran each migration's own verify, never the full `upgrade verify`.**
   `corpus_change_gate.py` now runs every applied migration's verify on the copy and reports
   `verify_all` and `verify_regressions` (commit "gate: full upgrade verify on the clone"); method
   step 7 requires it. Item 1's own fix (recompute the cut in naming migrations) is still open.
3. **The code digest covers the whole migrations package.** An import-order or docstring change in
   one migration made all three outcomes stale (2026-10-26: three extra gate runs). **Open:**
   whether the digest should be per migration plus what it imports.
4. **No way to re-run an applied migration for named episodes.** The 0030 re-run has to call the
   class from Python inside the api container. **Open:** an `upgrade rerun <id> --episode ...` path
   with its own receipt.
5. **FIXED (ledger, lock, outcome receipts and plan-diff now chowned to the corpus owner, commit
   "upgrade: ledger and outcome receipts stay the corpus owner's"). Was:** files left root-owned after each apply on the deployment: `upgrade_ledger.json` and
   `.podcast_scraper/upgrade-outcomes/<id>.json`, chowned by hand after 0034 and 0035. The runner
   writes these outside the corpus-owner path (`match_corpus_owner`, `upgrade/ownership.py`, is
   called by `write_with_backup` in `upgrade/file_rewrite.py` for rewritten files only). **Open:** fix in the runner.
6. **FIXED (same commit: `record_reverted` rewinds the version when the reverted migration is the
   last one and nothing has to be guessed). Was:** an undo removes the ledger entry but not the version. After an undo of 0035 the version
   stays 2.7.29 and is reset by hand (recorded in the 0035 record and the post-deploy plan).
7. **`write_with_backup` keeps the first backup.** A migration that writes a file twice (0030
   re-run) keeps the pre-first-run copy, so an undo would also drop what later migrations wrote.
8. **Every `upgrade run` copies the whole corpus (~30 GB) into compose-api-1's writable layer**
   (not a mount; a container recreate deletes it). Two runs with snapshots do not fit the disk. On
   2026-10-27 only the latest snapshot was kept, and rollback relies on each migration's own undo
   plus the daily off-host corpus backup. **Open:** where whole-corpus snapshots should live.
9. **PARTLY FIXED (`test_shipped_verdicts_complete.py`: one row per list entry, each decided row with
   its segment and a 64-char hash; a deleted kept row now fails). Still open:** nothing in the committed tests guards the verdicts' hash chain. The round-5 and round-6
   reviewers' mutants survived the tests: an altered `line_sha256`, a changed `segment_idx`, a
   deleted kept row. Only the outcome bound to the verdicts sha catches them. **Open:** a test that
   checks each row's `line_sha256` against the served segment text.

## B. Naming rules and data: what the release leaves as it is

10. **Two spellings of one person: 0035 versus 0025.** 0035 gives a voice its feed-wide fingerprint
    group's spelling; 0025 prefers the spelling the episode's own title or description states.
    After 0035, 0025's verify flags 5 episodes for that alone (one person id each; accepted as
    known, listed in the post-deploy plan item 18.4). In at least 2 of the 3 people
    the feed's spelling is the right one. **Open:** a spelling-precedence rule both share.
11. **ASR misspellings of known names stay served.** Readers marked 10-17 rows per batch
    `CORRECT_MISSPELLED` from their own knowledge of the person, not from the text; they were
    written as `CORRECT`. **Open:** a spelling repair with text evidence (title, description,
    another episode).
12. **The diarizer merges promo / ad / clip speech into a host's voice.** Many of the 0034 rounds 5-6 removals
    (basis `merged_voices`) were a promo, clip or ad inside another person's voice (not counted
    across all 175). The talk floor only reaches voices
    under 5%. **Open:** an upstream fix (diarization or ad-aware splitting), or a rule for the
    voices above the floor.
13. **Mixed-voice majority is measured in seconds** (decided 4 rows, 2026-10-26). It
    lives in the 0034 record. **Open:** write it into `docs/guides/CORPUS_CHANGE_METHOD.md` so a
    future reader rule set uses it.
14. **The cross-episode recurrence scan has blind spots** (`m0034` round 6): produced spots whose
    wording varies (a presenter role recurs by format, not text), lines under 40 characters,
    recurrence across feeds, voices that appear once, interview-structure cases. 38 promo
    self-introductions were not re-read.
15. **12 unpublishable KG-only persons** (garbled mention names) are left out of the bridge by 0036.
    A KG-side repair was proposed, never approved (0036 record, section 7).
16. **0035's open reviewer points, accepted:** 45 `undecidable` rows ship without a
    deciding segment; 3 thin-chain renames; 13 episodes with one name on two voices (a solo host
    split by the diarizer).
17. **The CIL readers read each feed's latest run only** (405 of 3,504 bridges on the snapshot), so
    a fix in an older run's bridge is invisible to them.

## C. Local development hazards on the dev machine

18. **The `podcast_scraper-0029` worktree's venv imports the main checkout's code in child
    processes** (`__editable__` `.pth` points at the main checkout's `src`).
    Export `PYTHONPATH=<worktree>/src` for `make test-integration` and `make ci-fast`, and delete
    the cached fixture search index if it was built without it.
19. **No full corpus copies in `/private/tmp` on the dev machine** (2026-10-27): the volume also
    holds the container VM's disk, and it fell under 5% free.

## D. Already tracked elsewhere

- Naming gaps: #2352; D37: #2353; non-English ASR: #2187; model pick: #2251.
- Later naming rounds (plan items 23-25): the "stated host answered as guest" refusal, two co-hosts
  typed as guests, the resolver prompt's evidence line (needs a GPU-host measurement).
- Shadow evaluation: its own handover, not in this repository.

## E. Added by the 2026-10-27 validation pass (read this first)

20. **0037 (`named_voices_corrected`) is LANDED on the deployment** (applied 2026-10-28 15:46Z with
    approval, after the deploy; corpus 2.7.31, 0 pending; verify OK; ownership
    PASS; enrich 4/4 and incremental index reindexed 4). Six wrong-person voices in four episodes
    (#2094 D5 and night 1's blind reads), a swap applied whole; the deployment's dry run equalled the clone
    gate counter for counter. Record `docs/wip/changes/0037_named_voices_corrected.md` (steps 1-12).
    D5's roles stay as listed (a one-off, no rule).
21. **The forced-host-seat veto was rejected and reverted.** It would have refused a feed's host seat
    on a voice that names the host in the third person (D5's cause). Replayed over 2545 served
    episodes it fired on 52 voices, most of them real hosts introducing themselves in forms the
    resolver does not read ("it's <host> with <show>", "your host, <host>", "<host> here"). Record:
    the change record `known_host_third_person_veto` (status rejected, no longer in the tree). **Open:** a second attempt starts
    at step 3, with those 52 as its test set.
22. **0032 was retired** without running (assessment in git history): 67 of its
    unnamings contradict 0034's review. **Its 91 uncovered voices** (48 additions, 29 renames, 14
    unnamings) become their own reviewed list, AFTER the shadow evaluation: tracked in the shadow
    handover, not here.
23. **D4's 43 candidates: 0 wrong names** (classification in git history). The big
    unnamed voices beside them are hidden insights, a coverage question for its own record.
24. **`d99b4720-283b-4f07-89c4-791b12acba2d`'s cause is unaddressed:** a feed host's name forced on a
    clip voice that never mentions the host. 0037 fixes the served episode only.
25. **A gate row is missing:** "wrong-person voices before / after" for a list the gold does not
    cover. 0037's fix of 6 voices moved no scoreboard number (the step-8 read named it).
26. **What a review is: method v0.9, section 4a** (written by a fresh advisor on request after a step-8 "review" re-ran the full verify twice for over an hour). A review
    reads, produces no numbers, 5 minutes; recomputing is the tests' job. **Open:** the first read
    under it used 128,701 tokens against a 40k budget in 55 s: narrow the inputs, or a launcher with
    a budget; the method says nothing enforces it.
27. **Epic #2097's #2075 measurements, re-derived** (measurements in git history):
    151 of 3006 served episodes with no named speaker; the detector returns 72% of form-named
    guests; the golden set 10 of 11 (`hardfork-guest-in-record` fails, cause not established).
    Inputs to closure decisions, not closures.
28. **Closed by the deploy:** `player-digest-scheduler` now reports to the error tracker (06b3601c1).

## F. Later fixes: decided after the rollout to the new infrastructure

Decision, 2026-10-28: nothing below starts before the pending infrastructure move and the other
pending changes are done; then each one is decided. None is started. Monitoring and the shadow
evaluation are in the shadow-evaluation handover, not in this repository.

| # | Fix | What it would take | Status |
|---|---|---|---|
| F1 | The forced host seat on the wrong voice (D5's cause; item 21) | a second attempt from step 3: a signal that tells an interviewer naming the host from the host naming themselves, measured first over the 52 replay cases | not planned |
| F2 | A host name forced on a clip voice (`d99b4720`'s cause; item 24) | diagnosis by loss on how many served episodes have it, then a pipeline rule | not planned |
| F3 | D4's big unnamed voices (item 23) | a coverage change record: are they the small voice's person split by the diarizer, or someone else | not planned |
| F4 | #2304: 289 episodes with no Moments, 230 because naming resolved nobody | overlaps F3 and the #2075 count (151 served episodes with no named speaker) | open issue, not part of this arc |
| F5 | The gate row "wrong-person voices before / after" for lists the gold does not cover (item 25) | a gate row in `corpus_change_gate.py`, a test | not built |
| F6 | Review budget enforcement (item 26) | narrower review inputs or a launcher with a token and time budget | not built |
| F7 | The golden set's `hardfork-guest-in-record` failure (item 27) | find whether the snapshot predates a fix (a `record` case) or the code regressed | cause not established |
| F8 | Item 1's open half: a naming migration that re-renders text recomputes the ad cut | in the shared rewrite, or 0030's resync at the end of each such migration | open |
| F9 | Item 9's open half: a test of each verdict row's `line_sha256` against the served segment | a unit test over the shipped verdicts | open |
| F10 | D5's diagnostics `resolution_attribution` still describes the original naming run | decide whether migrations rewrite that history block or readers ignore it | open, internal only |

Decisions, from measurements already made: whether to close #2075, #2093 and #2095
(measurements in git history).

## NOT covered by this handover

- Epic #2097 was not re-read.
- Items 3, 4, 5 and 8 are design questions; no measurement of their cost was made.
- Whether the deployment's 0030 re-run and 0036 completed is not checked here.
- The step-5 and earlier naming-arc notes were not merged into this list.
