# Handover: speaker naming and identity, after D2 (2026-11-03)

For the next agent on naming, identity and corpus-migration work. It replaces the 2026-10-27
naming-debt handover (git history keeps it). Items that landed or were closed since are dropped.
Ids only. The plan is RFC-131 (plan V3 in the eval-data repo); the change records are in
`docs/wip/changes/`.

## 1. Where it stands

**D2 is on prod** (`sha-8906696`, main `890669690`; the operator accepted it on 2026-11-03).
- **Ledger:** D2's rules (s5c, s5d, s5o, s6a–s6h), 644 joins, 40 blocked.
- **Migrations 0039–0042:** applied in four runs, each through the outcome hook (verdict PASS).
  - 0039 and 0040 equal the clone.
  - 0041 and 0042 were applied with accepted plan diffs: 14 and 1 entries, every cause verified.
  - The accept files are on prod under `.podcast_scraper/upgrade-outcomes/`.
- **After the apply:** enrich and the incremental index were run, then the bounded verify of
  0039–0042 (all OK).
- **Rollback:** each migration's own receipts and backups under `.podcast_scraper/upgrade-backups/`,
  plus the daily snapshots. The pre-apply snapshot and the rollback image holds were removed on the
  operator's go.
- **Records:** d2_name_consolidation (the outcome table), 0039–0042 (steps 1–8), s5d–s6h.

## 2. Next steps, in order

1. **Close D2's records.**
   - Write steps 10 and 11 (the prod dry runs, the plan diffs and their causes, the applies, the
     verifies) into 0039–0042 and d2_name_consolidation.
   - Record d2_verify_bounded's prod result.
   - Set the four migration records to `status: landed`.
2. **Infra:** tell the infra session D2 is on prod. PR #2405 (server module renames) waits for it.
   The D2 file list was sent on 2026-11-02.
3. **The reviewed list:** add the 11 near-spelling pairs prod's report listed on 2026-11-03, all
   read as one person, to `upgrade/lists/s5d_reviewed_fuzzy_joins.json`, and update the list test's
   counts. The report reads 0 on prod after the next deploy carries the list.
   - The pairs: `asime-nite`/`asime-nute`, `willem-els`/`willem-als`, `luis-guerra`/`louis-guerra`,
     `robert-caro`/`robert-carrow`, `bobby-kennedy`/`robert-kennedy`, `nina-panikova`/`ina-panikova`,
     `davi-alcolumbre`/`davi-alcolumbri`.
   - Plus the episode-scoped bare surnames joined to their full name: `robin-hanson`,
     `aleksandar-vucic` (and `alexander-vucic`), `noam-chomsky` (4 ids), `nosa-laulu`.
4. **Bug: `person_profile` is empty for some speakers**, for example `person:robert-caro` and
   `person:anita-anand`.
   - The search shows grounded quotes with their `speaker_id`, and `person:dwarkesh-patel` returns
     16.
   - It is empty on the D1 snapshot too, so D2 did not cause it.
   - `server/cil_queries.person_profile` finds episodes through each bridge's GI ids; the cause is
     in that lookup.
   - Repro first.
5. **Bug: root-owned backup directories.** m0023's `_write_text_with_backup`, which 0041 uses for
   transcript text, creates backup parent directories and does not pass them to
   `match_corpus_owner`; `write_with_backup` does. On prod, 0041 left 5 root-owned directories,
   since chowned. Fix with a test. It moves the code digest of the migrations that import it, so
   it lands with the next migration batch.
6. **The feed-host spelling rule (deferred, not refused).**
   - `person:dara-obrey`'s 12 dev_v1 gold insights are an accepted D2 loss. The group is all ASR
     spellings of the feed's stated co-host.
   - Fixing it needs a rule that matches a feed's stated host to a group by first name and the host
     role, not by letters.
   - It would also decide `person:luisa-lemi` and `person:john-glenn-hill`.
7. **`known_hosts` is mixed evidence** (record s6h, 4b): correct self-introductions, ASR spellings
   from the recurrence scan (`workflow/stages/processing._recurrent_hosts_from_disk`) and parse
   artifacts. The joins, the display rule, and m0041's and m0025's stated-name reads take it as
   stated. Fix at the pipeline.
8. **Strict-tier losses D1 → D2 not addressed:** `sarah-gonzalez`-shaped (16 gold insights),
   `erika-barris` (8), `david-sanger` (8). These are majority or publisher picks (record s6h,
   Not covered).
9. **Bare names in a show,** pooled again only with voice-fingerprint evidence, after the
   fingerprint shadow validation (record s5c).

## 3. Lessons from D2's run, for the next deploy runbook

- **`index-two-tier` is a full reindex** (`drop_existing=True`). The incremental reindex after a
  migration is `index --output-dir <root>` without `--rebuild`, as the nightly runs it.
- **One regime migration per `upgrade run`:** the CLI refuses a second one whose plan is not
  empty. Each run snapshots the corpus (~30 GB) unless the job says `"snapshot": false`. The skill
  supports it since 2026-11-02.
- **The ledger refresh after `upgrade run` and `enrich` respells** (`identity/refresh_guard`). It
  defers the respelling until 0041 is applied. `enrich-edges` is not guarded (it is inside the
  migrations' digest), so no `enrich-edges` pass may run between a deploy and a respelling
  migration.
- **A clone gate does not run `upgrade run`'s path** (the hook, the refresh). Read that path before
  writing the runbook.
- **Prod's plan differs from the clone's** wherever prod has newer episodes or a different ledger.
  Expect `--accept-plan-diff` and budget time to verify each cause from the rows and the snapshot.
- **The prod-deploy preflight counts a path-filtered workflow as not run.** A commit that changes
  only web or tests skips Docker Build & Test and CodeQL, and needs the operator's waiver.
- **CI runners are pinned to `ubuntu-24.04`** (2026-11-02): `ubuntu-latest` moved to 26.04, which
  has no prebuilt Python 3.11.8. Revisit the interpreter pin there.
- **Run a gate on committed code:** `build_outcome.py` and the outcome hook refuse a gate measured
  on a dirty tree.
- **My measuring scripts skip episodes whose metadata stores an absolute transcript path**
  (after_checks, the first gold-move script). Resolve the transcript by name in the run's
  transcripts folder.

## 4. Open from before D2 (carried over)

**Corpus migrations: how they are built or run**
- A naming migration that re-renders text keeps the stored ad cut, though the cut depends on the
  labels. It should recompute the cut as the pipeline does, or end with 0030's resync.
- The code digest covers a migration and everything it imports, and that is most of the package.
  Any shared edit makes outcomes stale.
- No way to re-run an applied migration for named episodes. Today it means calling the class from
  Python in the api container.
- `write_with_backup` keeps the first backup. A file written twice would be undone to its
  pre-first-run state.
- Nothing in the committed tests checks each verdict row's `line_sha256` against the served
  segment.
- A gate row "wrong-person voices before / after", for lists the gold does not cover.

**Naming rules and data**
- ASR misspellings of names with no publisher text or second spelling stay served. They need text
  evidence.
- The diarizer merges promo, ad and clip speech into a host's voice. The talk floor only reaches
  voices under 5%.
- Mixed-voice majority is measured in seconds (decided 2026-10-26, in record 0034). Write it into
  the method guide.
- The cross-episode recurrence scan's blind spots (record 0034, round 6).
- 12 unpublishable KG-only persons left out of the bridge by 0036. A KG-side repair was never
  approved.
- 0035's accepted reviewer points: 45 undecidable rows with no deciding segment, 3 thin-chain
  renames, 13 episodes with one name on two voices.
- The CIL readers read each feed's latest run only, so a fix in an older run's bridge is invisible
  to them.
- The forced host seat on the wrong voice (D5's cause): the third-person veto was rejected. A second
  attempt starts at step 3, with its 52 replay cases as the test set.
- A feed host's name forced on a clip voice (`d99b4720-283b-4f07-89c4-791b12acba2d`'s cause).
- D4's big unnamed voices: a coverage question for its own record.
- 0032's 91 uncovered voices become their own reviewed list, after the shadow evaluation.
- The golden set's `hardfork-guest-in-record` failure: cause not established.

**Method**
- Review budgets (method v0.9 §4a) are not enforced. The first read under it used 128,701 tokens
  against a 40k budget.

**Local development on the dev machine**
- A worktree's venv can import the main checkout's code in child processes. Export
  `PYTHONPATH=<worktree>/src` for `make test-integration` and `make ci-fast`.
- No full corpus copies in `/private/tmp`: that volume also holds the container VM's disk.

**Tracked elsewhere:** naming gaps #2352; D37 #2353; non-English ASR #2187; model pick #2251;
#2304 (episodes with no Moments); the shadow evaluation has its own handover.

## Not covered by this handover

- The player UI was not opened after D2. Prod's spot checks went through the API.
- Which rule step lets `person:robert-carrow` join `person:robert-caro` on prod but not on the
  clone (5 vs 6 episodes) is not traced.
- 0042's 3 extra resynced quotes on prod are not traced per quote (accepted).
- Epic #2097 was not re-read for this handover.
