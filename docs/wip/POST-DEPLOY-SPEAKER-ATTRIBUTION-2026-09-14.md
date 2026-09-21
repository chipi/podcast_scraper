# Post-deploy: repairing host/guest attribution on production

**Context:** PR #2065 fixes host/guest attribution at cause. This note is what to run on prod
*after* that deploys, and what each step can and cannot fix.

**Written 2026-09-14.** Numbers below are measured on a 287-artifact production staging copy, not
on the live corpus. Re-measure before trusting the counts.

---

## Why the deploy is not the end of it

Two different things are broken and they need different repairs.

- **The pipeline** was computing the wrong answer. PR #2065 fixes it — but only for episodes
  ingested *from now on*.
- **The artifacts already on disk** have the wrong answer stored. Code cannot reach back and
  change them.

And the stored artifacts are broken in two distinct ways, which is why there are two repair
passes and not one:

| | what is wrong on disk | repair | cost |
|---|---|---|---|
| **A** | the roster in `content.speakers` is CORRECT; the graph never received it | migration m0009 | no LLM, no GPU, minutes |
| **B** | the roster is **itself wrong** (`['Host']`, an org, a show name) | `pipeline_stage=relabel_only` | LLM for the GI/KG cascade; no ASR, no GPU diarization, no audio download |

A migration cannot fix class B. There is no correct answer on disk for it to copy.

---

## Step 0a — FIRST: measure the idx-collision damage (#2082)

**Operator decision 2026-09-16: the order is deploy -> REPAIR #2082 -> quality work.** Not deploy
-> quality. Every quality metric here reads `content.speakers`, and on an affected episode that
field describes a different episode — so quality measured before the repair is partly measuring the
wrong episodes, and a quality change cannot be told apart from the contamination. Measurements
already taken during the #2075 arc should be re-derived after repair, or restricted to episodes this
audit clears.


**Run this before anything else, and do not run a migration until you have the number.** It costs
one read-only pass and it decides whether the rest of this document is measuring a corpus or a
corrupted one.

On a reprocess, `idx` used to come from the episode's filename prefix. Every run directory numbers
from `0001`, so a feed with fourteen run dirs has fourteen "episode 1"s, and `idx` keys per-episode
state and output filenames — so they collided. Re-measured on `snapshot-prod-20260920`
(2026-09-21), which is the number to quote:

```
metadata files pointing at ANOTHER episode's transcript   147 of 2,297  (6.4%)
  of those, roster matches the WRONG transcript             119
  roster matches its own                                      0
  unclassified — NOT proven safe                             28

repairable in place (own transcript still on disk)           42
needs re-download + re-ASR + re-diarize                     105
```

**The 275 / 146 / 233 this section used to print were wrong**, and anything scoped off them — GPU
estimates, repair batches, #2082's headline — was scoped off a number 87% too large. A plain stem
equality test called 128 episodes mispaired that are not: the metadata filename truncates the
title and the transcript filename does not, so `0006 - This Funding Model is Helping Fi_<guid>`
and `0006 - This Funding Model is Helping Fight Climate Change_<guid>` compared unequal. The
corrected test (`utils.filesystem.names_the_same_episode`, shared with the pipeline since
2026-09-21) rescues those 128 and **not one** of the real mismatches.

119 episodes credit their quotes to people from a different episode — "DHH's new way of writing
code" is credited to Addy Osmani; "How Kent Beck shapes the software engineering industry" to Grady
Booch. Those names reach `content.speakers`, the KG `Person` nodes, `SPOKEN_BY` and search.

```bash
make transcript-pairing-audit CORPUS_DIR=<prod corpus>
```

Read-only. **Exits 1 when any mismatch is found**, so it can gate this step. It reports three
verdicts, because they need different work:

| verdict | meaning |
| --- | --- |
| `roster_matches_wrong` | the speakers are in the transcript it points AT and not its own — **confirmed misattribution** |
| `roster_matches_own` | the pointer is stale but the roster is right |
| `inconclusive` | names in neither or both — usually a roster built from metadata. **NOT proven safe**, just unclassified |

and two repair routes: `repairable` (own transcript still on disk — scoped `relabel_only`, no
audio, no GPU) and the rest (needs re-download + re-ASR + re-diarize).

Do not quote the confirmed count as a total until the unclassified ones have been looked at.

**Take this number BEFORE Step 2, and do not compare it verdict-by-verdict afterwards.** m0010
rewrites `content.speakers` in `.metadata.json`, and this audit classifies by matching roster
names against transcript labels — so after Step 2 some episodes move toward `inconclusive` without
anything having been repaired or broken. The *count of mismatched pointers* stays comparable; the
verdict split does not.

**`exits 0` is not reachable by repair alone.** The audit exits 1 on ANY finding, and the 105 that
need a re-ingest stay findings until they are re-ingested. Treat "42 fewer findings after Step 1b"
as the success criterion, not a clean exit.

Why this comes first:

1. **Every measurement below is taken from `content.speakers`.** On an affected episode that field
   describes a different episode, so the coherence before/after number, the migration previews and
   the "roles look like this afterwards" tables are all reading corrupted rows until this is known.
2. **m0009 promotes and demotes roles from those same rosters.** Running it first bakes the wrong
   attribution deeper and makes the damage harder to see.
3. **The 42 repairable ones are cheap** — a scoped `relabel_only`, no audio, no GPU. Do them before
   the migrations so the migrations see the corrected rows.
4. **The 105 are not cheap** and are a separate decision: their transcript does not exist anywhere
   in the corpus, because `relabel_only` overwrites the transcript it picks, so where episode A was
   relabelled onto B's file, A's was never written and B's was overwritten with A's labels. Only a
   full re-ingest recovers them. Size that GPU bill deliberately; do not let it start by accident.

The code fix (unique per-run `idx`, `Episode.on_disk_idx` kept only for the legacy search) is on
`fix/duplicate-people-variant-resolution` and stops NEW damage. It repairs nothing already written.

## Step 1 — Deploy

Images publish only from `main`, and one tag pins all three services. This fixes every future
ingest and changes nothing already stored.

**PIN THE SHA — do not let the deploy choose.** Publishing is `stack-test.yml`'s `publish` job
(not `docker.yml`, which builds with `push: false`), gated on `refs/heads/main` **and stack-test
having succeeded**. `deploy-prod` with an empty `override_image_sha` deploys the newest published
`sha-<7>` — so if stack-test is red or still running, it silently ships the PREVIOUS image and
every step after this runs the old code. Step 1b in particular would then re-inflict the damage it
is there to repair.

```bash
# 1. wait for: python-app -> Stack test -> publish -> verify-manifests
# 2. take sha-<7> from the publish run summary
# 3. dispatch deploy-prod with override_image_sha=<that sha>
# 4. confirm on the box, do not assume:
grep PODCAST_IMAGE_TAG /srv/podcast-scraper/.env
docker ps --format '{{.Image}}'
```

## Step 1b — REPAIR the cheap half of #2082, AFTER the deploy and BEFORE any migration

Step 0a only MEASURES. The repair is a separate step, and this file used to have no repair step
at all — it went 0a → 0 → 1 → 2, which reads as "measure the damage, then migrate on top of it".

**It goes AFTER the deploy, and that is not a detail.** The repair runs pipeline code, and the fix
that makes it safe (below) ships in this deploy. Run it against the old image and it re-inflicts
the damage. An earlier draft of this section numbered it 0b, i.e. before Step 1 — wrong.

**The command.** There is no per-episode parameter on `POST /api/jobs` (`routes/jobs.py` takes
`feed`, `max_episodes`, `episode_offset`, `episode_order`, `episode_selection`, `profile`,
`pipeline_stage` — an earlier draft of this file invented `&episode=<id>`). Per-episode scoping is
`--reprocess-episode-ids <file>`, which **Reprocess prod corpus** reaches via:

```
selection        : episode_ids_worklist
episode_ids      : <paste the 42 ids, or leave empty to use the file on the box>
pipeline_stage   : relabel_only        # added 2026-09-21; without it this workflow only did
                                       # a full re-download + re-ASR, which is not the cheap route
use_transcript_cache : false
cost_cap_usd     : <state it — a relabel has no ASR cost but the GI/KG cascade does>
```

Get the ids from the audit itself — `--worklist <path>` writes the **repairable** ones only, so
the 105 that need a re-ingest cannot ride along in a cheap batch:

```bash
# via Inspect prod corpus -> checks: transcript_pairing_audit, or on the box:
python scripts/audit/transcript_pairing_audit.py --corpus-dir /app/output --worklist /app/output/pairing_repair_worklist.txt
```

**What the fix changed, and what to watch.** `_transcript_beside_metadata` used to try the stored
`content.transcript_file_path` FIRST. On all 147 damaged records that pointer names another
episode's transcript — which exists, with its `.segments.json` — so a relabel of A opened B's
file, overwrote it with names re-resolved from B's words, and rewrote B's roster
(`_rewrite_speaker_record_in_place` derived the record from the transcript's stem). The repair
re-inflicted the damage and left A untouched. Now the episode's own sibling wins, the pointer is
accepted only when it names the same episode, and a relabel repairs the pointer as it goes.

Watch each `reprocess: [seq] (on-disk idx) '<title>' -> <path>` line: the path's stem must be the
episode's own. Then re-run the audit — it should report exactly 42 fewer findings.

**Why it cannot wait.** m0009 reads `content.speakers` and demotes a graph node the roster
contradicts. On an idx-collided episode that roster describes a DIFFERENT episode, so the
migration demotes the people who actually spoke. Measured on `snapshot-prod-20260920`, the
step-0 preview's three `guest -> mentioned` transitions — the direction step 4 says to stop on —
are exactly this:

| node demoted | roster on disk | why |
| --- | --- | --- |
| Krishna Rao (*"Anthropic's CFO"*) | `Sam Altman` | metadata points at `0001 - Sam Altman….txt` |
| Brian Chesky (*"AI Founder Mode"*) | `Matthew Smith` | points at `0002 - Matthew Smith….txt` |
| Bruce Lanphear (*Ground Truths*) | `Professor Bruce Lanphier` | not #2082 — a name-matching gap, fixed 2026-09-21 |

Both #2082 cases are in the audit's findings, so step 0b removes them. Do the 42 first; the 105
that need a re-ingest are a GPU decision (activity 5), not a blocker for the migrations.


## Step 1c — Pre-flight (was "Step 0")

**Numbered after the repair deliberately.** Every measurement here reads `content.speakers`, and
on a #2082 episode that field describes a different episode — so a BEFORE number taken ahead of
Step 1b is partly measuring the wrong episodes, and the AFTER/BEFORE comparison at Step 4 cannot
tell a quality change from the contamination. The m0002 precondition check below is read-only and
can be run at any time; run it early, because it decides whether Step 2 is minutes or a rebuild.


```bash
make upgrade-status CORPUS_DIR=<prod corpus>
make speaker-migration-preview CORPUS_DIR=<prod corpus>
make speaker-coherence CORPUS_DIR=<prod corpus>          # record the BEFORE number
```

### CHECK m0002's THREE PRECONDITIONS FIRST — it is migration 2 of 10, and the chain stops at it

`upgrade run` **stops at the first failing migration**, so anything wrong at m0002 means 0007–0010
never run. m0002 is a no-op only when ALL THREE hold (`m0002_two_tier_native_reindex.py:57-59`):

1. `<corpus>/search/lance_index` exists,
2. it is not schema-stale (`lance_index_is_stale` — an `index_meta.json` version below the code's),
3. the `<corpus>/search/metadata.json` offset sidecar is present (#1010).

Miss any one and it runs a **full native index build** — which needs `sentence_transformers`, so on
a machine without the ML extras it aborts the whole chain. Verified on 2026-09-21 against the live
prod API: `corpus_status` returns `index.present: true`, and semantic search returns results (a
stale index makes read paths report `no_index`), so **1 and 2 hold on prod**. **3 was NOT verified
from any read-only surface** — check it on the box before step 2:

Two ways to settle it, both read-only:

```bash
# On the box:
ls -l <prod corpus>/search/metadata.json      # present => m0002 no-ops; missing => full reindex
```

Or dispatch **Inspect prod corpus (read-only)** with `checks: upgrade_dry_run` (added 2026-09-21,
so it must be on `main` before it can be dispatched). It prints that `ls` AND the whole Step 0
plan from the prod image, including m0002's own words for which branch it will take:

```
"LanceDB index already present — no-op."                    <- safe, step 2 is minutes
"...missing the metadata.json offset sidecar ... reindex"   <- full index rebuild
"...schema-stale — rebuild natively."                       <- full index rebuild
```

That option exists because Step 0 had **no prod execution path at all** — `make upgrade-status` and
the migration preview were being asserted against production rather than measured on it.

A local drill on a restored backup CANNOT tell you this: `make restore-corpus-prod` prunes
`search/` as regenerable, so m0002 always tries to build there. On 2026-09-21 that is exactly how
the chain died at migration 2 of 10, leaving a ledger with only `0001` recorded.

**EXPECT ALL TEN MIGRATIONS PENDING, not two.** An earlier version of this file said "confirm
pending is exactly [0008, 0009]". That is wrong: production carries **no `upgrade_ledger.json` at
all**, so `upgrade status` reports `Applied: none` and all of them pending, at version
`2.7.0.dev0`. (**NINE until 2026-09-21**, when `0010_canonical_person_names` was added — see its
row below.)
The artifacts are nonetheless already modern — kg schema `2.0`, gi schema `3.1`, m0007 scoped ids
present — because the pipeline writes those natively and re-ingestion satisfied the migrations
without recording them. Rehearsed against the real `snapshot-prod-20260914` corpus (2,257
artifacts) on 2026-09-14.

What each pending migration actually does on that corpus:

| migration | effect |
| --- | --- |
| 0001, 0004 | no-op |
| 0002 | **no-op on prod** — skips when a healthy index with its sidecar exists. It reports a build locally only because the backup deliberately prunes `search/` as regenerable |
| 0003, 0005, 0006 | genuine no-ops — 2,257 already-current |
| **0007** | **rewrites 153 episodes** (18 healed, 255 scoped) — see the numbers below, this is not a formality |
| **0008** | **rewrites all 2,257 artifacts** for a one-character change — see below |
| **0009** | 3,177 promoted, 195 demoted, 1,844 artifacts |
| **0010** | **rewrites 93 episodes** — 39 person ids remapped (33 of them MERGING into an id already in the same episode), 1 duplicate node id folded, 164 published names canonicalised (`snapshot-prod-20260920`; the 2026-09-14 snapshot gave 91/162). Idempotent on re-run, `verify` returns ok, 0 unparsable. **It also REISSUES m0009's role ledger** — 94 rows re-pointed, 32 episodes deliberately left unrestorable; see Rollback, and expect `upgrade-undo-roles` to exit 2. **Its numbers were measured on the UNREPAIRED corpus and inherit the Step 0a caveat below.** |

## Step 2 — Migrate (fixes class A)

### Before you start: pause the queue, and pick the window

Confirm no ingest job is running (`GET /api/jobs`) first: a mid-ingest snapshot is inconsistent.
`GET /api/jobs` is the check that matters, **not the lock** — the corpus lock covers a multi-feed
ingest, but a PER-FEED job locks `<root>/feeds/<slug>` instead, a different file. Per-feed is the
shape Step 1b and Step 3 use, so an `upgrade run` started during one is not refused. Never rely on
contention to protect you here.

```bash
touch /srv/podcast-scraper/corpus/.viewer/jobs.paused   # scheduler queues instead of failing
# ... Steps 2-4 ...
rm /srv/podcast-scraper/corpus/.viewer/jobs.paused      # and re-trigger anything that was skipped
```

**Window: 07:00–02:00 UTC.** The nightly ingest fires at 03:00 UTC and the corpus backup at 05:37.
The `prod-corpus` concurrency group serialises *workflows*, and an SSH-driven `upgrade run` is not
a workflow — so a backup running over a migration gets `file changed as we read it`.

### The snapshot is the ONLY rollback for 0007 and 0008 — mount it or lose it

`make upgrade-corpus` is the local form. **On prod, do not run it without `--snapshot-dir` on a
host-mounted path**, and here is why that is not a style note:

* `_snapshot_corpus` defaults its destination to `corpus_root.parent` (`cli_handlers.py:171`);
* the corpus is a bind-mounted volume at `/app/output` (`stack.yml:132`, repointed by
  `prod.yml`), so that default is `/app/…` — **the container's own writable layer, not the bind**;
* step 2 runs under `docker compose run --rm`, which deletes the container on exit.

So the default writes a 4.2 GB snapshot inside the container, prints
`rollback = replace the corpus with this dir if the upgrade fails`, and then throws it away. The
operator is told they hold a rollback that does not outlive the command that made it.

**And m0009's role ledger does not cover the rest.** 0007 rewrites 153 episodes and 0008 rewrites
all 2,265; `upgrade verify` reports "no verification defined" for both and there is no undo for
either. The snapshot is the only way back for them.

```bash
# On the box. SNAP must be on a filesystem with >= the corpus size free (4.2 GB on the
# 2026-09-20 snapshot) — the `upgrade_dry_run` inspection reports `du -sh corpus` and `df -h`.
cd /srv/podcast-scraper
SNAP=/srv/podcast-scraper/upgrade-snapshots          # host path, OUTSIDE the corpus dir
mkdir -p "$SNAP" && df -h "$SNAP" | tail -1

SEC=''; [ -n "$(ls -A /dev/shm/podcast-secrets 2>/dev/null)" ] && SEC='-f compose/docker-compose.secrets.yml'
docker compose --env-file .env \
  -f compose/docker-compose.stack.yml \
  -f compose/docker-compose.prod.yml \
  -f compose/docker-compose.vps-prod.yml \
  $SEC \
  run --rm --no-deps \
  -v "$SNAP":/mnt/upgrade-snapshots \
  --entrypoint python api \
  -m podcast_scraper.cli upgrade run \
    --corpus-dir /app/output \
    --snapshot-dir /mnt/upgrade-snapshots \
    --yes 2>&1 | tee "/srv/podcast-scraper/logs/upgrade-$(date -u +%Y%m%dT%H%M%SZ).log"
```

**Capture the output.** m0009's SUSPECT/AMBIGUOUS lists and m0010's "left stale" lines are printed
once and are the only record of what needed a human; a terminal that scrolls is not a record.

### If it dies part-way: what each migration leaves behind

`upgrade run` records each migration as it COMPLETES, so the one that raised is not recorded and a
re-run resumes there. That makes most of the chain simply re-runnable. The exceptions are the
point of this table.

| dies at | what is on disk | can you just re-run? | recovery |
| --- | --- | --- | --- |
| 0001 | nothing | yes | — |
| **0002** | ledger `{0001}`, nothing rewritten | **no, not until the index has its sidecar** | run `reindex-prod.yml` (writes `search/metadata.json`), re-dispatch `upgrade_dry_run`, then re-run |
| 0003 / 0005 / 0006 | per-file atomic, already-current skipped | yes | re-run |
| 0004 | no-op unless the index is stale | as 0002 | as 0002 |
| **0007 / 0008** | per-file atomic, idempotent | yes | re-run — but there is **no undo and no verify** for these two, so the snapshot is the only way back |
| **0009** | per-episode write, then ledger append | yes, it converges | a re-run gets a NEW `run_id`, and the undo defaults to the newest — so a later rollback is partial unless you pass `--run-id` per run |
| **0010** | gi → kg → meta per episode; the ledger resync runs ONCE at the end | files yes, ledger **no** | see below |

**m0010 is the one with a real hole.** If it dies before the resync, every episode it already
rewrote carries a stale m0009 hash — and the re-run only resyncs the episodes IT rewrites, so
those rows stay stale permanently: `upgrade verify` will fail 0009 and the undo will refuse them
forever. There is no repair short of restoring the snapshot. If m0010 crashes, **restore rather
than re-run**, unless you are content to lose the role rollback for that slice.

Not verified: whether a crash *between* m0010's three per-episode writes (gi, kg, metadata) leaves
a trio the re-run converges. Assume it does not.

One misleading message to ignore: `cli_handlers` catches `RuntimeError` and prints "Upgrade
refused — the corpus is in use". No migration raises `RuntimeError` today, so if you ever see that
text from a migration failure, the corpus is *not* necessarily in use — read the traceback.

### Restart the API after Step 2 — and it is about file OWNERSHIP, not just caches

`--entrypoint python` bypasses `docker/api/entrypoint.sh`, which is where the privilege drop
happens, and the api image sets no `USER`. So **the migrations write as root**: every file they
rewrite via `os.replace` becomes `root:root`, including m0010's `.metadata.json` rewrites. Step 3
writes `metadata.json` in place as uid `podcast` and will hit `PermissionError` on exactly those
files. The restart re-runs the entrypoint's `chown -R podcast:podcast`, which is what repairs it:

```bash
docker compose --env-file .env -f compose/docker-compose.stack.yml \
  -f compose/docker-compose.prod.yml -f compose/docker-compose.vps-prod.yml \
  restart api viewer
```

**`restart`, never `--force-recreate`** — recreating without re-staged secrets kills the api
(DEPLOY_GOTCHAS §1b). The snapshot directory on the host is root-owned too, so removing it later
needs sudo.

**Check the snapshot exists on the HOST before trusting it** — the line the CLI prints is a
container path:

```bash
ls -ld "$SNAP"/.corpus-upgrade-backup-* && du -sh "$SNAP"/.corpus-upgrade-backup-*
```

If the snapshot cannot be written the run aborts before mutating anything (`_snapshot_corpus`
returns `None` and the caller stops), which is the safe direction — but it is a failed deploy
step, not a warning, so check the free space first.

**"4.2 GB" is the PRUNED backup, not the live corpus.** The backup drops `search/` and the audio;
the box measured 48 GB in 2026-08. Size this from `du -sh corpus` in the dry-run output, not from
this page.

### How to actually restore from it — UNTESTED, write it down before you need it

The runbook said "replace the corpus with this dir" and gave no command, which is not a procedure.
This is the shape; **nobody has run it**, so rehearse it on a copy before trusting it:

```bash
touch /srv/podcast-scraper/corpus/.viewer/jobs.paused
docker compose --env-file .env -f compose/docker-compose.stack.yml \
  -f compose/docker-compose.prod.yml -f compose/docker-compose.vps-prod.yml stop api viewer
# --delete, but NOT the job registry: .viewer/ is live state, not corpus content.
rsync -a --delete --exclude '.viewer/' \
  "$SNAP"/.corpus-upgrade-backup-*/ /srv/podcast-scraper/corpus/
chown -R 1000:1000 /srv/podcast-scraper/corpus      # the snapshot is root-owned; see the restart note
docker compose ... start api viewer
```

### 0008 is the BIGGEST write and does the LEAST work — expect it

It stamps `schema_version` `2.0` -> `2.1` and changes nothing else. Readers already accept both
(`kg/schema.py`: `_ACCEPTED_SCHEMA_VERSIONS = {"2.0", "2.1"}`), so nothing is gated on it and it
fixes no violations. It is still correct to run — the stamp is what makes the corpus honestly say
it can hold `Object` nodes (#2057) — but know two things before watching it:

- it **rewrites every artifact in the corpus** (2,257) for a one-character change, which is real
  I/O and a full-corpus churn in the next backup snapshot;
- it invalidates the entire `corpus_delta` fingerprint set, so the NEXT pipeline run re-upserts all
  2,257 episodes into the search index instead of skipping unchanged ones. Budget for that on the
  DGX; it is a cost, not a defect.

Work done per artifact written, across the three:

| migration | artifacts written | violations fixed |
| --- | ---: | ---: |
| 0007 | 153 | 2,327 |
| 0008 | 2,257 | 0 |
| 0009 | 1,844 | 227 |

### 0007 does more good than 0009 on the coherence measure — do not skip it

Measured on the real corpus, in order:

```
coherence violations   3,783   before anything
                       3,556   after m0009 alone     (-227)
                       1,229   after m0009 + m0007   (-2,554 total, -67%)
```

The 2,702-violation class m0009 cannot touch is `quote_two_speakers` — a quote pointing at BOTH a
bare id and its episode-scoped twin (`person:selena` and
`person:unresolved-selena-<episode>`), which is bare-name-scope residue, not a role problem. 0007
is the migration for exactly that.

### What the roles look like afterwards

| role | after | before the arc |
| --- | ---: | ---: |
| mentioned | 67.6% | 89.5% |
| host | 21.5% | 9.9% |
| **guest** | **10.9%** | **0.6%** |

That guest column is the operator's original report, fixed at production scale.

### Two classes needing a human, from the real preview

- **49 SUSPECT demotions** — the node had a voice, which is also what an eponymous-show host looks
  like. All 49 on this corpus are genuine show names: `Machine Learning Street` ×29,
  `Conversations with Tyler` ×18, `Turkey Book` (feed *Turkey Book Talk*), `Africa Tech Summit`.
  **If a real person appears in this list, stop.**
- **2 AMBIGUOUS nodes**, left untouched and both correctly refused: `William de Rimpel` (an ASR
  mangling of *William Dalrymple*, but the roster also holds a bare `person:william`) and
  `Jill Lepore` (the roster holds `jill-laporte` AND `jill-lapour`). Guessing either would have
  been wrong.

## Step 3 — Re-label the rest (fixes class B)

`relabel_only` re-resolves speaker NAMES on the **frozen** diarization and cascades GI/KG. No
audio download, no re-ASR, no re-diarization.

```
POST /api/jobs?pipeline_stage=relabel_only[&feed=<feed>][&profile=<dgx profile>]
```

It gets `--reprocess-existing-only` automatically, so the corpus can only shrink, never grow.
Point `profile=` at a DGX profile so the GI/KG cascade runs on local vLLM rather than a paid API.

**Scope it.** After step 2, 50 of 287 sampled episodes (17.4%) across 18 feeds still violate
coherence. Worst offenders in the sample:

| episodes | feed |
|---|---|
| 6 | The China-Global South Podcast |
| 6 | Biz Talk |
| 5 | The Daily |
| 5 | Colombia Calling |
| 5 | Round Table China |
| 4 | Conversations with Tyler |
| 3 | The Naked Pravda |

Run per-feed on the affected feeds rather than the whole corpus.

> `reprocess_dgx_no_llm` is **not** a cheaper substitute here — it sets `generate_metadata: false`,
> which skips the very stage the roster lives in.

## Step 4 — Verify

Every step here is a runnable command. An earlier version of this section named a function with no
CLI and told you to "re-run the migration", which the upgrade ledger blocks — it was a description,
not a procedure (advisor S5).

**On prod these are `Inspect prod corpus` dispatches, not `make` targets.** The scripts behind
them live in `scripts/`, which no runtime image carries (`docker/api/Dockerfile` copies `src/`
only) and the box has no Python — so until 2026-09-21 this step had no execution path on
production at all. Use `checks: speaker_migration_preview`, `speaker_coherence`,
`transcript_pairing_audit`. The `make` forms below are the local equivalents.

```bash
# What m0009 would do, bypassing the ledger. Run this at Step 1c as the pre-flight, and again
# after step 3 to confirm it has nothing left to do.
make speaker-migration-preview CORPUS_DIR=<prod corpus>

# Is the corpus self-consistent now?
make speaker-coherence CORPUS_DIR=<prod corpus>

# Did the migrations actually land, and is m0009's ledger still honoured? This one DOES run in
# the prod container (it is `src/`, reached through the same `upgrade` CLI as step 2) and it is
# the only prod-executable check of the chain — expect `[OK] 0009: N of N` and, after m0010,
# `[OK] 0010: every person name is canonical`.
docker compose ... run --rm --no-deps --entrypoint python api \
  -m podcast_scraper.cli upgrade verify --corpus-dir /app/output
```

**After Step 3, expect `upgrade-undo-roles` to refuse far more than 32** — a relabel rewrites
`.kg.json`, so m0009's recorded hash no longer matches and those episodes refuse by design. That
is the "undo BEFORE step 3, or not at all" rule, stated as a number.

**Read the preview's two human classes before running anything.** They exist because the
predicates are imperfect and say so:

- **SUSPECT demotions** — the node being demoted also had a voice. That is exactly what a host of
  an eponymous show (`Lex Fridman Podcast`, `Rich Roll Podcast`) looks like. On the 287-artifact
  sample all 6 are genuine show names; on a corpus containing such a feed, the host would appear
  here. **If a real person is in this list, stop.**
- **AMBIGUOUS nodes** — matched more than one roster entry, so they were left untouched. 0 on the
  sample; the guard is for the 85% of production not sampled.

**Read the ROLE TRANSITION table, not the violation count.** The coherence checks share the
migration's own predicates, so a wrongful demotion scores as a violation **FIXED**, not introduced.
`82 -> 56 violations` cannot detect the damage this migration is most likely to cause. The
transition table reports what changed per node:

```
mentioned -> guest      195
mentioned -> host       185
host      -> mentioned   21     <- the only destructive direction
guest     -> mentioned    0
```

Expected on the sample: 380 promoted, 21 demoted. Anything in `guest -> mentioned`, or a demotion
count far above 21, means look before proceeding.

**Restart the API after step 2.** Not optional. The migration rewrites `*.kg.json` and the upgrade
ledger; `perf_cache.corpus_mtime` now includes the ledger so the mtime-tokened projections do
invalidate, and `corpus_graph` / `cil_queries` now carry the token too — but a restart is still the
only thing that guarantees every in-process cache is gone.

**Reindex only if needed.** `relabel_only` reindexes incrementally on its own (advisor S7): the
finalize path calls `maybe_index_corpus`, and `prod_dgx_full.yaml` sets `vector_search: true` with
no skip flag. Check `vector_index_seconds` in the run summary; a full `make index-two-tier-docker`
is only needed if that is 0.

## Known limits — read before declaring victory

- **AFTER the chain, step 4's own preview trips step 4's own stop condition — on a real person.**
  Re-running `make speaker-migration-preview` on the migrated corpus reports 39 demotions, all
  `Peter Attia (host) — The Peter Attia Drive`, 33 of them in the hand-read SUSPECT list. Nothing
  demoted him during the run (the pre-migration preview has zero Peter Attia); m0010 canonicalises
  `Peter Attia, MD` to `Peter Attia`, and `names_the_show` matches a multi-token PREFIX of the
  article-stripped feed title, so `("Peter Attia", "The Peter Attia Drive")` is True where the
  3-token credentialled spelling was not. **Do not read those 39 as damage from the migration, and
  do not "hand-read and approve" them either.** The same composition reaches the LIVE pipeline —
  `roster.py:960` canonicalises the published name and `metadata_generation.py:1159` drops placed
  speakers that match — so that feed loses its host on future ingests. Measured across all feeds on
  `snapshot-prod-20260920`: 6 feed/name pairs match, 5 are genuine show names (MLST, Conversations
  with Tyler, Trivium China, Turkey Book, Africa Tech Summit) and **only The Peter Attia Drive is a
  person**.

  **UNRESOLVED, and a fix was tried and withdrawn on 2026-09-21 — read this before trying again.**
  Nothing available locally separates the two cases: not the voice (Machine Learning Street has
  one on 4 of 6 episodes), not the feed author (it is the show's own name for Trivium China,
  Africa Tech Summit and MLST), and not a generic-suffix rule (it would swallow *The Tim Ferriss
  Show* and *Lex Fridman Podcast*).

  The withdrawn attempt used the feed's `known_hosts` as an operator statement that a name is a
  PERSON. **It does not mean that.** `tried.known_hosts` in the diagnostics is the MERGED
  `cfg.known_hosts + feed_hosts` (`providers/ml/diarization/pipeline.py`), and `feed_hosts` falls
  back to the RSS author — which carries the SHOW's name. Measured on `snapshot-prod-20260920`:
  2,094 of 2,492 diagnostics have a non-empty `known_hosts`, and for five feeds that value IS the
  show name, so the guard would have re-seated the show as host on **111 episodes** — exactly the
  #2064 damage it was meant to prevent. Any future attempt must read the operator's CONFIG, not
  this field.

  m0009 is deliberately NOT changed either: in chain order 0009 runs before 0010, so it never sees
  the canonical spelling, and sparing voiced nodes there would undo 59 correct show-name demotions
  to protect nobody.

- **An org in the host seat whose name is not the show's** survives step 2 entirely: `China Plus`,
  `Mercatus Center at George Mason University`, `Brandon Anderson, RJ Honicky, and Latent.Space`.
  No predicate can tell these from a person. They are the bulk of the 56 remaining violations and
  they need step 3.
- **39 roster names have no matching node** in their episode's graph — near-miss spelling variants
  (`bernt børnich` vs `bernt bornich`). The migration reports them and deliberately does not insert
  a node, because that would put the same human in the graph twice. Needs step 3 or #2056's
  variant resolver.
- **Does a `relabel_only` run rebuild the search index by itself? UNVERIFIED, and this file used
  to answer it both ways** — "reindexes incrementally on its own" under Step 4, "assume not" here,
  pointing at a "step 4.3" that does not exist. Treat it as unverified: after Step 3, read
  `vector_index_seconds` in the run summary and reindex explicitly if it is 0.
- **The 17.4% figure is from the 287-artifact sample** and the real proportion has not been
  measured. Corpus size depends on what you are counting and this file used to give three answers:
  `snapshot-prod-20260920` holds **2,298 episode artifact sets** (2,297 with a transcript pointer);
  prod's API reports ~1,956 EPISODES. Metadata files include superseded runs — never compare the
  two directly.
- **The full chain HAS now been run end to end** (2026-09-21, four times, on a restored
  `snapshot-prod-20260920`) — but with m0002 recorded as applied, because `restore-corpus-prod`
  prunes `search/` so it can never no-op locally. 0003-0010 are exercised for real; **m0002 is
  the one migration no local drill can cover**, which is why its prod plan line is a hard gate.
  For the same reason `drill-corpus-upgrade.yml` cannot rehearse this chain: it restores the same
  pruned backup and dies at migration 2 of 10 every time.
- **Three feeds will light up the `empty_host_anchor` rework queue at once, and that is the fix
  working.** The feed-host work may now only REMOVE a host, so 63 episodes across The Rest Is
  History (`Norman Conquest`, 84 voices), Latin America in Focus (`Americas Online`, 32) and
  Colombia Calling (`Emily Hart`, 8 — a wrong name on Richard McColl's voice) move from a junk host
  name to no host at all. 126 files, 63 episodes; 20 other episodes GAIN a named host in the same
  change. Nothing regressed: every name lost there was junk, an organisation, or the wrong person.
  Whoever watches that queue after deploy should not read the spike as new damage.
- **A relabelled episode's record used to keep the OLD names.** Until `51a202bb` no reprocess stage
  rewrote `content.speakers` — verified by diffing before/after on two DGX episodes, byte-identical.
  Every episode repaired by a relabel BEFORE that fix therefore has a transcript with new labels and
  a record with old names, and every number read from such a record describes the pre-repair state.
  The same defect wrote a SECOND record into the reprocess's own fresh run directory, which then
  made `rederive_only` refuse the episode. Both are fixed; the corpus cleanup is tracked on #2097.

## Rollback

**The role changes are reversible (#2069).** m0009 writes `speaker_roles_ledger.json` at the corpus
root recording every `(episode, node, role_before, role_after, route)` transition, and:

```bash
make upgrade-undo-roles CORPUS_DIR=<prod corpus>              # replay every role_before
python scripts/ops/undo_speaker_roles.py --corpus-dir <c> --show   # read it, change nothing
```

### On PROD, that `make` target does not exist — here is the command

`undo_speaker_roles.py` lives in `scripts/`, which no runtime image carries and the box has no
Python for. This is the ONLY write in this document that has no workflow behind it, so it is
written out in full; assembling it during an incident is how a rollback gets skipped.

```bash
cd /srv/podcast-scraper
touch corpus/.viewer/jobs.paused                        # and confirm GET /api/jobs is idle
rm -rf /tmp/undo_pkg && mkdir -p /tmp/undo_pkg          # stage OUR code next to the image's deps
# from a checkout, or scp these two from your laptop:
#   src/podcast_scraper -> /tmp/undo_pkg/podcast_scraper
#   scripts             -> /tmp/undo_pkg/scripts
SEC=''; [ -n "$(ls -A /dev/shm/podcast-secrets 2>/dev/null)" ] && SEC='-f compose/docker-compose.secrets.yml'
C="docker compose --env-file .env -f compose/docker-compose.stack.yml \
   -f compose/docker-compose.prod.yml -f compose/docker-compose.vps-prod.yml $SEC"

# READ FIRST — lists every run in the ledger. More than one means m0009 ran twice.
$C run --rm --no-deps -e PYTHONPATH=/mnt/undo_pkg -v /tmp/undo_pkg:/mnt/undo_pkg:ro \
  --entrypoint python api /mnt/undo_pkg/scripts/ops/undo_speaker_roles.py \
  --corpus-dir /app/output --show | head -20

# THEN write. Drop --run-id only when --show reported exactly one run.
$C run --rm --no-deps -e PYTHONPATH=/mnt/undo_pkg -v /tmp/undo_pkg:/mnt/undo_pkg:ro \
  --entrypoint python api /mnt/undo_pkg/scripts/ops/undo_speaker_roles.py \
  --corpus-dir /app/output [--run-id <newest>]
```

**It runs as root** (`--entrypoint python` bypasses the privilege drop), so restart afterwards
exactly as after Step 2 — the entrypoint's `chown` is what makes the files writable by the
pipeline again.

**Expect a non-zero exit.** After the full chain 32 episodes refuse by design; after Step 3 far
more will. Non-zero here means "partial", not "failed" — read the `REFUSED` lines.

**If m0009 ran more than once, the default undoes only the newest run.** `--show` now names every
run and warns when there is more than one; undo them newest-first, one `--run-id` per call. There
is deliberately no "undo everything" flag: the order matters, and the wrong order writes an older
run's `role_before` over a newer one's.

Verified twice. On a 287-artifact staging copy with nothing else touching it: 401 changes applied,
401 restored, 0 refused, corpus sha256 **byte-identical**.

And on the real `snapshot-prod-20260914` corpus (2,257 artifacts), migrations run in the ORDER a
real `upgrade run` uses — 0007, 0008, 0009:

```
ledger rows : 3,370
restored    : 3,370
skipped     :     0
refused     :     0
```

A clean, complete rollback at production scale.

### That measurement predates m0010, and m0010 broke it — fixed 2026-09-21

**Re-measure whenever a migration is added after 0009.** The run above stopped at 0009. m0010 runs
*after* it and rewrites the same `.kg.json` files, so every episode it touched then failed its own
`file_sha_after` check. Measured on `snapshot-prod-20260920` (2,298 episodes) with the full
0001→0010 chain, before the fix:

```
ledger rows : 3,285  across 1,775 episodes
REFUSED     :   134  rows /  73 episodes   <- unrestorable
make upgrade-verify -> [FAIL] 0009: 3283 of 3285 recorded role(s) still present   (exit 2)
```

The two rows verify could not find addressed `person:peter-attia-md`, an id m0010 **merges** into
`person:peter-attia` — those a hash refresh alone could never recover.

The fix is `role_ledger.resync_after_rewrite`, called by m0010 for the episodes it rewrote: each
row is re-pointed at the surviving node and the new file hash. It **deliberately refuses four
cases**, because a receipt it cannot honour is worse than an admitted gap:

1. a row whose node vanished entirely;
2. a merge where the survivor holds a role other than the one recorded (its role came from the
   other half of the merge);
3. a merge that absorbed a node holding `host`/`guest` that **no row describes** — the pipeline,
   not m0009, wrote that role;
4. a duplicate-id FOLD that discards any copy holding a speaking role.

**3 and 4 are not hypothetical, and the ledger's own totals cannot see them.** Measured on
`snapshot-prod-20260920` before they were guarded: 30 episodes where the pipeline's `host` on
`person:peter-attia-md` was folded into `person:peter-attia` and the undo then restored
`mentioned`, plus 1 where `person:jen-kha` appeared TWICE — "Jen Kha" (`mentioned`) and "Jen Kha)"
(a name cut at a bracket, `host`) — and the fold kept one. **All 31 were reported as
`restored …; refused 0`.** Do not trust the restored/refused counts alone; diff the roles against
a pre-upgrade copy, which is the only check that sees this class.

Same corpus and chain, after the fix:

```
0010 ... 94 role-ledger row(s) re-pointed, 32 episode(s) left stale
make upgrade-verify     -> [OK] 0009: 3284 of 3284 recorded role(s) still present   (exit 0)
make upgrade-undo-roles -> restored 3244 role(s); skipped 0; refused 32             (exit 2)
```

**`upgrade-undo-roles` now exits NON-ZERO on a full-chain rollback, and that is correct** — 32
episodes genuinely cannot be restored, and a partial rollback must not read as a complete one.
Role diff of all 2,298 episodes against the pristine snapshot: 2,260 identical, 38 differing =
32 refused + 6 where only the id was canonicalised (role preserved), **0 speaking roles lost, 0
unexplained**.

### Do NOT undo and then re-run the chain

After an undo the ledger reads `0010 applied, 0009 pending`, so the only forward path re-applies
m0009 against names m0010 has already canonicalised. `Peter Attia, MD` is now `Peter Attia`, which
`names_the_show` matches against *The Peter Attia Drive* — so the re-run DEMOTES him where the
first run did not. If you have undone, restore from the pre-upgrade snapshot instead of re-running.

**An earlier version of this section claimed 54 refusals as "the realistic case". That was wrong,
and the error was mine:** I had rehearsed 0009 BEFORE 0007, so 0009 recorded hashes that 0007 then
invalidated. In the real order 0009 runs last and its hashes are current. Re-measured above.

The refusal path is still real and still matters — but the thing that triggers it is **step 3**, a
`relabel_only` re-enrich, which rewrites kg.json after the migration. That is why the rule below
is: undo BEFORE step 3, or not at all. When it does refuse, it refuses correctly: a role-only
check would have seen matching roles, demoted everything, reported zero refusals and called it a
clean rollback while destroying the newer work.

**Restart the API after an undo, exactly as after the migration — and do not rely on the token
here.** The undo un-records 0009 from `upgrade_ledger.json`, which moves the `perf_cache` token,
but only when it actually rewrote something: `record_reverted` returns False without writing if
nothing was restored, or if 0009 was not in `applied` to begin with. A run that restores 0 rows
therefore leaves the token where it was. That is the correct behaviour — nothing changed — but it
means "the undo moved the token" is not something to assume. Restart.

**The undo now REFUSES rather than racing.** If an ingest or a migration holds the corpus lock,
`undo_speaker_roles.py` prints `REFUSED: …` and exits 1 having changed nothing. Stop the other job
and re-run. Previously it detected the contention and then proceeded unlocked, which is the one
case the lock existed for. `upgrade run` takes the same lock in the forward direction, around
**both** the pre-upgrade snapshot and the migrations — the snapshot is minutes of `copytree` and
is the rollback you are being handed, so it must not be copied out from under an ingest.

**What that lock does and does not cover, precisely:**

- **Covered:** a multi-feed ingest (`cli.py`, `service.py`) — the shape prod runs — and a
  concurrent `upgrade run` or undo. All take the same parent lock, so contention is detected.
- **NOT covered:** a single-feed ingest with `single_feed_uses_corpus_layout`, which locks
  `<root>/feeds/<slug>` instead — a different lock file. A migration and such an ingest can still
  run concurrently with no refusal. Prod's schedule is multi-feed, so this is a gap on paper today.
- **Expect the reverse, too:** a scheduled ingest that STARTS during the migration window now
  fails loudly rather than interleaving (the CLI raises, `service.run` returns `success=False`).
  That is the intended behaviour — **re-trigger the ingest after the migration finishes.**
- **Verify on the box:** if `PODCAST_SCRAPER_CORPUS_LOCK=0` is set in the deploy environment, every
  lock above is inert. Check before relying on any of this.

**`make upgrade-verify CORPUS_DIR=…` now means something for 0009.** It checks the ledger's rows
against the artifacts: `401 of 401 recorded role(s) still present` after a run, `0 of 401` after an
undo. Previously it returned "no verification defined".

**So: undo BEFORE step 3, or not at all.** After a re-enrich the ledger's episodes are refused by
design, and that refusal is correct — the re-enriched answer is the better one.

Three outcomes, not two: `restored` / `skipped` (already back where it started) / `refused`
(something else owns that file now). The CLI exits non-zero when anything is refused, because a
partial rollback must not read as a complete one.

**Byte-identity has a scope.** A kg.json last written by a MIGRATION comes back byte-identical. One
last written by the PIPELINE (a new ingest, or a `rederive_only` since the last migration) uses a
different serialiser — those files are refused by the sha check anyway, so the undo never touches
them.

**That changes the decision.** Running the migration is no longer a one-way door: the question is
"is the damage bounded and recorded", not "can we prove zero damage in advance". Read the SUSPECT
and AMBIGUOUS lists first anyway — the ledger makes a mistake recoverable, not free.

### Step 2 snapshot (still recommended)

`upgrade run` snapshots the whole corpus before any migration, and aborts if the snapshot fails.
That covers migrations with no ledger; the role ledger is the finer-grained undo for m0009
specifically. Keep both until step 4 passes.

Step 3 regenerates artifacts from audio-derived data that is not itself modified, so re-running it
is safe; there is no ledger for it — take a snapshot first if the corpus state matters.
