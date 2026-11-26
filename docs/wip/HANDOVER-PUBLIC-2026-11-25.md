# Handover: the public session, 2026-11-25

For the next public-repo agent. It covers state as of the evening of 2026-11-25 and follows
`HANDOVER-PUBLIC-2026-11-24.md`, whose working agreements still hold. Check each "open" line against
GitHub before acting on it.

## Corrections to the 2026-11-24 handover

- **Main's protection rule:** ruleset `17052614` (`gh api repos/chipi/podcast_scraper/rules/branches/main`).
  - It requires a pull request with an approving review, and blocks deletion and non-fast-forward pushes.
  - Every recent merge into main (#2504 to #2538, #2542) went in as `REVIEW_REQUIRED`, through the
    operator account's bypass, with `--admin` for a PR.
  - A bypass needs the operator's go, per merge, like the merge itself.
- **W2's network go was given** (the operator, 2026-11-24, CRFC-006 §6). The drift measurement is
  the database session's.
- **W9 (the news re-check job, the version rows, the API history) is the database session's**, not
  ingestion's: the operator answered "None for now" to ingestion taking it (ingestion, 2026-11-25).

## Done on 2026-11-25

| What | Where |
| --- | --- |
| The re-check's normaliser, compare and hold rule, and per-item schedule (CRFC-006 B.2) | #2540, release branch d6a4a5c69 |
| A content-duplicate skip is remembered, so `episode_selection=unprocessed` stops re-offering it every night | #2542, main 55b03ede1 |
| One engine per role, and a `corpus_ro` engine with its own 5 s and pool (E06 item 5, CRFC-005 §7) | #2544, release branch e9726b7c6 |
| The re-check's page fetch: conditional GET, gone (404/410) told apart from failed | #2545, into the release branch, open |

The nightly of 2026-11-25 (run `7b497207`, the first under `episode_selection: unprocessed`) was
read from VictoriaLogs.

- 24 feeds had new episodes, 28 in all, and none hit the cap of 10.
- One feed with a candidate saved no transcript: Simplecast `JT6pbPkg`. Its new episode has the same
  audio bytes as "AlphaGenome 2" and was skipped by the #1656 gate, then re-offered every night.
  #2542 fixes that.
- Prod's report also named an Omny feed, but that feed saved its episode
  (`transcripts_saved=1` at 01:29:37).

## The engines change, as built (#2544)

- **URL env:**
  - `PODCAST_DATABASE_CORPUS_RO_URL`: read by the `corpus_ro` engine.
  - `PODCAST_DATABASE_PIPELINE_RW_URL` and `PODCAST_DATABASE_MIGRATOR_URL`: named for infra; nothing
    reads them yet.
- **Corpus engine:**
  - `PODCAST_DB_CORPUS_STATEMENT_TIMEOUT_MS` (5000), `PODCAST_DB_CORPUS_POOL_SIZE` and
    `PODCAST_DB_CORPUS_MAX_OVERFLOW` (2 and 2).
  - It is read-only, with no schema map: the corpus tables carry schema `corpus` (ADR-184).
- **Corpus-only processes:** a plane engine opens only with a non-empty `PODCAST_DATABASE_URL`. So
  compose-api (`PODCAST_DB_PLANE=player`, an empty URL, the corpus URL) starts on the corpus engine
  alone.
- **The plane's ro engine is the rw engine** unless `PODCAST_DATABASE_RO_URL` names another URL.
  Plane reads on prod no longer carry `default_transaction_read_only`. The operator was told
  through the database session.
- **Infra** has the names: its #108, #109 and #110 are merged into the release branch.

## Open: check these first

1. **#2545:** merge into the release branch once CI is green (the operator's go is given).
2. **Waiting on the database session** (W9):
   - the job's five open items, sent 2026-11-25: the Guardian by id, Hacker News' linked URL,
     B.2 step 2's re-extraction, the second extraction, G7's state with the validators;
   - the names `PAGE`, `NOT_MODIFIED` and `Validators`;
   - a G9 request: the extraction method and the extractor's version on the document row.
3. **Waiting on the database session, for infra:** which containers run the projector, the corpus
   migrations and the import, and that the migrator URL goes only into a one-shot.
4. **The pipeline_rw engine and its pool env are public's**, with the writers-on-rows change.
   Proposed to infra, names only: `PODCAST_DB_PIPELINE_POOL_SIZE` and
   `PODCAST_DB_PIPELINE_MAX_OVERFLOW`, 1 and 1. Tell infra when they exist.
5. **After deploy, confirm #2542 on prod:** the Simplecast `JT6pbPkg` rerun downloads once more, and
   from then on the next nightly logs it as already ingested.
   - Not checked: that prod's fingerprint index sits under the same `output_dir` that selection reads.

## Next, in order

1. The 2026-11-24 handover's next-cycle rows:
   - the Library publisher filter from a server route;
   - P4, the dead diarization entry in `model_manifest.py`;
   - 13/P3, the index-page classifier, #2502;
   - 23, the feeds file round-trip.
2. Review the player's W8 `storyline_newsonly_*` template PR when it arrives.

## This workstation

- **This checkout's test database** (`podcast-scraper-test-db-public`) stopped accepting host
  connections after 6 days up, with Postgres healthy inside. `docker restart` of that container
  fixed it; it is now on port 33217.
  - `make gates-changed` finds it by itself.
  - When every database test errors at setup with "server closed the connection unexpectedly", check
    this first.
- **The `.venv` still has no `[news]` extra:** 14 to 36 news tests fail in `make gates-changed` with
  `ModuleNotFoundError` (trafilatura, feedparser, lxml). Run those files with
  `PYTHONPATH=src /work/closelistening/ingestion/.venv/bin/python -m pytest …`, where they pass.

## Not covered by this handover

- **The pool sizes infra set:** I didn't check them against CRFC-005 §7's table.
- **Other skips that write no metadata:** whether they also re-select every night (for example an
  unsupported language) wasn't checked.
- **Carried over from 2026-11-24, not touched today:**
  - item 34's half proof;
  - Snyk's 82 container findings;
  - the Guardian key in chipi/infra's `prod` environment.
