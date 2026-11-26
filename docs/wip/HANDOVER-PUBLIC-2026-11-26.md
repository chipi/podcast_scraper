# Handover: the public session, 2026-11-26

For the next public-repo agent, which starts on the E06 work (the corpus database). It covers state
as of early 2026-11-26 and follows `HANDOVER-PUBLIC-2026-11-25.md`. Check each "open" line against
GitHub and the trackers before acting on it.

## Where things are

- **The two trackers** (collection `items`; update only `owner: public` rows, pin each write with
  `if_version`, new rows get a `public-` doc id):
  - the cut-over plan: <https://claude.ai/artifact/2ySdEBp7RynTGTeNDgWKRi>;
  - the small deploy from main and its after-deploy checks:
    <https://claude.ai/artifact/Qn9BxTv5kHecoeADuX3RbC>.
- **Main** carries the small deploy's public fixes. **`feat/corpus-database-release`** (the E06
  release branch) has main merged in up to 6a9a4a735 (#2548, merge commit e29fcc3ad). Merge main in
  again with a merge commit, never a squash, so main's history stays an ancestor and the next merge
  does not conflict.
- **Public's `.venv`** now has the `[dev,ml,llm]` extras, the news libraries included (feedparser,
  lxml, trafilatura, readability-lxml). The news tests run on it: 116 passed on 2026-11-26.

## Done on 2026-11-25 and 2026-11-26

| What | Where |
| --- | --- |
| The re-check's page fetch: conditional GET, gone told apart from failed (CRFC-006 B) | #2545, release branch 5feda442b |
| The podcast ad cleaners skip a news article's text (backport of #2516) | #2549, main 616f92a90 |
| A source with `text_from: feed` never fetches the article page (SCMP) | #2550, main 17b4b6ad8 |
| A feed item's text keeps the words outside its HTML blocks (Nature) | #2551, main 6f6caa957 |
| The search backports: #2533, #2534, #2535, #2515 | #2553, main 9883acf27 |
| The `corpus_indexed` stage hook, run once the corpus index is built (Common #49) | #2554, main 7405cae5e |
| An insight's grounding confidence stays out of the episode fingerprint | #2557, main 6a9a4a735 |
| Main merged into the release branch | #2548, release branch e29fcc3ad |
| Nature and SCMP take their text from the feed | Common #82 and #81, merged |
| The eval-move cleanups and the stack Digest test, rebased onto main and merged | #2509, #2510, #2511, #2487 |
| The 2026-11-25 handover | #2546 |

### What each one found

- **Nature (Common #50).** From prod's address Nature answers every article page with a 3,038-byte
  JavaScript "Client Challenge" page, so the standfirst rule found nothing. The prod session found it
  by fetching one URL from the nightly's image. The feed's `content:encoded` carries the same
  standfirst as loose text after a date line's `<p>`; `extract._blocks_text` dropped it. 47 of 47
  news items in the feed saved 2026-11-25 now get text.
- **The nightly reads `/app/output/feeds.spec.yaml` on the prod box**, not Common's
  `config/news/news.feeds.yaml`. The Nature and SCMP switches reach prod only when the prod session
  edits that file after the deploy, on the operator's go (tracker row `pd-feeds`).
- **#2535's fingerprint.** As backported it hashed `grounding_confidence` into each episode's index
  fingerprint. Prod has 3,549 of 4,089 GI files with it, so the first nightly after the deploy would
  have re-embedded nearly the whole corpus (about 700,000 vectors at 4 to 20 per second, against the
  index subprocess's 7,200 s limit). The operator held that part back; #2557 restores the old
  fingerprint byte for byte. Old insight rows keep confidence 0.0 until re-indexed for another
  reason.
- **`backbone_changed_relpaths` is observational.** The index fingerprints every episode on each
  build. A change to `_episode_fingerprint` or to what goes into a row re-embeds every episode it
  touches.

## Open: check these first

1. **The small deploy** is the prod session's to run (`sd-release`, `sd-deploy`). After it, public
   owns:
   - `pd-marker`: a PR into main setting `config/last_deployed_prod_version.json` to the deployed
     release, then main merged into the release branch again;
   - `public-pd-dupes`: after the first nightly, #2542's duplicate episode is in
     `audio-fingerprints.json`'s `duplicates` and is not offered again.
2. **Common #49:** Common registers its insight-cluster rebuild on `PipelineHooks.corpus_indexed`
   (`(corpus_root: Path) -> None`). The hook adds no guard: Common's hook must catch its own errors.
   It is not called from the single-feed finalize or from `POST /api/index/rebuild`; say so if asked.

## E06: public's rows in the cut-over tracker

All wait on the schema from the database session, except where noted.

- `w-public`: public's pipeline stages and enrichers write rows. It includes the extraction method
  and extractor version on the document row (G9), and reading `PODCAST_DATABASE_PIPELINE_RW_URL`.
  `episode_selection: unprocessed` and #2542's duplicate list decide "already ingested" from
  metadata files and `audio-fingerprints.json`; on rows they must read rows.
- `public-pipeline-rw-engine`: the `pipeline_rw` engine and its pool env. Names proposed to infra,
  not in code: `PODCAST_DB_PIPELINE_POOL_SIZE` and `PODCAST_DB_PIPELINE_MAX_OVERFLOW`, 1 and 1.
- `r-search`: search hydration from rows; `search_chunks` fed by the index build.
- `r-graph-engine`: the graph engine in the server (ADR-183).
- `r-workbench`: the Workbench on slices.
- `n-api-versions`: the API's view of versions. It waits on the version rows and W9's withdraw state.
- `a-e09`: search on Postgres, after the cut-over.

The engines change (#2544) as built is in the 2026-11-25 handover.

## Corrections: where this session went wrong

- I told the database session that older insight rows keep no confidence after #2535. That was
  false: I had not read that `backbone_changed_relpaths` does not decide the skip. The re-embed was
  caught before the deploy only because I went back to the code.
- My first Common #81 and #82 descriptions said a deploy changes what prod reads. The prod session
  corrected that (the feeds file above).
- Twice I sent a message to another session without running `session-handoff-msg`'s
  `lint_msg.py` first.
- #2554's first CI run failed on mypy in its test file. I had type-checked only the `src` files; CI's
  `make type` checks everything.
- A ci-watch pipe ending in `| tail` hid a non-zero exit; end such commands with `EXIT=` instead.

## Working notes

- Inline `python -c`, heredocs to `python -` and `GIT_EDITOR=` are blocked by the shell guard. Write
  a script file; use `git -c core.editor=true rebase --continue`.
- Committing in a Common worktree needs `PODCAST_SCRAPER_ROOT=/work/closelistening/public` for its
  corpus change gate.
- The Snyk app check ("You have used your limit of private tests") and a GHCR login timeout in
  `Snyk - Docker` are quota and network failures, not findings; re-run the failed job.
