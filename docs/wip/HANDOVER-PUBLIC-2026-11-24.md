# Handover: the public session, 2026-11-24

For the next public-repo agent. It covers state as of the end of 2026-11-24. Check each "open"
line against GitHub before acting on it; several move overnight.

## Working agreements in force

- The release work lands on `feat/corpus-database-release`. Each change is a feature branch cut
  from it, with a PR into it. Never rebase or force-push the release branch.
  - `main` stays prod's deployable line. Urgent fixes and CI-only changes go to main.
  - Item 25 (W0) ships from main before the cut-over.
- **A merge into the release branch or a push to main needs the operator's go, per action.** Get
  it by asking in the session, as a survey with a recommendation.
  - The push to main prints "Bypassed rule violations for refs/heads/main". Main has a protection
    rule that the operator's account bypasses. Nobody has checked which rule it is.
- The database session coordinates the release (`uds:` socket named `database`). The player
  session owns the board summarizer and W8. Ingestion owns the nightly and the re-check job. Prod
  owns the prod box: no session here can ssh to `deploy@prod-podcast`.
- **This workstation has no Postgres test database and no `[news]` extra in `.venv`.**
  - The server unit tests that need the database fail locally with `DatabaseNotConfigured`; CI
    has one. `make test-db-up` starts one, which nobody has tried this session.
  - The news tests run with `/work/closelistening/ingestion/.venv/bin/python`, with
    `PYTHONPATH=src`. Nothing was installed.

## Done on 2026-11-24

| What | Where |
| --- | --- |
| Board summarizer: the news path imports no fastapi | 558cb1bce, main and release branch |
| PR workflows run on PRs into `feat/*-release` | #2517 (release branch, 231b7594e), main 17caa859e |
| Item 35 / GlitchTip 1191: news text skips the podcast ad cleaners | #2516, release branch beb5e9602 |
| Item 35 / GlitchTip 758: one BBC Business item; the model left its JSON bullets unclosed | accepted by the operator; the constrained-output fix is a measured change for later |
| Item 24: after a cursor, take the oldest items first, so none over the cap is skipped | #2536, release branch 10a952593 |
| Item 24: the `news_source_listing` event (backlog, cursor lag, lost interval for RSS) | in #2536 (`news/listing.py`) |
| Item 34: Snyk Docker SARIF, dropping the invalid fix suggestions | main e03704b91 |
| Item 33: a CI Gate verdict for another commit no longer fails the check on main's head | main 5f94a9bf6 |
| `NEWS_API_KEYS.md`: the key goes in chipi/infra's `prod` environment | main 59786c911 |
| Podcasts: `episode_selection: unprocessed`, `episode_since: 2026-11-24` | applied by the prod session at 13:56:24Z; undo with `/app/output/viewer_operator.yaml.bak-20261124T1356Z-selection` |

## Open: check these first tomorrow

1. **PR #2537 (item 25, W0, into main).** A storyline's news comes from its member topics.
   - Watch its CI. Merging needs the operator's go.
   - Not done: the articles-per-storyline measure before and after on a prod copy. The
     2026-11-24 backup has no `search/` directory, so the themes file isn't in it. Ask database or
     prod for a copy of prod's `search/topic_clusters.json`, or a read-only run on the box.
   - Before touching Common's `topic_theme_clusters` enricher: this PR needs no Common change.
     Tell database what it reads.
2. **Main at 5f94a9bf6:** at hand-off, 22 checks passed, 2 were skipped and test-e2e was still
   running. Read its verdict: `ci_watch.py 5f94a9bf6 --repo chipi/podcast_scraper`.
3. **Item 34 is only half proven.** The upload succeeds on main since 5f94a9bf6, but the step's
   log says `results whose fix suggestions were dropped: 0`. The bad shape didn't recur, so the
   step is proven only on a sample.
   - Database's row 34 should say exactly that. Check Common PR #60's wording.
4. **The first nightly under the podcast change and the news cursor change** runs at 00:00Z on
   2026-11-25.
   - Prod reports per podcast feed how many episodes were taken, and which feeds hit the cap of
     10.
   - For news, read the `news_source_listing` events for the 7 feeds over their cap on
     2026-11-23. Backlog should fall to 0 over a few nights, with no "never listed" warnings in
     GlitchTip.
   - The news change is on the release branch, not main. Until the release deploys, prod still
     runs the old newest-first code, and no `news_source_listing` events will appear.
5. **#2515 (ingestion), into the release branch:** database merges it when it's green.
   `codecov/patch` failed at 1c67ec456; ingestion pushed d9117240c with tests.

## Next, in order (agreed with database; recorded in E06-plan, Common PR #61)

1. **Part B in public, on the release branch** (CRFC-006 §3, W2's public half and W9's API half).
   - The fetch, normaliser and compare as pure code on fixtures.
   - Then the item read's served version, plus the history option: version number, `fetched_at`,
     `change_kind`, hash, never an old body.
   - It stays off while every source's re-check window is 0.
   - Propose the names B.4 leaves open in the PR; don't choose them silently.
   - Read CRFC-006 on Common branch `plan/news-2-and-e10` first.
2. **W8's `storyline_newsonly_*` template is the player's** (the operator, 2026-11-24). Review its
   PR into public when the player sends it.
3. **The rest of my next-cycle rows** (Common `docs/wip/NEXT-CYCLE-AFTER-NEWS-RELEASE.md`):
   - the Library publisher filter from a server route, replacing 69e68fa54's sweep;
   - P4: the dead diarization entry in `model_manifest.py`, now that 0051 has run;
   - 13/P3: the index-page classifier, #2502;
   - 23: the feeds file round-trip.

   15 and 16 are News 2.0, later. Item 4 of database's News 2.0 list (the snippet rights value)
   is later too.

## Not covered by this handover

- The news-only summary that GlitchTip 758 left without a summary isn't re-run. That would need
  an LLM call, and the operator's rule allows prod fixes from stored evidence only.
- Nobody checked why main's protection rule is bypassed.
- The Guardian key isn't set yet: no `PROD_GUARDIAN_API_KEY` in chipi/infra's `prod`
  environment. The feed stays commented out in Common's `config/news/news.feeds.yaml`.
- Snyk's own results (82 container findings) weren't read. Only the upload was checked.
