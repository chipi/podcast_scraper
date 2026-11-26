# Handover: the public session, 2026-11-26 (afternoon)

For the next public-repo agent. It follows `HANDOVER-PUBLIC-2026-11-26.md` (the morning) and covers
state as of about 12:30Z. Check each "open" line against GitHub and the trackers before acting.

## The rule that changed today

- The operator, 2026-11-26: everything we work on now goes with the E06 / News 2.0 deploy. No
  more small deploys from main. The database session confirmed it for every repo.
- Every public PR targets `feat/corpus-database-release`, not main. There are no more
  main-into-release merges, except a prod hotfix on the operator's go.
- The release scope is closed (cut-over tracker row `d-scope`). A new item joins only on the
  operator's go.
- **Pushes to the release branch run no CI.** The merge commits de7b52fcd (#2563) and 48d896703
  (#2564) have 0 check-runs (`gh api repos/chipi/podcast_scraper/commits/<sha>/check-runs`). The
  PR's own run is the only proof, so read a PR's checks before merging, not after.

## Trackers

The same two trackers as in the morning handover:

- cut-over: <https://claude.ai/artifact/2ySdEBp7RynTGTeNDgWKRi>;
- small deploy: <https://claude.ai/artifact/Qn9BxTv5kHecoeADuX3RbC>.

Collection `items`. Update only `owner: public` rows, and pin each write with `if_version`.

## Done today (afternoon)

| What | Where |
| --- | --- |
| `pd-marker`: prod marker at sha 6a9a4a7, migrations 0001 to 0053 without 0032, pinned fixture ledger regenerated | #2563, release branch de7b52fcd |
| The marker checked against prod's own ledger: 52 ids, version 2.8.3, the same `to_version` for 0046 to 0053 | prod session, read-only |
| `pd-release-branch` closed for public | tracker |
| The served-surfaces list for the parity harness (`s-harness`) | #2562 (open, CI green) |
| Review of the documents schema (#2564, merged as 48d896703) and of `corpus_keys` (#2569) | messages to database; answers on #2564 issuecomment-7189482266 |

How the marker was made: infra's `scripts/ops/bump_prod_marker.py`, run against the deployed
registry (public 6a9a4a7 plus Common afabad2) in a throwaway venv. Its list is identical to the one
infra run 45778271234's control-plane job computed. Nothing on public main reads the marker except
main's own upgrade tests; infra's `deploy-prod.yml` only computes the bump and prints it.

## Open PRs (public's)

| PR | What | State at handover | Operator's go |
| --- | --- | --- | --- |
| #2565 | `pipeline_rw` engine: `PipelineSettings`, `init_pipeline_engine`, `pipeline_rw_engine`; 1+1 pool, no statement timeout | all checks passed except `test-e2e-fast`, still running | merge on green, given |
| #2566 | `document_cut.cut_body` and `paragraph_cuts`, the cutter the writer and the importer share | same | merge on green, given |
| #2562 | `docs/wip/HARNESS-PUBLIC-SURFACES-2026-11-26.md` | CI green | not asked |
| branch `feat/chunk-segment-range` (803c4b8ea, pushed, no PR) | `cut_range`: a body span as `(start_seq, start_offset, end_seq, end_offset)` over the cuts; the start of `r-search` and #2518's missing offsets | stacked on #2566 | open its PR after #2566 merges: rebase onto the release branch and drop the cutter commit |

Merge these PRs with `--squash`. Main-into-release merges used merge commits. For evidence on PR
2566, the database session swept the prod snapshot (snapshot-prod-20261124) with it: 10,577
documents, 8,425,432 segments, 8,425,432 cuts, 0 join failures, 0 segments not found. The comment
is on the PR.

## w-public: what is settled

The decisions are in the database session's note `note_rules_2` on tracker row `w-public`:

- The pipeline upserts `sources` and `content_items` itself, keyed by `podcast_scraper.db.corpus_keys`
  (#2569). The lookup is `(source_id, external_item_id)`.
- Different body bytes make a new version superseding the old. `change_kind` is `created` on v1;
  later versions name the cause (`repunctuated`, `retranscribed`, `translated`, `corrected`).
  Same bytes: no write.
- Variants: `raw` is the source text and `en` the English render. The analysis text is `en` if it
  exists, else `raw`; `cleaned` and `adfree` derive from it. `.anon.txt` is never written, and two
  files mapping to one variant are refused.
- `cleaned` and `adfree` are cut one segment per line with null times (no timing for them on the
  prod snapshot). `speaker_id` stays NULL until the identity writers.
- `extractor_version` is `name/package-version`, `feed/1`, `standfirst/1` or `guardian/api`.
- One sync per item: after the metadata stage, after translation, repunctuate and
  retranscript_only, and after an article body is written.
- On failure, log and count before the cut-over. One strict setting fails the item after it; the
  runbook switches it.

The planner's call-site map (subagent, line numbers not re-checked by hand):

- transcript bodies: `workflow/episode_processor.py` `_save_transcript_file` :786,
  `_write_transcript_file` :5464, `_save_transcript_segments_file` :819, the repunctuate commit
  :4065 and the retranscript rewrite :3468;
- `.cleaned.txt`: `workflow/metadata_generation.py` :2865 and :2943;
- translation: `translation/artifacts.py` `_swap_in_translation` :493;
- article body: `workflow/text_in.py` :280;
- the metadata write the item sync hooks after: `metadata_generation.py` `_serialize_metadata`
  :3456 and :3484;
- `.adfree.txt` and `.anon.txt` are written by stage hooks outside public.

Risk to check: `text_in.py:280` writes the article body with no `encoding=`, while the code hashes
`item.text.encode()` as UTF-8.

## Waiting on others

| Item | Needs | From |
| --- | --- | --- |
| The writer's item upsert | Where the metadata-to-row mapping lives (title, dates, `duration_s`, language, flags, `type_attrs`, `served`) and what `served` 0, 1 and 2 mean; asked 12:20Z, no answer yet | database |
| The writer's first PR | #2565, #2566 and #2569 merged. Until the mapping answer it writes documents and segments only for an item whose row exists, and counts the rest. | CI, database |
| `search_chunks` and `embeddings` tables | A corpus revision after `corpus_0005`; agreed the database session writes it, and public fills and reads them | database |
| Insight and quote hydration, `r-graph-engine` | #2567 (`corpus_0005`, entities), then `s-intel` | database |
| `r-workbench` | `r-graph-engine` and `r-projections` | public, database |
| `n-api-versions` | documents version rows and `n-w9` | database |
| `public-pd-dupes` | tonight's nightly; read-only check that #2542's duplicate is not offered again | the nightly |

Review asks to expect: the database session asks public to review each importer PR
(`s-importer`) and the `w-leases` PR, since they touch public's artifacts and pipeline code.

## Corrections: where this session went wrong

- My fixture sweep script for the cutter printed "all joined back" unconditionally. That held
  only because `cut_body` raises on a mismatch; the database session pointed it out.
- The cutter's first version put a speaker label in the turn before it, against the agreed rule.
  The test I wrote for the rule caught it before the PR.
- I stopped `make gates-changed` for #2565 after 9 minutes in pytest. It had selected the e2e and
  analytical suites, because everything imports `db/engines.py`. I ran black, isort, flake8, mypy
  and the db unit tests (214 passed) instead, and said so in the PR.

## Working notes

- `ssh deploy@prod-podcast` is refused from the workstation (publickey). Prod facts come from the
  prod session.
- Running a worktree's tests: link `.venv` to `/work/closelistening/public/.venv` and set
  `PYTHONPATH=<worktree>/src`, because the venv's editable install points at the main checkout.
  The shell guard blocks `PATH=` overrides and inline Python; write a script file.
- `make test-db-up` in a worktree starts a container named after the worktree. Remove it with
  `make test-db-down` when done.
