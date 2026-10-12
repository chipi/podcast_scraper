# Post-deploy plan — next prod deploy (2026-10-13 or later)

The ONE list for this deploy. It merges every open item from the earlier lists:
`POST-DEPLOY-2026-10-09.md` (main checkout, untracked; that deploy and its second one are done,
and its open leftovers are carried below), `HANDOVER-prod-agent-player-1.0.4-post-deploy.md`
(the player session's), PR #2326's body (licence), and the naming arc's own items.
**Every prod-changing step needs the operator's go, one by one.** Prod fixes are text files only:
no DGX relabel, no LLM, no full re-index.

## State before the deploy (read 2026-10-12 ~16:30 UTC; re-read before starting)

| | |
| --- | --- |
| Prod images | `sha-3e06151` (2nd deploy of 2026-10-09) |
| Target | main `29f692d46` or later, with green CI (below) |
| Corpus ledger | `2.7.21` (0024–0027 applied); pending after the deploy: **0028, 0029** |
| Disk | 93 GB free on `/`; no upgrade snapshot left over. Each `upgrade run` takes ~13 GB |
| Backfill | #2187 chain running (batch 12, run 38273648211, 3 eps, ends ~20:00 UTC) |

## What the deploy carries

| Change | PR / commit | Needs after the deploy |
| --- | --- | --- |
| Naming ladder fixes, tools, docs | #2337 `ea1af916f` | first-nightly check (step 9) |
| Naming Track B: audit guards, record repair, coverage guardrail, embeddings capture | #2339 `29f692d46` | 0029 (step 3), metrics + alerts (steps 9–10), DGX redeploy (step 8) |
| Player 1.0.4: Moments, Brief search, Mine, Discover 2 + two player fixes | #2305 `e615e6d7d`, `4e9b1dc62`, `6ef835b43` | 0028 (step 2), milestone dedupe (step 5), checks (step 7) |
| Licence and copyright declarations | #2326 `37ff6563d` | licence backfill (step 4) |
| DGX converge no longer starts retired services | `a6375152a` | nothing on prod; makes step 8 safe |

## Before the deploy

- **B1 CI green at the deploy SHA.** Python application (full: unit, integration, e2e — these run
  on main only), Docker Build & Test, stack test (publishes `sha-<short>` for all images),
  viewer, player. Red is not advisory. Snyk errors on the account quota ("used your limit of
  private tests"); that is not a finding, but read its message each time.
- **B2 stop the backfill chain.** A deploy kills a running reprocess. Let the running batch finish
  or stop it cleanly; then stop `scratchpad/batch_chain.sh` so it does not start the next. The
  standing approval covers batches, not deploy timing.
- **B3 empty queue.** No reprocess container (`docker ps | grep reprocess`), not within the
  nightly (23:00 UTC start, ~3 h).
- **B4 deploy.** Dispatch `deploy-all-prod` at the target SHA; the operator clicks the prod gate.
  Never approved by me.

## After the deploy, in this order

### 1. Confirm

- Every container on the new `sha-…`; both smokes (operator + player) green on the first try.
- `upgrade status` in `compose-api-1`: 2.7.21, pending 0028 and 0029.

### 2. Migration 0028 `transcript_chunk_times` (2.7.22, player 1.0.4)

Times the transcript passages already in the search index from `*.segments.json` (rows sit at
0 ms today). Writes LanceDB rows in place + `search/metadata.json`; undo record first.

- `upgrade run --to 2.7.22 --dry-run`. Read the line: `N … at 0 ms; T timeable; left alone:
  a no segments, b transcript changed since indexing, c no overlapping segment, d no char span`.
  **Stop and report if `b` or `d` is large.** It has never run on prod data.
- Operator OK → apply → `verify` → spot-check: MCP `search_corpus` for a known sentence, the
  transcript hit carries a non-zero `timestamp_start_ms`.
- Undo: `m0028_transcript_chunk_times.undo(Path('/app/output'))` (command in the player handover).

### 3. Migration 0029 `naming_arc_voices_repaired` (2.7.23, naming)

Rewrites the 83 voices (76 episodes) whose stored name/role is the old code's and differs from
the merged ladder's; each entry was read. An entry applies only while the stored value is the old
one; a voice the backfill reprocessed since 2026-10-12 is skipped (already new) or refused.

- Coverage before: `scripts/measure/naming_coverage.py --corpus /app/output` (copy it in).
- `upgrade run --to 2.7.23 --dry-run`. Measured 2026-10-12: 76 episodes, 83/83, 0 refused, 10 s.
  Read every refusal.
- Operator OK → apply → `verify` → spot-check 3: Pragmatic Engineer "Code Complete" (Steve
  McConnell), Talk Eastern Europe (Alexandra Karppi), No Priors Chess.com (Erik Allebest on
  SPEAKER_00) → coverage after (expect below 289 nameless of 2,619).
- Undo: `upgrade undo 0029` (backups under `.podcast_scraper/upgrade-backups/0029/`).
- After both verify: delete the two snapshots (operator OK).

### 4. Licence backfill (#2326)

`python -m podcast_scraper.cli refresh-feed-metadata --corpus-dir /app/output [--feed-id ID]`
in `compose-api-1`. **It has no dry-run.** It re-fetches each feed's RSS and, in every existing
metadata file, refreshes the feed block (description, image, last_updated, category) and fills
MISSING licence records (show → `feed.licence`, episode → `episode.licence` by guid); a stored
record is never replaced. Runs after 0029 (both rewrite metadata files).

- One feed first, with a known declaration (a feed with `<podcast:license>` or a copyright line):
  read the diff of one metadata file before and after.
- Operator OK → every feed. Read the summary (`feeds seen / refreshed / files updated / skipped`)
  and the skipped list.
- Check in the player: an episode of that feed shows its licence.

### 5. Duplicate playback milestones (player 1.0.4)

`scripts/ops/dedupe_playback_milestones.py` (not in the image: copy into `player-api-1`).
Dry-run → read `{"mode": "dry-run", "users", "files", "dropped"}` → operator OK → `--apply`.
Exit 1 = re-run. Undo `--undo`. Remove the copies from `/tmp`.

### 6. Stale index rows (#2334)

Episodes serving superseded insight rows (the old prune guard refused slightly smaller
reprocesses; fixed in #2337). The list grows with every backfill batch, so build it now:
VictoriaLogs `_time:[2026-10-03T00:00:00Z, now] "REFUSING to prune"` → each `episode_id=`;
drop the two TRUE partials (BBC `m002w616` 31 vs 86, `m002vyjt` 39 vs 79) unless rebuilt since.
34 distinct episodes by 2026-10-12 10:11 UTC. Operator OK → re-index exactly those → check
`257da0ba-a334-11f1-8c3b-0feea6ade97e` (111 rows before, 55 after).

### 7. Player 1.0.4 checks (smoke account)

- Moments = Brief: anchor.fm 5a072801 "#10 Michael Montgomery: Psychophobia…" has
  `has_moments: true` and non-empty `/moments`; "AI Sychophancy with Steve Rathje", "How a
  Sardine Gets From the Ocean…", "Dylan Patel - Inside the Trillion-Dollar AI Buildout", "Why
  Apple Is Heading Into a Pivotal Moment", "Rory Johnston on Why His $200 Oil Prediction…" have
  `has_moments: false` and no door.
- Search: a speaker/topic-scoped search is slow (~3 s) only the FIRST time after an ingest; the
  Brief's search finds other word forms ("invested" → "investing").

### 8. DGX pyannote server redeploy (voice embeddings, N6)

The app deploy alone stores embeddings only for in-process diarizations; the DGX server sends
them after it is redeployed with the new `app.py`.

- Only when nothing diarizes: no backfill batch, not the nightly. Check `gpu-mode` first.
- `make dgx-deploy` is the full converge (rebuilds the pyannote image on `app.py` change, then
  `compose up -d`). Run pyinfra `--dry` first and read it; operator OK; run; `make dgx-verify`.
- Verify on the next diarized episode: `.speakers.diagnostics.json` has `voice_embeddings` whose
  `by_voice` keys equal its `voices[].voice`; `/api/corpus/text-file` refuses that file (400);
  MCP `episode_speaker_roster` has no `voice_embeddings`.

### 9. The first nightly after the deploy

- Ingest: what came in, ERROR lines, speed per audio-hour (baseline: ASR 112–203 s, LLM ~500 s),
  no kill, dup edges 0.
- Naming on every new episode: read the rosters. Cases seen on the 2026-10-10 nightly to re-check
  if their feeds publish: No Priors (Misha Laskin listed as host), Talk Eastern Europe (Alexander
  Carpe + Alexandra Karppi both hosts), Hard Fork (Max Reed as host). `decision_trace` present.
- Timed transcript rows: a new episode's transcript hit has a non-zero `timestamp_start_ms`
  without 0028.
- Coverage guardrail: `<corpus>/.podcast_scraper/naming_coverage.json` written by the batch
  finalize; `/metrics` serves `podcast_corpus_episodes_no_named_voice{language="en"}` and its
  siblings; the series appears in VictoriaMetrics (`instance="prod-podcast"`).
- If step 8 ran: the new episodes carry `voice_embeddings`.

### 10. Coverage alerts (homelab repo, operator OK)

In `agentic-ai-homelab` `infra/observability/backend/grafana/provisioning/alerting/rules.yaml`,
podcast group next to `podcast-ingest-stalled`, then the provisioning reload:

- `podcast-naming-coverage-rose`: `last_over_time(podcast_corpus_episodes_no_named_voice{instance="prod-podcast",language="en"}[1h]) - min_over_time(podcast_corpus_episodes_no_named_voice{instance="prod-podcast",language="en"}[3d]) > 10`, `for: 1h`, warning.
- `podcast-naming-coverage-stale`: `max(podcast_corpus_naming_coverage_age_seconds{instance="prod-podcast"}) or vector(99999999) > 172800`, `noDataState: Alerting`, `for: 1h`, warning.

### 11. Resume the backfill

Restart the chain after steps 2–6 are done (and outside step 8's window). Sizing: 1.2× the
330-min runner cut-off at the measured pace (1.3× real time on 2026-10-12, when the DGX was
shared with a translate workload).

## Carried over (not deploy steps)

- **A10 calendar, 2026-10-25:** nightly cron `0 23 * * *` → `0 0 * * *` (DST).
- **#2187 backfill decisions (operator, owed).** The chain skips these until decided:
  - 8 episode ids that contain a comma. The server side reads them since `8cdc9f90e` (#50, on
    prod), but `reprocess-prod.yml` splits its inline `episode_ids` input on commas, so they cannot
    go through the chain. Route: a newline-delimited work-list file on the box passed as
    `episode_ids_file` (writing that file is a prod change: operator OK). Run them, or leave them?
  - 2 episodes out of spec (their feed's reach marks them out of `feeds.spec.yaml` scope).
  - 3 The Gray Area episodes: the feed is not in `feeds.spec.yaml`, and the reprocess only walks
    spec feeds. Re-add the feed, or leave them.
  - Refused repair windows (16+ by 10-10, kept their old text): send to the #2187 owner?
- **Phase C (operator decides, after the deploy).** Nothing planned yet. Proposed order: voice
  identity, since step 8 starts the capture (let the bank build, measure how well voices
  separate, then a gated voice-identity rung); K8 with its cross-post guard (#2338); rebuilding
  embeddings for processed episodes only if the measurement says it pays. The parked S5 (96
  publisher-transcript episodes, #2332) and F7 (The Rest Is History, 30) go first or after, as
  the operator chooses.
- **P5 entity search (from 10-09):** still unverified; after an enrichment, search a new
  storyline/theme name (label snapshots `~deploy/topic_*clusters.snapshot-20261010.json`).
- **person_web (#2111):** its 281 misses were written 2026-09-17 with a 30-day TTL, so a plain
  enrichment reconsiders them from 2026-10-17; `refresh=true` does it now. Not orgs (#2110).
  Whether a refresh already ran: not checked.

## Not in this deploy

K8 (#2338, left out); voice identity stages 2–3 and rebuilding embeddings for processed
episodes (needs re-downloaded audio; prod keeps 147 audio files); the 96 publisher-transcript
episodes (S5, #2332) and The Rest Is History's 30 (F7), parked.

## Progress log
