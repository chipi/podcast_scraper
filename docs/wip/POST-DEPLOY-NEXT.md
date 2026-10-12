# Post-deploy list — next prod deploy (naming arc, #2304 Track A + B)

Started 2026-10-12. Prod deploys no earlier than 2026-10-13 (operator). **Every prod-changing step
needs the operator's go, one by one.** Earlier lists: `POST-DEPLOY-2026-10-09.md` (main checkout,
untracked; that deploy is done).

## What this deploy carries (naming)

- `ea1af916f` (#2337, merged 2026-10-12): the naming ladder fixes (#2329, #2330, #2331), the
  index prune fix, the replay/atlas tools, and the ladder guides.
- Track B, if merged before the deploy: the coverage guardrail (S8) and the record repair (S7),
  both `fix/naming-track-b-2304`.

## Pre-deploy

- **B0 stop the backfill chain.** A deploy kills a running reprocess job. Let the running batch
  finish (or stop it cleanly and re-queue its unfinished episodes), and do not start the next.
  The chain: `scratchpad/batch_chain.sh` (operator's standing approval covers batches, not this).
- **B1 CI green on main at the deploy SHA** (Python application full run, Docker Build & Test,
  stack test). Read Snyk's message: on 2026-10-12 it errored on quota on every PR.
- **B2 empty queue.** The nightly starts at 23:00 UTC.

## Post-deploy: naming

- **N1 deployed SHA** contains `ea1af916f`.
- **N2 record repair (S7, #2334).** Served records still carry the names the OLD code chose; the
  new ladder only applies to new or reprocessed episodes. Repair = re-run naming from each episode's
  stored diagnostics and rewrite its speaker record (text files only: no audio, no ASR, no LLM, no
  DGX).
  - Before: `scripts/measure/naming_coverage.py --corpus /app/output` (baseline).
  - **Scope = exactly what this deploy changes, not "re-run naming".** Measured 2026-10-12: stored
    records vs a replay of the merged code differ on 994 voices in 775 episodes, and 486 of those
    are names the replay REMOVES (282 stored from the feed's hosts, 137 self-intros, 43 forced; runs
    from Aug-Sep). Those are older code's arithmetic names that today's rules withhold on purpose,
    many of them correct (Gergely Orosz, Lenny Rachitsky); a blanket re-run would delete them. The
    repair touches a voice only when its stored record still equals what the DEPLOYED code
    (`sha-3e06151`) produces AND the merged code produces something else: the changes the arc
    replayed and read.
  - **It is migration 0029** `naming_arc_voices_repaired` (to 2.7.23), a frozen list of 83 voices
    (`NAMING_ARC_VOICES`, each read). Prod is at 2.7.21: **0028 runs first** (strict id order; it is
    the player 1.0.4 deploy's, check its own post-deploy notes), then 0029.
  - Dry-run measured on prod 2026-10-12 with the module loaded from a temp file (read-only): would
    rewrite **76 episodes, 83 of 83 voices, 0 refused**, in 10 s (quotes_moved 5,633, transcripts
    re-rendered 138, voices_named 56, roster_entries_unnamed 7). Expect the same at deploy unless
    those episodes are reprocessed in between (the backfill): a moved record REFUSES, untouched.
  - Steps: `upgrade status` -> `upgrade run --to 2.7.23 --dry-run` (read the per-episode list;
    refusals listed) -> operator OK -> apply -> `verify` -> spot-check 3 (Pragmatic Engineer
    "Code Complete" Steve McConnell; Talk Eastern Europe Alexandra Karppi; No Priors Chess.com
    Erik Allebest on SPEAKER_00) -> coverage again (`naming_coverage.py`, before/after).
  - Undo: `upgrade undo 0029` restores from `.podcast_scraper/upgrade-backups/0029/`.
- **N3 stale index rows (#2334).** 28 episodes serve superseded insight rows (the prune guard
  refused correct prunes from 2026-10-03 until `6551cecf9`). Re-index those 28 with the operator's
  OK; the list is in this file once prepared.
- **N4 first nightly after the deploy.** Naming on new episodes looks sane (read the rosters of
  every new episode); `decision_trace` present in new `.speakers.diagnostics.json`.
- **N5 guardrail (S8).** The nightly's batch finalize writes
  `<corpus>/.podcast_scraper/naming_coverage.json` (~46 s scan on prod); the api's `/metrics` then
  serves `podcast_corpus_episodes_no_named_voice{language="en"}`, `podcast_corpus_naming_episodes`,
  `podcast_corpus_quotes`, `podcast_corpus_named_quotes` and
  `podcast_corpus_naming_coverage_age_seconds`. No snapshot yet = no series (never zero).
  - After the first nightly: the series is in VictoriaMetrics (`instance="prod-podcast"`).
    Expected value: 289 of 2,619 before the N2 repair (measured 2026-10-12 with this code on prod),
    lower after it.
  - Alert rules (agentic-ai-homelab `infra/observability/backend/grafana/provisioning/alerting/rules.yaml`,
    podcast group next to `podcast-ingest-stalled`; add with the operator's OK, then the provisioning
    reload). Drafts:
    - `podcast-naming-coverage-rose`: `last_over_time(podcast_corpus_episodes_no_named_voice{instance="prod-podcast",language="en"}[1h]) - min_over_time(podcast_corpus_episodes_no_named_voice{instance="prod-podcast",language="en"}[3d]) > 10`, `for: 1h`, severity warning (model: `bugfix-mass-stuck`).
    - `podcast-naming-coverage-stale`: `max(podcast_corpus_naming_coverage_age_seconds{instance="prod-podcast"}) or vector(99999999) > 172800`, `noDataState: Alerting`, `for: 1h`, severity warning (model: `podcast-ingest-stalled`).

## Carried over

- **A10 calendar:** on 2026-10-25 change the nightly cron `0 23 * * *` to `0 0 * * *` (DST).
- **#2187 backfill:** continues between deploys (operator standing approval for batches).
  Left 2026-10-12 midday: 7 broken-to-end episodes, then 44 partial; 8 comma ids, 2 out of spec
  and 3 Gray Area need a decision.

## Progress log
