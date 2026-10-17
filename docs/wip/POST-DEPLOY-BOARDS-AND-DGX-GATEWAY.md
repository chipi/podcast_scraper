# Post-deploy plan — player 1.0.4: Discover, boards, board summaries, the DGX through LiteLLM

Written 2026-10-13 for the deploy of `feat/player-1.0.4` (109 commits ahead of `main` at
3dc44e395). The previous deploy (2026-10-12/13, `sha-c6cdc5d`) carried the first part of 1.0.4
(#2305: migration 0028, Moments = the Brief, search scopes); this one carries the rest. No
migrations, no new env beyond what is listed below. Every step that touches prod needs the
operator's OK, one at a time. Read the state again before starting.

## Status 2026-10-15: deployed

`main` was deployed as `sha-aaadbb2` (`deploy-all-prod.yml`, which runs `deploy-prod.yml` and
`deploy-player.yml`), then `deploy-config.yml` and `deploy-litellm.yml`. What differs from the
steps below:

- **Step 1 is superseded:** Orrery got its own Langfuse projects, `orrery-prod` (its prod key) and
  `orrery-dev` (dev default), instead of staying in `agents`.
- **Step 2, secrets:** the GH secrets `LITELLM_LANGFUSE_PUBLIC_KEY/SECRET_KEY` held `-`
  placeholders, so the first gateway deploy got 401 from Langfuse (about 10 minutes of traces
  lost). They were set from the mini's key file and the gateway redeployed. Check the secrets'
  lengths before a gateway deploy.
- **`deploy-config.yml` restarts Caddy when a vhost file changed** (it did at 05:57 on
  2026-10-15, with the new `operator.caddy`; a later run with nothing changed logs "no vhost
  changed — caddy not restarted"). That restart, during the player smoke, caused a 521 on that
  smoke; a re-run passed. Read the dry run's vhost lines first, and do not run it while a smoke runs.
- **MCP warm-up:** for about 3 minutes after an API restart, entity lookups block while the
  registry warms (step 8.1). Wait for it before running a smoke or an MCP check.
- **`litellm.alloy` now ships with `deploy-config.yml`** (host level, every surface; 3e3d836e6,
  b55358321). Before, only `infra/deploy/deploy-litellm.sh` copied it, so the box ran a stale copy
  and litellm logs had no `service` label. Live since 07:38 UTC (run 38700011300). The Alloy step
  now validates the whole set first and fails on a refused reload.
- Step 7: 7.1, 7.2, 7.3, 7.4 and 7.5 done (themes `id_history` present after batch 26's rebuild,
  07:38). 7.6 done by the homelab session (the Grafana items were already live;
  `vps/podcast-app.json` pushed as v3). 7.7 checked (see "After the deploy").
- Step 8 (one run at a time; batch 26, run 38689720595, finished green 07:38, 18-19 min per
  episode):
  - **8.1 / 8.7 done**; 8.2 waits for the first nightly.
  - **8.3 done** (run 38700512700, 09:52): 24 episodes, 7 repaired, 17 left as they were (every
    window refused), 0 failed. Only 9 of 38 windows passed (the plan expected about half):
    refused `lost_words` 13, `too_many_new_words` 10, `word_count_changed` 4, `repetitive` 1,
    `still_unpunctuated` 1. Look at the refusals before re-running these 17.
  - **BizNews "BN Daybreak"** (never attempted, outside every list): `repunctuate_only`, 1 of 1
    window repaired (run 38719287784).
  - **8.6 done** (run 38716746982): Gray Area "How trauma became a status symbol", full reprocess,
    repaired 1/1, host named, 1 window repaired. Its corpus dir is the old Vox feed
    `feeds.megaphone.fm/theezrakleinshow`.
  - **Re-scan after these (by episode guid, 2,530 episodes):** the 8.3 list missed nothing. 333
    asr records match no metadata file; 332 are older copies of episodes whose newest record does
    match, 1 has no unpunctuated window. Still flagged: the 18 above, plus 12 in the backfill pool.
  - **Next, after the next app deploy** (operator, 2026-10-15: a deploy during a reprocess is
    untested, and 4ae796413 gives each episode its own Langfuse trace with named stages):
    1. Backfill batches 27+: 15 episodes, 1,149 audio-min, about 5 h in 3 batches (measured about
       4x real time).
    2. 8.4 relabel: 91 episodes, list in hand (replay `c6cdc5d` vs `aaadbb2` modules, 2,684
       episodes: 101 voices change; 40 gain a name, 57 promo/ad voices lose a wrong one, 4 other
       losses). First a 5-episode batch to measure (6-11 h for all is a guess: no relabel run has
       been timed), then the rest in batches that fit the runner and the nightly. No overlap with
       the backfill pool.
    3. 8.5: recompute after 8.4 (Rest Is History episodes whose host seat is still unnamed).
       The original 15-seat list was in the replay kit and did not survive the container
       recreation.
- **Gateway:** `deploy-litellm` from c3b716195 at 09:53 sets `LANGFUSE_TRACING_ENVIRONMENT=prod`.
  Smoke: 11 aliases served, completions end to end. Still LiteLLM 1.95.0. Homelab confirmed
  environment=prod on the first traces (10:07). The image tag is pinned since e20cd3e59 (was
  `main-stable`, which had moved past the running image).
- **Edge:** Caddy reloads through a 0600 admin unix socket instead of restarting (60dea6847,
  ADR-114 amended, THREAT_MODEL T-02). Rolled out 08:03 UTC: `apply-edge.sh` / `verify-edge.sh`
  refreshed from main as root, once (backups in `/root/*.bak-20261015`; the installed copies were
  from July and still installed a Grafana Cloud systemd Alloy), then one last restart (1 of 150
  requests got a 521). After: a reload under load answered 150/150, verify-edge 15 pass. Orrery's
  vhost script reloads too (orrery 4ca9fdd4795).

## Deploy of `sha-f35a446` (2026-10-15 14:36–16:02 UTC) — what failed, and the fixes

Prod ended healthy on `sha-f35a446`: every container, all smokes green on the re-runs, player
smoke 52/52 (`smoke-player` run 38765316477). The way there:

1. **Run 38752921829 failed at the manifest check:** dispatched with `image_sha=f35a446b8`
   (9 characters), but the images are tagged `sha-<7>`. Prod was untouched. Fix: `e9396defb`
   (the guard shortens to 7 characters and checks the tag is published before deploying).
2. **Control-plane smoke, `/api/artifacts`:** it took about 8.5 s per call (three corpus walks,
   plus a publish date for 9,531 files when 1,191 are listed), against the smoke's 10 s limit.
   Under start-up load it went over, and every 15 s retry started another listing, so calls piled
   up to about 600 s with the API above 100% CPU. A re-run on the idle API passed. Fixes:
   `c90537ff0` (1.6 s, identical listing, proven on prod) and `b26c2bf0f` (concurrent calls share
   one computation).
3. **Operator smoke, read plane:** `/api/index/stats` hit about 200 s in the same pile-up (homelab
   traces). Fixes: `c3069005a` (one corpus walk instead of two, cold call about 5.6 s) and
   `b26c2bf0f`. The remaining 3.3 s newest-file scan is parked: it needs a change to the shared
   `discover_metadata_files` rule.
4. **Player smoke, "Search renders grouped results":** failed on both browsers. `7894ad89f`
   replaced the "N passages across M episodes" line with "Episodes · N", which the smoke still
   waited for. Fix: `4ba2f9878`. (`85f98f448` reformatted the spec with default Prettier and was
   reverted in `1a2b47b09`.)
5. **Caddy reloaded, not restarted:** `ActiveEnterTimestamp` is still 08:03:11 after both deploys
   (a fallback restart would have moved it). The scripts don't log which path they took.

Not measured: how long the API's cache warmer keeps it busy after a start. It logs no step
timings, only its start and failures at DEBUG.

## Deployed 2026-10-16 19:28-19:44 UTC: `sha-ded80e0` (run 38955105369)

- The operator smoke failed once: `/api/index/stats` did not answer during the first cache warm
  (308.6 s, `reel_clips` 243.3 s). It passed on rerun after the warm. Cause and fix:
  `3d7931cc9` (mobile session): reel clips are scored in a child process, not in the api.
- **Checks for the NEXT deploy (must carry `3d7931cc9`):** the `cache warm done` line still lists
  `reel_clips`; no "reel clips: isolated scoring failed" warning in operator-api / compose-api;
  `/api/index/stats` answers in seconds while the warm runs; the operator smoke passes on its
  first attempt.
- **Also for the NEXT deploy: `ed09aceaf`** (operator decision 2026-10-17, ingestion session): a
  language refusal meets the request. A `TranscriptLanguageMismatch` is logged at WARNING, filed as a
  `policy` incident (was `hard`), ledgered `skipped` (was `failed`); the work-list says "repaired N/N
  · N of them refused and not served (TranscriptLanguageMismatch N)" and exits 0. On `sha-ded80e0`
  the 2026-10-16 Turkish run (38976538076) went red for exactly this, though both refusals and
  withdrawals worked.
- **Fixed on prod 2026-10-17 06:17 UTC:** the nightly failed on 10-16 and 10-17 (401 from a stale
  direct `vllm_api_base: http://dgx-llm-1:8003/v1` in `viewer_operator.yaml`). Line removed, backup
  `.viewer/operator-config.before-vllm-direct-removed-20261017T061742Z.yaml`; `ad47a8bc0` warns when a
  config file overrides a profile's `vllm_api_base`. Check: the 2026-10-18 00:00 UTC nightly
  succeeds and deepens the 13 new feeds.

## Next deploy (candidate: main head when the operator names it) — the whole list

Written 2026-10-16. Prod runs `sha-f35a446`. Each prod step needs the operator's go.

**Checked 2026-10-16 against main `56f9af986` (re-run the first three at the candidate):**

- No Caddy vhost or Alloy file changed since `f35a446`, so no `deploy-config`:
  `git diff --stat f35a446 <candidate> -- infra/caddy infra/observability` prints nothing.
- Compose changes only add `service.version=${PODCAST_RELEASE:-unreleased}` to the OTEL tags.
  `PODCAST_RELEASE=sha-f35a446` is already in `/srv/podcast-scraper/.env` and in all five podcast
  containers, so the deploy sets it.
- No migrations.
- `deploy-operator.yml` now alerts with `PROD_SENTRY_DSN_API` (the secret exists, set
  2026-10-15).
- The guard (`e9396defb`) run locally against ghcr: `f35a446b8`, `sha-f35a446` and `f35a446` all
  resolve to `sha-f35a446`; `0000000` fails (HTTP 404). In Actions, run 38861127936
  (`image_sha=0000000`): guard failed, all three surfaces skipped, prod unchanged. That run's
  report said "deployed sha- " and went green; `b8a677dc6` makes it fail.
- Not checked: an EMPTY `image_sha` (the guard's package listing needs `read:packages`, which my
  local token lacks). Pass an explicit sha.

**Before:**

- Main's CI green on the candidate: Python application, Docker Build & Test, Snyk, CodeQL.
- No reprocess running (`gh run list --workflow reprocess-prod.yml --limit 3`) and no smoke.

**Deploy:** `deploy-all-prod.yml`, `confirm=DEPLOY_ALL`, `image_sha=<candidate>`.

**After, in order:**

1. **Images and smokes.** `docker ps --format '{{.Names}} {{.Image}}'`: every podcast image on
   `sha-<7>`. Control-plane smoke: `/api/artifacts` answers on its first attempt (`c90537ff0`,
   `b26c2bf0f`). Operator smoke green (`c3069005a`). Player smoke green (`4ba2f9878`). Caddy
   reloaded, not restarted: `systemctl show caddy -p ActiveEnterTimestamp` still 2026-10-15
   08:03:11.
2. **Cache warmer:** `docker logs compose-api-1 2>&1 | grep 'cache warm done in'` gives one line
   with every step's time (`37f0a8fd1`). Record the total here.
3. **Logs:** no uvicorn `INFO:     <ip>` access lines in the API logs (`8ec79649e`).
4. **Traces:** player-api, player-mcp, operator-public-api and podcast-pipeline carry
   `service.version=sha-<7>` (`5d90d71f8` plus the compose tags).
5. **Langfuse:** traces carry `release`; the run-level trace is named "run <id>" (`a0747249c`,
   `a96c56a1d`). Homelab confirms.
6. **GlitchTip:** an operator smoke failure lands in podcast-api, not player-api (`2343dea12`).
   Seen only when a smoke fails.
7. **Tell homelab on mini:** the deploy run id and the sha each surface reports.
8. **Not checkable from prod:** `3819acff6` changes `AppProcessPlugin.java` and reaches Android
   users only with a new build.
9. **BBC plays over https (`f4f3efb42`):** `audio-source` for an In Our Time and a Curious Cases
   episode returns an `https://` URL that answers `206 audio/mpeg`; an https feed's URL comes back
   unchanged. Play one BBC episode in the web player: no "Blocked by the browser".
10. **Ingestion arc (from the ingestion session, 2026-10-16): the candidate must be `08a41f01a` or
    later** (the audio-sync floor of item 12 is later: `75304e525`). It carries `fe2316648` (#2363
    title names as spelling references), `fc4e0dbda` (#2364 Patreon shout-outs are ad voices),
    `28b10ef3c` (#2366 question-lead host seat), `e7234aaec` (#2370 word timestamps), `e49583eb4`
    (#2375 / RFC-130: an episode in another language than its show is identified and translated, or
    refused with the language recorded) and `669bcbf29` (a language refusal marks every run of the
    episode unusable, so the catalog, player API, digests and MCP hide it; the search indexer
    withholds unusable runs and deletes the rows of an episode with no servable run). Their gates on
    `669bcbf29`: full `make ci-fast` green; after the rebase onto `9b11d49b9`: format, lint,
    docstrings, mypy and 2,776 affected tests green. On today's snapshot no episode is unusable, so
    the deploy alone changes nothing.
    **Follow-up, approved by the operator 2026-10-16 (withdraw both): full reprocess (a relabel does not run the guard) of the 2
    Turkish Ottoman History Podcast episodes,** about 37 + 48 audio-min:
    `tag:blogger.com,1999:blog-1793063735579568706.post-7556576044575364143` (Soykırımın
    Bürokrasisi) and `tag:blogger.com,1999:blog-1793063735579568706.post-1515154927178009499`
    (Cümle Fener Burada). Expected:
    - each is REFUSED; `<corpus>/.language_refusals/<sha1(episode_id)>.json` holds
      `"detected": "tr"`; the run log shows "REFUSING episode:", a `TranscriptLanguageMismatch`
      incident, and "withdrew a served run of a refused episode: …metadata.json" once per old run
      (`run_5da86385…20260912` and `run_ed9ab40e…20261002`);
    - the next index build (incremental is fine) logs "two-tier index: withdrew N row(s) of
      unservable episode(s)";
    - neither episode is in the catalog, in `/api/app/search` results, or reachable via the player
      API.
10a. **MCP tool events reach Umami (`ab8290f6f`, from the homelab session):** before it, Umami's
    bot filter dropped every player-mcp event (UA "podcast-mcp/1.0 (+server-side)" got HTTP 200
    `{"beep":"boop"}`, nothing stored). After the deploy, make one MCP tool call and tell the homelab
    session; it checks the event lands in Umami with hostname `mcp.closelistening.app`.
10b. **MCP for every user + participant attribution (homelab session, 2026-10-16): the candidate
    must be `bf243ded2` or later** (it includes `75304e525`). `529531f17`: new accounts and legacy
    profiles get `mcp_access=true`, an explicit admin revoke stays false. `28e1f6451`:
    `/internal/mcp/verify` returns `analytics_id`, which every tool call carries (Umami payload id,
    span `mcp.participant`, log `participant=`); every tool's `mcp_tool_calls_total` exists at 0 from
    start (needs api and mcp on this sha). `bf243ded2` is dashboard only (live via obs-sync).
    `infra/observability/player.alloy` changed in comments only: no `deploy-config`.
    **One-time grant, on the operator's go (writes prod user data):** existing profiles still hold
    the old default `false`. Inside `player-api-1` (`APP_DATA_DIR=/app/appdata`):

    ```bash
    docker exec player-api-1 python -m podcast_scraper.server.app_users_cli --data-dir /app/appdata grant-mcp-all          # dry run
    docker exec player-api-1 python -m podcast_scraper.server.app_users_cli --data-dir /app/appdata grant-mcp-all --apply
    ```

    It skips a user whose latest `admin.user.mcp_access` entry in `audit.jsonl` is a revoke; stderr
    prints "N granted, M already had it, K skipped". Send the dry-run counts to the homelab session.
    **Check:** after one MCP tool call (10a), the homelab session checks 2 zero-registered series
    per tool, `participant=` log lines, `mcp.participant` spans, Umami `mcp_tool:<tool>` events under
    a participant, and GlitchTip 29.
10c. **The observability MCP observes itself, and MCP calls are authorised per request (homelab
    session, 2026-10-16): candidate must include `0f821b3c8` and `7709337b4`.**
    - `0f821b3c8`: player-obs gets its own errors, traces, logs, metrics and Umami events
      (`docs/guides/OBSERVABILITY_CONTROL_PLANE.md`, "Its own signals").
    - New deploy secret `PROD_SENTRY_DSN_PLAYER_OBS` → GlitchTip project `player-obs` (id 33).
    - The obs image gains sentry-sdk, the OTel SDK + OTLP exporter + ASGI instrumentation, and
      prometheus-client.
    - `player.alloy` gains the `player_obs` scrape: **run `deploy-config`**. It validates with
      `alloy validate` v1.17.0 (checked locally against a stub base, with a negative control).
    - `7709337b4`: a tool call is scope-checked, logged and traced as the request that carries it,
      not the request that opened the session. Before it, a read-only token could call a write tool
      inside a session opened with a read-write token.

    Already verified before the deploy:
    - the obs image builds on Python 3.12.14 and passes the stub end-to-end run;
    - a real event through the new DSN reached `player-obs` (deleted after);
    - the obs browser UA passes Umami's bot filter (probe deleted after).

    **Probes after the deploy** (prod agent, inside the containers; nothing user-visible):

    ```bash
    # 1. VPS -> GlitchTip through the new DSN (the homelab session deletes the issue afterwards)
    docker exec player-obs-1 python -c "import sentry_sdk; from podcast_obs import telemetry as t; print('dsn', t.init_sentry()); sentry_sdk.capture_message('player-obs post-deploy probe (delete me)', level='error'); sentry_sdk.flush(10)"
    # 2. VPS -> Umami: one obs_tool event (the homelab session deletes it afterwards)
    docker exec player-obs-1 python -c "import time; from podcast_obs import telemetry as t; t._emit_umami('post_deploy_probe', True, None); time.sleep(6); print('sent')"
    # 3. The edge: unauthenticated MCP is refused, /metrics is not public
    curl -s -o /dev/null -w "mcp %{http_code}\n" -X POST https://obs.closelistening.app/mcp      # 401
    curl -s -o /dev/null -w "metrics %{http_code}\n" https://obs.closelistening.app/metrics      # 404
    ```

    Then the operator makes **one obs call and two content-MCP calls in one conversation** from
    claude.ai (e.g. "prod health" on the Obs connector, then two lookups on Close Listening).
    Tell the homelab session the time. It runs its check (`obs_mcp_verify.py <since>`, homelab
    scratchpad), which failed every check on the pre-deploy baseline, as expected:
    - `up{job="player_obs"}=1`, 58 zero-registered `mcp_tool_calls_total` series, `process_*`;
    - access lines with `trace_id`, no healthcheck or uvicorn access lines;
    - the tool-call line, its `player-obs` span with `mcp.participant`, and an Umami
      `obs_tool:<tool>` event under a participant;
    - each player-mcp call with its own trace id (the baseline is 3 calls, 1 trace id).

    **After the deploy, in order (10a, 10b and 10c together):**

    1. **Images** (`deploy-all-prod`).
    2. **`deploy-config surface=player`** once `player-obs-1` is healthy: a dry run, then real.
       Images go first, so the new `player_obs` scrape never reads `up=0` during a gap.
    3. **The three 10c probes** above (prod agent). Send the homelab session the time; it deletes
       the GlitchTip probe issue and the Umami probe event.
    4. **The operator's calls**, in one claude.ai conversation: one Obs call, then two Close
       Listening calls. They are also the "one MCP tool call" that 10a and 10b need.
    5. **`grant-mcp-all` dry run** (10b). Send the homelab session the counts.
    6. **`grant-mcp-all --apply`**, only on the operator's go.
    7. **The homelab session's checks:** `obs_mcp_verify.py <since>` (10c) and
       `podcast_mcp_verify.py <since>` (10a, 10b). It reports each check pass or fail, and
       deletes its probes.
11. **Add the 13 smoked beta feeds to the spec** (operator, 2026-10-16), after checks 1-10 pass:
    `spec_add.py --apply` inside `compose-api-1` (same validation as `PUT /api/feeds`; it writes
    `feeds.spec.yaml.bak-<UTC>-pre-13beta` first). Dry run 2026-10-16: 67 -> 80 feeds, only the 13
    appended. Tonight's 00:00 nightly deepens them (`max_episodes: 10`, about 90 audio-h, 24-30 h).
    Not verified: the newest 10 of Explain It to Me / The Curiosity Shop are all from the current
    show.
12. **Audio sync, #2359 / RFC-129 (from the mobile session, 2026-10-16, simplified plan): the
    candidate must be `75304e525` or later** (reference format version 2, long windows: the app
    downloads ~6 MB instead of ~23 MB an hour after changed breaks). A backfill run on an older
    image writes version 1 references; then rerun it with `--force`.
    The only step is a one-off reference backfill, on the operator's go; no DGX, no LLM; no daily
    job (drift-check and build-maps are manual diagnosis only). New episodes get their reference at
    the end of ingest. Runbook `docs/recipes/audio-timing-drift.md`. On the prod host, NEW image:

    ```bash
    cd /srv/podcast-scraper
    C="docker compose --env-file /srv/podcast-scraper/.env -f compose/docker-compose.stack.yml -f compose/docker-compose.prod.yml -f compose/docker-compose.vps-prod.yml"
    # 1. dry run, 20 episodes
    $C run --rm --no-deps pipeline-llm python -m podcast_scraper.audio_sync build-references --corpus /app/output --rclone-remote audioarchive --limit 20
    # 2. backfill, detached, resumable (~2,600 originals; one temp folder per episode, deleted after)
    $C run -d --no-deps --name audio-refs pipeline-llm python -m podcast_scraper.audio_sync build-references --corpus /app/output --rclone-remote audioarchive --write
    docker logs -f audio-refs
    ```

    - **Check:** `/api/app/episodes/<slug>/audio-reference` answers 200 with checkpoints, and the
      count of `*.audioref.json` under `/srv/podcast-scraper/corpus` grows.
    - BBC: drift and maps measure `https_audio_url(url) or url` since `c838c6a9c`, the URL the player
      gets. `--base-path ""` matches prod's archive. Prod disk: 93 GB free; the references take
      about 2.6 MB per hour of audio, about 7 GB for the backfill.

**Then the long runs, one at a time (concurrency group prod-corpus), by `postdeploy_chain.sh`:**
the #2370 batch, then #2332 batches sized at dispatch to end by 23:30 UTC; the #2332 episodes
left over run after the nightly. Dry run 2026-10-16: #2370 (11 eps), then #2332 batch 1 of 17 eps /
16 audio-h.

The id lists are on the mini in `/Users/Shared/postdeploy-2026-10-16/`. Every run below is
`reprocess-prod.yml` with `selection=episode_ids_worklist`, `episode_ids=<the file's ids>`,
`pipeline_stage=full`, `use_transcript_cache=false`, `profile=config/profiles/prod_dgx_full.yaml`,
`timeout_minutes=330`. Wall times use the 3-4x real time measured on backfill batches 27-31; each
run must also end before the 00:00 UTC nightly.

1. **#2370 re-transcribe, `r2370_ids.txt`: 11 episodes, 817 audio-min, about 3.5-4.5 h.** Needs a
   deployed sha that contains `e7234aaec`; a relabel keeps the bad timestamps.
   - The 10 found by the ingestion session (`/Users/Shared/doubled_stream_10_episodes.tsv`),
     including The Terror Part 4 (`7d4f6bf6-bcb2-11f1-9e30-afa90f070c5b`): take it off the
     backfill hold list.
   - Plus "Lizzi Lee on Involution" (Sinica), found by the overlap scan: 527 overlapping segments,
     worst 23.6 s. Backfill batch 27 re-transcribed it on 2026-10-15 with the unfixed code.
   - **Sizing (2026-10-16, `scripts/audit/asr_segment_overlaps.py` over 2,504 served episodes):**
     2,023 have no overlap; 646 have 1-9, worst under 5 s in all but 4 (ordinary jitter). The
     drift shows as hundreds: 8 episodes with 175-2,330 overlaps, 7 of them in the 10, plus Lizzi
     Lee. Two of the 10 had only 3 and 5 overlaps (found by listening), so the scan cannot clear
     an episode. The 122 backfilled episodes were all checked: only Lizzi Lee is hit beyond the 10.
   - **Not checked:** the 4 episodes with a few overlaps of 5-25 s (one each in feedburner
     `2aa371eb`, megaphone `3581c092`, megaphone `9316148d`, soundcloud `c56bfb45`).
   - **Check after:** run the scan in the api container
     (`python scripts/audit/asr_segment_overlaps.py --corpus-dir /app/output --min-overlaps 50`):
     none of the 11 may appear. The fix's own measurement went from 470 overlaps to 1.
2. Keep `/srv/podcast-scraper/punct-2368/` until the ingestion session says the #2368 analysis is
   done.
3. **#2332, 94 of the 96 episodes with an unlabelled publisher transcript.** A full reprocess,
   since there are no voices to relabel.
   - **Sample before the deploy** (operator, 2026-10-16), `s5_sample_ids.txt`: 5 typical-length
     episodes, 134 audio-min. 2 Explaining Brazil (12.6 min each; the feed's median, and its
     episodes are "(preview)" cuts), 1 The Flip, 2 In Moscow's Shadows (48 min each). It answers
     the #2332 question: does a re-transcribe give these shows named voices? It runs on
     `sha-f35a446`, without the #2370 fix, so scan these 5 for overlaps too and add any hit to a
     post-deploy run.
   - **After the deploy**, `s5_batch1..4_ids.txt`:

     | batch | episodes | audio-h | wall |
     | --- | --- | --- | --- |
     | 1 | 15 | 15.0 | 3.7-5.0 h |
     | 2 | 19 | 15.0 | 3.7-5.0 h |
     | 3 | 31 | 15.0 | 3.7-5.0 h |
     | 4 | 24 | 3.7 | 0.9-1.2 h |

   - All 96 were found served on prod 2026-10-16, all `transcript_source=direct_download`, and every
     audio URL answered (`206`).
   - **AI and Design (58 min) and The Every Podcast (33 min) are back in** (operator, 2026-10-16):
     their feeds are among the 13 added in item 11, and the reprocess only runs feeds in the spec,
     so the spec add must come before the chain starts. The chain's pool is now 91 episodes,
     50.2 audio-h (`s5_pool.json`).
4. **One relabel, `relabel_combined_ids.txt`: 131 episodes, `pipeline_stage=relabel_only`** (not a
   re-transcribe), on a sha that includes `2b580bb1e`. The operator's go is needed for the run.
   - The 91 from 8.4 (`relabel_84_ids.txt`: replay of `c6cdc5d` vs `aaadbb2` modules, 2026-10-15;
     none of them was re-done by backfill 27-32).
   - The ingestion session's 44 (`/Users/Shared/relabel_44_episodes.tsv`, with what changes per
     episode): one replay of the deployed roster (`f35a446b8`) against main (`2b580bb1e`) over the
     2026-10-15 snapshot. 41 voices in 40 episodes: 27 respellings to the title's spelling (#2363),
     8 false guests removed (#2364), 6 from the question-lead host seat (#2366); plus 4 stale
     records (#2365, Kara Swisher's Pivot tour promo stored as a guest).
   - 4 episodes are in both lists; 1 ("Taking down the cultural elite") is dropped because the
     #2370 re-transcribe relabels it anyway. 91 + 44 - 4 - 1 = 130, plus Hard Fork below = 131.
   - **Hard Fork "A.I. Backlash Turns Violent + Kara Swisher on Healthmaxxing"**
     (`25d702f6-ed6a-4565-8065-e729eb0f2650`), added back on the ingestion session's correction: the
     stored "Kara Swisher" is SPEAKER_01 (46%), which is both hosts merged, so the relabel removes a
     wrong name. Not fixable by a relabel: SPEAKER_02 holds both Casey Newton and Kara Swisher
     (diarization merged them); only a rediarize could, and that is the operator's call.
   - **Time:** about 17 min per episode by the ingestion session's notes (GI dominates), so about
     37 h for 131. No relabel has been timed on prod: run 5 first and measure, then batches that fit
     the runner and the nightly, after the #2370 and #2332 runs.
   - **Check after:** send the run ids to the ingestion session; it compares the replay's changes
     with the served records on the next snapshot.
5. **8.5:** recompute after the relabel (Rest Is History episodes whose host seat is still
   unnamed).
6. **Migration `0030_adfree_cut_resynced` — LAST** (ingestion session, 2026-10-16; the deploy
   must carry `ded80e06f`, the floor for everything above, so new ingests and the migration cut
   ads the same way). It brings served episodes to today's ad cut (`4515bce1c`, #2360): ~421
   re-cut + ~304 ad-map rewrites on the 2026-10-15 snapshot, moves every GI quote, drops quotes in
   ad text, removes ~54 ad-built insights and ~174 sponsor entity nodes, syncs bridge / counts /
   context. Text files only, with backups and undo; no DGX, no LLM. Run it after the #2370 run,
   the #2332 batches, the 131 relabel and the Turkish reprocess (a later run would overwrite its
   output), with the pipeline idle (the runner takes the corpus lock). No deploy path runs
   migrations by itself (checked: the deploy workflows and the API).
   1. `df -h /` first: every `upgrade run` takes a whole-corpus snapshot (13 GB on 2026-10-03).
   2. Dry run (`inspect-prod-corpus.yml checks=upgrade_dry_run`, or in `compose-api-1`
      `python -m podcast_scraper.cli upgrade run --corpus-dir /app/output --dry-run`): only `0030`
      pending (0029 too → stop, tell the ingestion session); `refused` must be 0 (else send the
      list, do not apply). The other counts will have moved from the snapshot numbers.
   3. Apply on the operator's go: `upgrade run … --yes` (or `make upgrade-corpus`).
   4. `upgrade verify` prints "every served ad-free base is today's cut"; no root-owned files
      (`find /app/output -user root -newer adfree_cut_resynced.jsonl | head`).
   5. Search: the next pipeline run's incremental upsert re-embeds the changed episodes, or run
      `index-two-tier` WITHOUT drop/clear. Never rebuild. Then
      `python -m podcast_scraper.cli insight-clusters --output-dir /app/output`.
   6. Spot checks: Invest Like the Best "Krishna Rao - Anthropic…" has no Felix by Rogo insight;
      Lenny's "Head of Claude Code…" `.adfree.txt` keeps the guest intro, not the WorkOS read; The
      Rest Is Science "The Evolution of Colour Vision" keeps its cold open; for any quote
      `adfree_text[char_start:char_end] == quote.text`; `make corpus-completeness-check`.
   - Undo: `from podcast_scraper.upgrade.migrations.m0030_adfree_cut_resynced import undo;
     undo(Path("/app/output"))`. Receipts `adfree_cut_resynced.jsonl`, backups
     `.podcast_scraper/upgrade-backups/0030/`. Accepted loss (operator): 2 real insights and 3 real
     entity nodes that share an ASR segment with ad text (`docs/wip/AD-CUT-SEGMENT-STRADDLE.md`).

## Deployed as `sha-f35a446` (candidate `0bc13f92c`, 2026-10-15) — steps, kept for reference

**What it carries since `aaadbb2`:**

- the search fix `5426067f8` (a storyline's exact or old name leads its category);
- the work-list "declined" report `d0f4951ce`;
- Langfuse trace = OTel trace, with stage-named generations (`4ae796413`);
- the o11y MCP fixes (`2399f5c14`, `8c74a140d`);
- GlitchTip grouping and release tags (`0bc13f92c`);
- the source audio length (#2359, `bd37c9189`, new episodes only);
- the reload-not-restart edge scripts (`60dea6847`);
- the e2e fixture re-timing.

No vhost or Alloy file changed, so no `deploy-config` this time.

**Before:**

- No reprocess runs and no smoke during the deploy (a deploy during a reprocess is untested).
- Main's CI green on the candidate (Python, Docker, Snyk, CodeQL).

**Deploy:** `deploy-all-prod.yml` with `image_sha`.

**After, checks in order:**

1. **Containers and smokes.**
   - All containers healthy; player and operator smokes green.
   - **First deploy that RELOADS Caddy:** `deploy-player.sh` and `deploy-operator.sh` must log the reload, not a restart. The smokes running right after are the test: no 521.
2. **GlitchTip (`0bc13f92c`).**
   - operator-api, player-api and player-mcp log `sentry init complete … release=sha-<short>` (was `podcast-scraper@2.7.0.dev0`).
   - Player web reports release `sha-<7>`.
   - Over the next days, one GlitchTip issue per error (podcast-pipeline had 424 issues for 72 errors).
3. **Search (`5426067f8`).** Re-run the 33-name storyline check as the smoke listener (`per_category=3`): expect every name, current and old, to lead the storylines category, including "model distillation" and the old name "embodied cognition".
4. **o11y MCP** (homelab's checks, inside player-obs-1):
   - `python -m podcast_obs deploys`: the newest entries are today's deploys, each with a `workflow` field.
   - `python -m podcast_obs surface pipeline --window 1h`: its errors list differs from `--window 14d`.
5. **Langfuse (`4ae796413`):** podcast-prod traces keyed by OTel ids with stage-named generations, and `board-summarizer-*` sessions. Homelab confirms.
6. **Work-list (`d0f4951ce`):** the next reprocess with refusals logs "left as they were by design (…)" at WARNING, not ERROR.
7. **#2359:** the first newly processed episode has `source_audio_duration_s` in its `.asr.json` and in the manifest's asr metrics. A reprocessed episode counts, since the backfill re-transcribes.
8. **Then the long runs, one at a time:**
   - backfill batch 27+ (15 episodes, about 5 h);
   - the 8.4 relabel: 5 episodes to measure, then the rest in batches;
   - 8.5, recomputed after 8.4.

**Separately: `deploy-litellm`, any time the gateway is idle, on the operator's go.**

- What it changes: `LITELLM_SENTRY_DSN` now points at GlitchTip project litellm-prod (27) over the tailnet HTTPS node (the old one was a deleted project at an address the VPS can't reach, so gateway errors went nowhere), and `SENTRY_ENVIRONMENT=prod`.
- The image stays pinned at v1.95.0 (`e20cd3e59`).
- Check after:
  - `docker exec litellm printenv SENTRY_ENVIRONMENT` → `prod`;
  - the deploy smoke (aliases served, completions end to end);
  - `_time:10m AND app:litellm AND -service:*` → 0;
  - a real gateway error lands in litellm-prod.

**DGX: the homelab session owns it.** pyannote and faster-whisper read `SENTRY_ENVIRONMENT` from env files under `/opt/*/` on the DGX, which still say `production`; moss's is in `infra/dgx/converge/deploy.py`. Homelab changes them during a DGX converge window.

## What the deploy carries

App (server + player), checked in "After the app deploy" below:

- Discover (ADR-163/164/165, PRD-048, RFC-126): quote cards with a "why", rails paged on the
  server, cold start (Everyone's until topics are picked), "Less like this", Hear another view,
  People you keep hearing, Continue the thread, More from what you captured; Umami rail events.
- Explore: its own tab (telescope icon), search at the top, Trends paged, the catalog behind
  "See all".
- Short listens (15 minutes on a topic/theme/storyline), Learn paths (a day at a time, "Next day
  now"), Home "In progress", the stats story on Profile and Home.
- Brief: word search over the whole episode, folding results, bookmark a result.
- Search: subjects once at the top, matched by name; recent searches synced per account.
- Cards and player: tap opens paused, download in the ⋯ menu, the server-unreachable banner fix,
  settings synced with the server.
- Boards: the Brief sheet and boards in Search (#2340, #2341), Library tab "Collections".
- Board summaries (RFC-127): the `board-summarizer` sidecar on the player stack.
- #2342: every DGX LLM call through the LiteLLM gateway (a transparent proxy), so it is metered
  and in Langfuse. Summary/GI/KG/quotes/naming on `:8003` and translation on `:8005`.
- Langfuse projects: prod gateway default → `podcast-prod` (was `agents`); dev → `podcast-dev`.

## Order: the gateway first

The DGX profiles now point `vllm_api_base` at `LITELLM_API_BASE`. The vLLM provider's served-model
check fails closed, so the gateway must serve the alias **before** the app is deployed.

### 1. Orrery's traces stay in `agents` (before the gateway deploy)

The gateway's default Langfuse project becomes `podcast-prod` at its next deploy (the GH secrets
`LITELLM_LANGFUSE_PUBLIC_KEY/SECRET_KEY` were pointed at `podcast-prod` on 2026-10-13). Orrery
shares this gateway, so give its key its own logging first, to `agents` where its traces go today.
Per-key logging is not Enterprise-gated in LiteLLM 1.96 (verified on the mini with
`proj-podcast-dev`: its calls land in `podcast-dev`, keyless pass-throughs in the default).

On the VPS, with the `agents` key pair (on the mini: `~/signal-fleet/fleet-gateway.env`) and the
master key from the `litellm` container env, `POST /key/update` on `proj-orrery-lab` with
`metadata.logging = [{callback_name: langfuse, callback_type: success_and_failure, callback_vars:
{langfuse_public_key, langfuse_secret_key, langfuse_host: http://homelab:4000}}]`. Read the key's
token from `LiteLLM_VerificationToken` by `key_alias`; never print it. Keep the key's existing
metadata (merge, do not replace).

### 2. Deploy the prod gateway (`deploy-litellm.yml`)

Carries `infra/litellm/config.yaml`: alias `NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4` (hosted_vllm at
`dgx-llm-1:8003`, $0, no fallback, 900 s), pass-throughs `/tokenize` (`:8003`) and
`/v1/translate/v1/completions`, `/v1/translate/v1/models`, `/v1/translate/tokenize` (`:8005`).
The smoke only checks the alias is served (it is not `podcast-*`), so the deploy does not fail
while the GPU is in another mode.

Check, with the pipeline's key (`proj-podcast-prod`), from the VPS:

- `GET /v1/models` lists the alias; one 5-token completion through it.
- `POST /tokenize` returns the same token ids as `dgx-llm-1:8003/tokenize`.
- `GET /v1/translate/v1/models` lists `google/translategemma-12b-it`; `POST /v1/translate/tokenize`.
- Langfuse: the calls are in `podcast-prod`; one Orrery call is still in `agents`.

### 3. Deploy the app (`deploy-prod.yml`)

`LITELLM_API_BASE` now passes through to the `pipeline` and `pipeline-llm` services. Check on the
first pipeline run (or a one-episode reprocess):

- `llm_cost` events: `provider: vllm`, `model`/`served_model` the real id, as before.
- No `served-model` errors; no `/tokenize unavailable` warnings.
- Langfuse `podcast-prod` holds the run's calls: summary, GI, KG, quotes, entailment, naming,
  translation (pass-through). Grafana "LLM calls, tokens, cost" panels unchanged.

### 4. Deploy the player (`deploy-player.yml`)

Stages two new files in `/dev/shm/player-secrets/`: `podcast_sentry_dsn_pipeline`
(`PROD_SENTRY_DSN_PIPELINE`) and `litellm_api_key` (`LITELLM_PROJ_PODCAST_PROD_KEY`); writes
`LITELLM_API_BASE` into `.env.player`. Starts `board-summarizer`. Check:

- `docker logs player-board-summarizer-1`: `start:`, then a `board_summary.tick` every 5 minutes.
- A test board (two episodes, then quiet 10 minutes): `board_summary.written`, the Brief shows the
  summary; `llm_cost` with `stage: board_summary`, $0; in Langfuse `podcast-prod`.
- A tick with nothing changed makes no request (no `GET /v1/models`, no completion).
- Kill switch: `{"enabled": false}` in `player-appdata/board_summaries_settings.json` →
  `stopped: switched_off` on the next tick.
- GlitchTip `podcast-pipeline`: nothing new from the sidecar.

### 5. After the app deploy: the player (smoke account, browser and the installed native app)

- Discover: quote cards load with their "why"; a new account sees Everyone's, labelled, until it
  picks topics; "Less like this" hides the reason and Profile undoes it; rails page.
- Explore tab: search first; Trends "Show more"; "See all" opens the full lists with filters.
- A short listen starts at once, says its real length, never plays an ad or repeats a line.
- Learn: today's step; earlier days closed and replayable; "Next day now".
- Home: "In progress"; the week line. Profile: the stats story, no zero counts.
- Brief: search a word only in the transcript body; results fold; the bookmark saves.
- Search: a person/topic name lists them once at the top; recent searches follow the account
  to another device.
- Boards: a board's Brief opens from its button; boards appear in Search; "Collections" tab.
- Umami: the Discover rail events arrive (operator-dev / operator-prod site as appropriate).
- **The installed native 1.0.3 app still works** against the new server (pages, saves, Library,
  Discover). See `TODO-remove-pre-1.0.3-compat.md` §"what 1.0.3 itself still needs".

### 6. The first nightly after the deploy

As usual: run green, `llm_cost` per stage, now also in Langfuse `podcast-prod`; GlitchTip quiet.

### 7. Stable references and storyline summaries (#2345, #2344) — each step needs the operator's go

From `feat/2344-2345-refs-and-storyline-summaries`. The sidecar runs both on its own every 5
minutes (the reference pass first, no model; then boards; then storylines, capped at 20 a sweep).
These are the one-time fills, so nothing waits on the cap and the history from before ids were
kept is used.

1. **After the first nightly on the new code**, both artifacts carry `id_history`:
   `topic_theme_clusters.json` (`data.id_history`) and `search/topic_clusters.json`. Check both.
2. **Re-key existing user state**, with the renames from before ids were kept read from the
   2026-10-10 snapshots' artifacts. Dry run, read the counts, then apply (receipts in
   `<appdata>/rekey_user_references.jsonl`; `--undo` puts back what the apply wrote):

   ```sh
   # the two snapshot artifacts into the sidecar (it mounts the corpus read-only)
   docker cp <snapshot>/enrichments/topic_theme_clusters.json player-board-summarizer-1:/tmp/prior-storylines.json
   docker cp <snapshot>/search/topic_clusters.json player-board-summarizer-1:/tmp/prior-themes.json
   docker exec player-board-summarizer-1 python -m podcast_scraper.server.rekey_references \
     --appdata /app/appdata --corpus /app/output \
     --prior-storylines /tmp/prior-storylines.json --prior-themes /tmp/prior-themes.json   # dry run
   ...                                                                                   # then --apply
   ```

   Expect: renames re-keyed silently, the rest flagged "No longer available" in the player,
   nothing deleted.
3. **Fill every storyline's summary** (99 on prod), dry run first:

   ```sh
   docker exec player-board-summarizer-1 python /app/board_summarizer.py --storylines-backfill
   docker exec player-board-summarizer-1 python /app/board_summarizer.py --storylines-backfill --apply
   ```

   Needs the DGX in research mode (the `:8003` slot), like the board summaries; $0 budget holds.
4. **Durable storylines (RFC-128, #2349): seed the registry from history.** Any time after the
   deploy. Replays the registry month by month over the corpus (no model, seconds) and merges it
   with today's storylines. It never writes the storyline artifact: `--apply` writes
   `enrichments/storyline_registry.seed.json`, and the next enricher run takes it in, so it cannot
   race the nightly. Applying the same seed twice is a no-op. Dry run, read the counts, then apply:

   ```sh
   docker exec compose-api-1 python -m podcast_scraper.search.storyline_history --corpus /app/output
   docker exec compose-api-1 python -m podcast_scraper.search.storyline_history --corpus /app/output --apply
   ```

   Expect, as on the 2026-10-14 copy: `current` 99, `replayed` 59, `with_history` 51, `folded` 59,
   `added_from_history` [], `candidates` 47. After the next nightly, the log line
   `storylines: registry health=ok source=seed storylines=99 quiet=17 ... shared_topics=0
   vanished=0`; then `source=artifact` on every later run. 99 storylines with the same ids, names
   and anchors (pages and follows do not move); 6 trade a few topics with each other.

   Undo: `... storyline_history --corpus /app/output --undo` withdraws a seed not yet taken; once
   taken, it writes a seed that puts back the registry the seed replaced (none, if the seed was
   taken by the first run on the new code — then there is nothing to go back to).
   The seed can also be previewed, applied and undone from the viewer dashboard (Intelligence →
   **Storylines & themes**), which calls the same code.
5. **Reload Alloy on prod** with `infra/observability/operator.alloy`: its new last stage ships an
   emitted event's JSON alone, so VictoriaLogs unpacks `llm_cost` (and every pipeline event) into
   fields — before, they arrived behind the log prefix and `event_type:` matched nothing. The
   config passes `alloy fmt` (checked in the prod container, 2026-10-14); whether the stage
   behaves in the running pipeline is proven only by this reload. Check after: `_time:1h
   instance:prod-podcast AND event_type:llm_cost | stats count()` in VictoriaLogs is above 0 once
   a pipeline run has called a model.
6. **Sync Grafana**: the alert rule `enrich-common-storyline-registry-bad` (fires on any
   `storylines: registry health=bad` line), the two new panels on `vps/podcast-app.json`
   ("Storylines per run", "Themes per build"), its fixed "LLM cost (24h, pipeline)" panel (the
   query asked for a `job` label pipeline lines never carry), and the LLM cost dashboard
   (`common/grafana-dashboard-llm-cost.json`, rewritten from Loki LogQL to VictoriaLogs; its
   queries read events from before and after the Alloy reload), and the other seven `common/`
   dashboards (pipeline execution, pipeline stage, deploy observability, ingestion ×2, logs, DGX),
   which were Grafana Cloud exports and rendered nothing here — each query checked against
   production data (2026-10-14). Dry run, then apply:

   ```sh
   make obs-sync              # lists what would change
   make obs-sync APPLY=1
   ```

   **Done 2026-10-15 by the homelab side** (prod needs no token for this). All of the above was
   already live in Grafana; podcast-app was re-pushed for the metrics→traces links. The dry run now
   diffs the live Grafana (NEW / CHANGED / MOVE / UNCHANGED) when `GRAFANA_<ACCOUNT>_URL`/`TOKEN`
   are set. Alert rules (including the storyline-registry rule) are owned by agentic-ai-homelab's
   Grafana alerting `rules.yaml` and are no longer synced from here. The two storyline/theme panels
   stay empty until the first storylines run / theme build after this deploy.

7. **Check in the player:** a storyline page and a storyline's Learn path start with the summary;
   a storyline the seed gave an old name (10 on the copy) says "Also called …" under its title,
   and searching its current or old name lists it under the search's storylines (the card at the
   top is the topic of that name: storylines are named after a member topic); a quiet one says
   "Quiet since …";
   `GET /api/app/discover/references/gone` answers for the smoke account. In the viewer
   dashboard the Storylines & themes card shows the counts, the chart back to 2024 (from the seed)
   and the last run's changes; after the next theme rebuild, its "Kept their id" is close to the
   theme count (themes now keep their id by meaning too, and their names stick).

### 8. Naming, punctuation and the MCP (2026-10-13/14 changes) — each prod step needs the operator's go

The deploy carries: Phase C naming (#2347), K8 (`ee4d19961`), F7 storyteller seats (`2fd6411dc`),
the punctuation filler rule (`81e3e8610`), the MCP entity-resolver refresh (`e6cc51be2`), the
new `repunctuate_only` reprocess stage, and the 2026-10-14 naming fixes (`19da42c29`: a new promo
for another show is an ad voice, "I'm <role> <stated name>" names a stand-in presenter, the BBC
opener hail counts). The code only changes NEW episodes; served ones keep their
old names and text until steps 3 and 4.

1. **MCP:** after the restart the log shows the entity registry warming at startup; then one
   `resolve_entity` call (e.g. "reward hacking") answers in seconds, not a 2-3 minute timeout.
2. **First nightly:** read the new episodes' rosters (no promo voice named as a guest, hosts named);
   compare the naming-coverage gauge with the day before.
3. **`repunctuate_only`** on the episodes still flagged unpunctuated (32 measured on 2026-10-14;
   25 of their 48 windows are expected to pass, `scripts/measure/punctuation_refusals.py`).
   REBUILD the list right before the run: Ottoman History's "Hebrew Orientalism" was reprocessed
   after the measurement and belongs on it. The 1-episode trial is DONE (2026-10-14, In Our Time
   "Archive: The Measurement of Time", run from a source tree before the deploy): only the flagged
   window changed (the other 983 segments byte-identical), 0 -> 114 sentence ends, same 4
   speakers, words 1,769 -> 1,727 (mostly "clock making" -> "clockmaking"; it is a re-decode, so
   wording can shift slightly: "was" -> "is"), summary/GI regenerated, $0. So: the rest directly. Through `reprocess-prod.yml` with
   `pipeline_stage=repunctuate_only` and the episode ids (newline- or space-separated); not inside
   a backfill batch. An episode with nothing left to repair is recorded and not touched.
4. **Relabel the served episodes whose names change** (Phase C, K8, F7: the replays list them):
   `relabel_only` batch, the LLM on the DGX — operator decides.
5. **F7 A2:** re-diarize The Rest Is History (`rediarize_only`, approved 2026-10-14, AFTER this
   deploy so the new naming code names them) to capture voice embeddings for the 15 storyteller
   seats nothing in the text names.

6. **Gray Area "How trauma became a status symbol":** never reprocessed — it was in backfill
   batch 2 on 2026-10-10, before the feed was back in the spec, so the chain counts it done. Put
   it in a backfill batch by hand.
7. **Live spec:** `account.live.spec.ts` now expects the Library tab "Collections" (renamed
   2026-10-13, #2327). Prod before this deploy still says "Boards": a manual `smoke-player` run
   BEFORE the deploy fails on that one line; after the deploy it must pass.

## After the deploy: open work (2026-10-14 handover)

- **7.7 checked 2026-10-15** (API as the smoke listener):
  - all 21 storylines on the rail have a summary;
  - 10 carry old names (12 in all) and 1 is quiet ("parental responsibility", since 2026-07-03);
  - `/discover/references/gone` answers 200 (0 items).

  Search by a storyline's name, as the player calls it (`per_category=3`), 33 names (21 current,
  12 old): the storyline led its category for 29, was lower for 2, missing for 2 ("model
  distillation"; old name "embodied cognition" of "ai in education"). The card at the top is
  always the TOPIC of that name: a storyline is named after one of its member topics, and the
  resolver checks topics first; the topic card links to its storyline. Fix: an exact current or
  old name now puts the storyline first in its category (`_storylines_named`, app_search.py) —
  reaches prod with the next app deploy. (An earlier note here said search never shows the
  storyline; that was tested on the plain search mode, not the player's, and was wrong.)
  Not checked: the browser view, the Learn path's first screen, and the viewer dashboard card.
- **8.3 list rebuilt 2026-10-15:**
  - Prod corpus, newest record per episode guid: 2,497 episodes, 37 with an unpunctuated window
    (81 windows).
  - 24 of the 37 had a repair refused → the `repunctuate_only` list.
  - 9 never had a repair attempt and are in the backfill pool; 2 are in batch 26.
  - 2 are neither: Gray Area "How trauma became a status symbol" (step 8.6) and BizNews "BN
    Daybreak: Swimming SA crisis …".

  The 439 unmatched asr records were the scan's own pattern (it missed the `_285e51f2` run-dir
  suffix); the re-scan finds nothing missed (see Status above).

- **From the 8.4 replay (read 2026-10-15), naming quality:**
  - "Matt Iglesias" for Matthew Yglesias (Capitalisn't "Why Matthew Yglesias Is Skeptical…"): the
    name in the middle of a title is not extracted as a stated name (titles like "…, with X" or
    "X | Y" are), so the host's spoken "Matt Iglesias" is published as heard. With the stated
    name, the canonicalizer would snap it (Matt↔Matthew nickname, surname 1 edit). Fix = title name
    extraction plus a corpus replay. Parked.
  - Capitalisn't "Japan, with Takeo Hoshi": served "Luigi Zingales" sits on the voice that opens
    with a cold-open quote; the new code drops the name (unidentified). Unresolved from the text.
  - a16z "The Infrastructure Behind the Machine Age": not a served loss (SPEAKER_03 is unnamed on
    prod already); but the served roster is wrong elsewhere ("Ben Horowitz" is the voice that says
    "Ben, Martin, Ragu, welcome", i.e. the interviewer).
- **Host seat typed commercial** (mobile session, fixture timeline change, local branch): agreed
  the guard belongs in `classify_voices` (shared cleaning, ADR-137), by talk share, since the
  cleaning runs before naming; prod has no commercial voice above 6.5% of its episode.
- **Batch 26 "Dr. Sharon Blackie"** (The Living Philosophy): 1 of 2 voices named, no host resolved
  (`empty_host_anchor`), 29% of the talk unattributed.

- **Naming leftovers (#2304), to work on after the deploy:**
  - **Phantom Pain** and **Welcoming Misha Glenny** (Curious Cases / In Our Time): the host voice
    carries the merged BBC ad tag and its first turns name nobody ("That is a phantom pain.",
    "why do you think you're the right person for this role?"), so the opener rule cannot seat
    it. Needs a new signal (who asks the questions?).
  - **StarTalk's Patreon supporters** ("I'm Joel Cherico, and I support StarTalk on Patreon")
    stay named as guests: the episode descriptions list them, so the promo rule treats them as
    stated people. Needs a "supporter shout-out" rule.
  - **Kara Swisher's Gray Area promo** is not caught by the promo rule (no self-introduction +
    promo cue pair in that voice). Read the voice, find what marks it.
- **The 19 episodes a full backfill reprocess left unpunctuated:** 17 are on the
  `repunctuate_only` list (step 8.3) — 11 of their 31 windows pass with main's checks, 20 still
  refuse (mostly `lost_words`, retained 73-90%). 9 of the 15 `lost_words` windows are the
  episode's last, short window: hypothesis to test — the clip decode drops the end of the audio.
- **S5 (#2332) answered:** `relabel_only` cannot name the 95 episodes without diagnostics; they
  are publisher transcripts with no diarization ("segments carry no speaker identity"; trial run
  38557754426, $0, nothing changed). The route is `rediarize_only` (audio + DGX) — operator decides.
- **DR drill red (10-07, 10-14):** restore works; `stack-playwright`'s Firefox graph test times
  out — the data loads in ~16 s, then >45 s of browser graph merge/layout for 1.9k nodes (it
  passed at 56.7 s of 60 on 09-30). Operator chose (2026-10-14) to RAISE the timeout now: the
  canvas wait is 150 s (test budget 240 s) in `tests/stack-test/stack-viewer.spec.ts`. Verify on
  the next drill, 2026-10-21. The slow graph load itself is still open (profile the viewer).
- **GI observation:** regenerated GI repeats some quote texts as separate Quote nodes (2 -> 7 on
  the trial episode); the old GI did it too. Not investigated.
- **Cleanup, each path needs the operator's OK:** prod `compose-api-1:/tmp/nc` (gold sets +
  replay kit, real data — keep until the post-deploy naming checks and F7 A2 are read), host
  `/tmp/punct` (measurement scripts and outputs, real text), `/app/output/repunct_trial_ids.txt`,
  the `python:3.12-alpine` image; local scratchpad `gtrees`, `gv`, `k8trees`, `f7trees`.

## Carried over (operator items, not deploy steps; from the 2026-10-12/13 plan)

- **A10 DST cron:** DONE 2026-10-13 (`cron: 0 0 * * *`, scheduler reloaded).
- **#2187 backfill decisions:** DONE 2026-10-13/14 — the comma ids go through the chain (inline ids
  split on whitespace, `60631f53d`); The Gray Area is back in `feeds.spec.yaml`; the "2 out of
  spec" did not exist; the refused repair windows → step 8.3.
- **Phase C / K8 / F7:** shipped in this deploy (step 8). S5 (#2332): trial DONE 2026-10-14, see
  "After the deploy". F7 A2 → step 8.5.
- **P5 entity search:** checked 2026-10-14 through the MCP; found the registry never refreshed and
  built inside the first request → fixed in `e6cc51be2` (step 8.1).
- **person_web (#2111):** DONE 2026-10-14 — the 304 misses written before the Wikidata resolver
  were removed (backup `/app/output/.backups/person_web_pre_wikidata_misses-20261013.tgz`); the
  14 Oct nightly re-looked up 129 people, 72 found.

## Dev (done 2026-10-13, for reference)

Homelab gateway: the same alias and routes (DGX by tailnet IP); `proj-podcast-dev` restricted to
the alias, its calls logged to `podcast-dev`. `.env.obs.dev`: GlitchTip `podcast-pipeline-dev`
(25) / `podcast-api-dev` (26), Langfuse `podcast-dev`. Keyless pass-throughs (`/tokenize`,
translation) are logged to the mini gateway's default project, `agents`.

## Not in this deploy

- The `VLLM_API_KEY` secret in the unused `baselines` environment (delete later, operator).
