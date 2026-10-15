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
- Step 8: run one at a time after backfill batch 26 (run 38689720595): 8.3 (24 episodes, list
  rebuilt 2026-10-15), 8.6, 8.4, 8.5; then the backfill chain resumes at batch 27. A batch of 5
  episodes took 18-19 minutes per episode.

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

  The scan could not match 439 asr records to their metadata file (different file stems), so it
  can miss episodes.

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
