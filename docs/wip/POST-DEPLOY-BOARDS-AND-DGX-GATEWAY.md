# Post-deploy plan — player 1.0.4: Discover, boards, board summaries, the DGX through LiteLLM

Written 2026-10-13 for the deploy of `feat/player-1.0.4` (109 commits ahead of `main` at
3dc44e395). The previous deploy (2026-10-12/13, `sha-c6cdc5d`) carried the first part of 1.0.4
(#2305: migration 0028, Moments = the Brief, search scopes); this one carries the rest. No
migrations, no new env beyond what is listed below. Every step that touches prod needs the
operator's OK, one at a time. Read the state again before starting.

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

## Carried over (operator items, not deploy steps; from the 2026-10-12/13 plan)

- **A10 calendar, 2026-10-25:** nightly cron `0 23 * * *` → `0 0 * * *` (DST).
- **#2187 backfill decisions (owed):** 8 episode ids containing a comma (`reprocess-prod.yml`
  splits inline ids on commas — route via an `episode_ids_file` on the box, a prod change); 2
  episodes out of `feeds.spec.yaml` scope; 3 The Gray Area episodes (feed not in the spec); the
  refused repair windows (16+ by 10-10).
- **Phase C (operator decides):** proposed order voice identity → K8 with its cross-post guard
  (#2338) → embeddings rebuild only if measured worth it; S5 (#2332, 96 publisher-transcript
  episodes) and F7 (The Rest Is History, 30) parked.
- **P5 entity search:** after an enrichment, search a new storyline/theme name (snapshots
  `~deploy/topic_*clusters.snapshot-20261010.json`).
- **person_web (#2111):** 281 misses (TTL 30 days from 2026-09-17) reconsidered from 2026-10-17;
  `refresh=true` does it now. Whether a refresh already ran: not checked.

## Dev (done 2026-10-13, for reference)

Homelab gateway: the same alias and routes (DGX by tailnet IP); `proj-podcast-dev` restricted to
the alias, its calls logged to `podcast-dev`. `.env.obs.dev`: GlitchTip `podcast-pipeline-dev`
(25) / `podcast-api-dev` (26), Langfuse `podcast-dev`. Keyless pass-throughs (`/tokenize`,
translation) are logged to the mini gateway's default project, `agents`.

## Not in this deploy

- The `VLLM_API_KEY` secret in the unused `baselines` environment (delete later, operator).
