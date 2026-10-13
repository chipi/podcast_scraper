# RFC-127: Board Summaries — the first LLM call outside the pipeline

- **Status**: Implemented in part (phase 1 built, not yet deployed)
- **Authors**: Marko, Claude (Opus 5.5)
- **Stakeholders**: Consumer App, Server API, Infra (player stack, DGX vLLM)
- **Related RFCs**:
  - `docs/rfc/RFC-119-holistic-collections.md` (boards; the 2026-10-12 amendment adds the Brief)
- **Related ADRs**:
  - `docs/adr/ADR-142-litellm-prod-gateway.md` (the gateway, for a profile that uses it)
- **Related UX specs**:
  - `docs/uxs/UXS-011-consumer-learning-app.md` (Collections; `BoardBrief`)

## Abstract

A board's Brief gets a **summary of the whole board**: one paragraph on what the board is about,
written by an LLM from the Briefs of the episodes on it. It is the first time the product calls an
LLM outside the pipeline, so this RFC fixes where the call runs, what starts it, what it reads, and
what stops a misused UI from reaching the bill. The call runs in a **background sidecar on the
player stack**, never in a request, with the **same profile and provider code as the pipeline**:
`prod_dgx_full`, the model on our own DGX, at a cost of zero. The budget is **zero**, and enforced, so
a change of profile to a paid model cannot spend anything until the budget is changed on purpose.
Every board of every user gets a summary; a **kill switch** read on every sweep stops them without
a deploy.

## Problem Statement

The board Brief (#2340) is built from what is on disk: its voices, its key points, the storylines,
themes and topics of its episodes. It lists; it does not say what the board is about. Stitching the
episodes' summaries together reads as a list of sentences, not as one account (operator
2026-10-12: "option two").

Two facts about prod shape the design:

- **Boards live only on the player plane.** `users/<id>/collections.json` is in the player
  appdata (`compose/docker-compose.player-public.yml`), read and written by the player API alone.
- **That plane has no LLM.** The player API runs with no LLM keys, no `[llm]` SDKs and no scheduler
  (`PODCAST_SERVE_APP_ONLY=1`). The keys and the scheduler are on the operator plane, which does not
  mount the player appdata. Its image deliberately leaves out `[llm]`.

Putting keys into the public-facing API would make every request handler one bug away from the
LLM. Giving the operator plane the player's user data would join two planes kept apart on purpose.

## Goals

1. **Never in a request.** No user action calls the LLM: no open, no tap, no route. A summary is
   made in the background and stored on the board.
2. **Only when the board changed**, fresh within minutes of the last edit, and never more often
   than every 15 minutes per board.
3. **The pipeline's profile and provider code**: the profile names the model; the code is the
   provider factory the pipeline uses, with its `llm_cost` events, traces and response guardrails.
   Changing the profile changes the model and nothing else.
4. **A budget the UI cannot move, and it is zero**: the job refuses any call that would cost
   money, and limits per day and per user cap the load on the DGX.
5. **A kill switch without a deploy**: on by default, flipped by editing one file.
6. **Inputs from our own artifacts**: the episodes' Brief content and the words of saved moments,
   never text a user typed.

## Constraints & Assumptions

- CI never calls a real LLM (AGENTS.md). Tests use a stub provider with recorded answers.
- The player API keeps no LLM keys and no `[llm]` dependency.
- A board's content is private to its user. Sending it to a model provider is new data processing
  and the privacy policy changes in the same release (`PrivacyPolicy` says so of itself).
- Boards are small: tens of episodes, not thousands.

## Design & Implementation

### 1. Where it runs: a `board-summarizer` sidecar on the player stack

A new service in `docker-compose.player-public.yml`, shaped like `digest-scheduler`:

- **Run as the pipeline is run**: the `pipeline-llm` image with its own entrypoint (secrets shim →
  `entrypoint.sh` → `opentelemetry-instrument` when tracing is on), the pipeline-llm service's
  runtime env (`PODCAST_ENV`, `PODCAST_RELEASE`, `PODCAST_GIT_SHA`, the `OTEL_*` settings,
  `OTEL_SERVICE_NAME=podcast-pipeline`), and the CLI's startup (`init_sentry("pipeline")`,
  `init_otel()`, console logging at INFO). Only the command differs. So its `llm_cost` lines,
  traces and errors arrive where the pipeline's do, told apart by `stage=board_summary`.
- **Mounts**: the corpus **read-only**; the player appdata **read-write** (it writes summaries).
- **Network**: egress to the LiteLLM gateway in front of the DGX (`vllm_api_base` =
  `LITELLM_API_BASE`, as for the pipeline; #2342) and the homelab traces endpoint. No inbound
  port, no docker.sock.
- **Secrets**: the pipeline's own — its GlitchTip DSN (`PROD_SENTRY_DSN_PIPELINE`) and its LiteLLM
  gateway key (`LITELLM_PROJ_PODCAST_PROD_KEY` → `LITELLM_API_KEY`), staged by
  `deploy-player.yml` into the player tenant's dir and mounted into this service only. The gateway
  base comes from the same `vars.PROD_LITELLM_API_BASE` deploy-prod uses.
- **Logs**: `player.alloy` keeps `player-board-summarizer-*` and labels it `surface=pipeline`.
- **Loop**: wakes every `BOARD_SUMMARY_INTERVAL_SECONDS` (default 300, offset 60 s so it never
  wakes with the digest), does one sweep, sleeps. A sweep with nothing changed reads files only.
- **DGX availability**: the `:8003` slot is up only in the GPU's research mode. A sweep that cannot
  reach it skips every board, counts `model_unreachable`, and the next sweep tries again.
- **Kill switch**: `board_summaries_settings.json` in the player appdata root, read at the start
  of every sweep, as the API reads `access_policy.json`. Absent, unreadable or `{"enabled": true}`
  runs; `{"enabled": false}` stops new summaries from the next sweep on (within five minutes) — the
  model is not even built. Summaries already made stay. To stop it:
  `echo '{"enabled": false}' > /srv/podcast-scraper/player-appdata/board_summaries_settings.json`
- **The DGX's address**: Docker cannot resolve MagicDNS, so `deploy-player.sh` resolves
  `dgx-llm-1` on every deploy (`tailscale ip -4`) into `DGX_TAILNET_IP`, as it does `homelab`.
  Unresolved, the name points at loopback and the sidecar summarises nothing.

**Nothing changed, nothing sent.** A sweep with no eligible board makes no request to the model
server at all: the provider is built every sweep (that touches no network) but initialised only
before the first call it actually makes, since initialising the vLLM provider asks the DGX which
model it serves. At a sweep every five minutes that would otherwise be 288 requests a day for
nothing. A provider that cannot get ready ends the sweep as `model_unreachable`.

The API keeps no keys and gains nothing that can call a model. The worst a UI or API bug can do is
mark a board as changed; the sidecar decides what runs and when.

### 2. When: changed, quiet for 10 minutes, and not summarised in the last 15

The API already rewrites a board on every add and remove. It also records `content_changed_at` on
the board row; nothing else changes in the API.

A sweep summarises a board when all of these hold:

| Rule | Default | Why |
| --- | --- | --- |
| Its content fingerprint differs from the stored summary's | — | No change, no call |
| No change for `QUIET_MINUTES` | 10 | A burst of adds is one summary, not ten |
| Last summary older than `MIN_HOURS_BETWEEN` | 0.25 (15 min) | A board being edited is not re-summarised every sweep |
| It has at least `MIN_EPISODES` episodes | 2 | One episode already has its own Brief |
| The user is under `MAX_PER_USER_PER_DAY` | 30 | One account cannot take the DGX for the day |
| The sweep is under `MAX_PER_SWEEP` | 50 | A backlog drains over hours, not in one burst |
| Today's spend is under `DAILY_BUDGET_USD` | **0.00** | Zero, and enforced: see below |
| A sweep makes at most one call at a time | 1 | The DGX also serves the pipeline |

**The zero budget.** Before a call, the job asks the profile's provider what the call would cost
(the same pricing `llm_cost` uses). A local model prices at zero and runs. Any price above zero is
refused while the budget is zero, so moving the profile to a paid model stops the job instead of
spending; raising the budget is a deliberate change. After each call the recorded cost is added to
the day's ledger and checked again.

The **fingerprint** is a hash of the sorted episode slugs and the saved moments' ids. The stored
record also names the model and the templates it was written with (each template's name and content
hash, e.g. `openai/summarization/board_system_v1@5c9d61efc4c3+…`); a new model or an edited
template re-summarises each board once. Adding a show, a topic or a link changes no input, so it
calls nothing.

### 3. What it reads: the episodes' Briefs, not the episodes

For each of the board's episodes (saved episodes, and the episodes of saved moments, as the Brief
counts them), in board order, capped at `MAX_INPUT_EPISODES` (20):

- title and show;
- `summary_text` and the summary key points (`build_episode_recap`);
- the top three insights, with their speakers.

Then the words of the saved moments, which are transcript text the user chose.

Nothing the user typed goes in: not the board's name, not link titles, not saved searches. That
closes the prompt-injection route through a board named "ignore your instructions". Each part is
cut to a fixed length, so the prompt has a ceiling whatever the board holds.

### 4. The call: another use case of the provider's own `summarize`

The sidecar loads its profile by name (`BOARD_SUMMARY_PROFILE`, default `prod_dgx_full`: the
DGX vLLM, Qwen3-30B-A3B) and builds the provider with `create_summarization_provider(cfg)`, exactly
as the pipeline does. The board summary is then **just another use case of `summarize`**, the
method that writes every episode summary (operator 2026-10-13): no new provider method, so it gets
the same retries, call metrics, `llm_cost` record, response guardrails (`check_chat_response`) and
prompt metadata the episode summary has in production.

```python
provider.summarize(text="", params={"use_case": SummaryUseCase(
    system_prompt="summarization/board_system_v1",   # cfg.board_summary_system_prompt
    user_prompt="summarization/board_user_v1",       # cfg.board_summary_user_prompt
    document_prompt="summarization/board_document_v1",  # cfg.board_summary_document_prompt
    stage="board_summary", max_length=400,           # cfg.board_summary_max_tokens
    params=material)})
```

`providers/common/summary_use_case.py` holds what a use case changes, and nothing else: the
prompts, the template parameters (the material, not a transcript), the `llm_cost` stage and the
reply budget. Without a use case `summarize` is exactly what it was; a parity test
(`test_providers_are_equal.py`) fails the build if any provider stops carrying it.

**Prompts** are Jinja files in the prompt store, versioned and hashed like every other prompt:
`prompts/shared/summarization/board_system_v1.j2`, `board_user_v1.j2` and `board_document_v1.j2`.
Each provider resolves them under its own folder (`<provider>/summarization/…`) and falls back to
`shared/`, the store's rule for summarization prompts, so one provider can be tuned later without
touching the others.

**Every summary provider** carries the use case:

| Provider | What it renders | Budget |
| --- | --- | --- |
| OpenAI-compatible (vLLM, LiteLLM, OpenAI, DeepSeek, Qwen, Groq), Anthropic, Gemini, Grok, Mistral, Ollama | system + user prompts | `max_tokens` (cloud ones through `cloud_structured_max_output_tokens`) |
| SummLlama (local chat model) | system + user prompts, through its chat template | `max_new_tokens` |
| BART/LED (`transformers`) | the document: the material written as prose, no instructions (a seq2seq model follows none), summarised by its own map/reduce with `cleaning_none` | single-pass `max_length` and reduce `max_new_tokens` |

The sidecar picks chat or document from what the provider can render
(`use_case_input_kind`), prices that input, and records those templates; a provider that can render
neither is refused (`unsupported_provider`). Local providers (ollama, SummLlama, `transformers`) and
the DGX vLLM price at $0 in `config/pricing_assumptions.yaml`.

It **never fails over**: the sidecar builds the profile with `summary_fallback_providers=[]` and
refuses a provider that still comes back wrapped for failover (`failover_configured`). Every call is
priced against the primary provider before it is made; failing over would make it on a provider
nobody priced, a free DGX call quietly becoming a paid cloud one. A summary that cannot be made now
is made on a later sweep.

### 5. What it writes, and what the Brief shows

`users/<id>/board_summaries.json`, under a FileLock as the board store uses:
`{board_id: {text, fingerprint, model, prompt, created_at}}` (`prompt`: the templates' names and
hashes). The day's spend is a ledger
in the appdata root, `board_summaries_ledger.json`. An answer over 1,200 characters is not the
paragraph asked for and is not stored.

`GET /collections/{id}/brief` adds `summary: {text, created_at, stale}`. `stale` is true when the
board changed since. The sheet shows it first, as the episode Brief shows its Summary, with
"Summarised Oct 13". While stale it adds "updating — the board changed since". With no summary yet,
the section is absent. A failed or skipped run leaves the last good summary in place.

## Key Decisions

1. **A sidecar, not the API.** The model endpoint and its key stay out of every process that
   serves a request. This repeats the `digest-scheduler` split.
2. **The DGX profile, a zero budget.** The model is ours and costs nothing; the budget is zero so
   a profile change can never quietly start spending (operator 2026-10-13).
3. **Everyone, with a kill switch.** Every board of every user is summarised; the switch is on by
   default and is there to stop it without a deploy if it misbehaves (operator 2026-10-13).
4. **Changed, quiet, fresh within minutes.** First decided as once a day; changed the same day for
   the beta (operator 2026-10-13): testers build and edit boards and should see the summary follow
   within about 10–15 minutes of their last edit, not the next day. The model is ours and free, so
   the limits bound DGX load and runaway loops (one call at a time, 15 minutes per board, 30 per
   user per day, 50 per sweep); the zero budget still bounds spend. Still never on a user action:
   an edit only makes a board eligible, the sidecar decides.
5. **Brief content in, never user text.** It is cheaper, faithful to what the app already shows, and
   closes injection.

## Alternatives Considered

1. **Generate on open / on a button.** A user action starts a paid call, which the operator ruled
   out (2026-10-12). It is also latency on screen, and the rate limiting would sit on the hot path.
2. **On the operator plane's scheduler.** It would need the player's user data mounted, joining
   two planes kept apart on purpose.
3. **In the player API with keys.** Every request handler would be one bug away from the LLM.
4. **Deterministic stitching.** No cost, but it reads as a list. Kept as the fallback when a
   summary is absent: the rest of the Brief.

## Testing Strategy

- **Unit**: the use-case helpers (`test_summary_use_case.py`); the parity test that every provider
  carries the use case; the prompt contract and packaging tests for the three templates.
- **Provider tier** (`test_summary_use_case_providers.py`, SummLlama and BART/LED in
  `providers/ml/`): for each provider, with its SDK mocked and the REAL templates rendered, the
  board templates are sent, the budget is the use case's, the cost record has
  `stage=board_summary`, the metadata names the templates, and without a use case the episode
  summary is unchanged.
- **Integration**: a sweep over a fixture appdata and the fixture corpus with a stub that renders
  the real templates and returns recorded text; every eligibility rule on its own; that board
  names, link titles and searches never reach the prompt; the material's ceiling and clipping; a new
  template or model re-summarising once. The sidecar runner: a full cycle on the real
  `prod_dgx_full` provider (HTTP client mocked), on SummLlama and on BART/LED, each at $0; a paid
  profile priced and never called; failover and unsupported providers refused. It It writes the summary, skips an unchanged board, respects the per-user
  and daily caps, and leaves the last summary when the stub fails. `/brief` carries it, with
  `stale`.
- **Vitest + Playwright**: the sheet shows the summary, the date, and the stale note; with no
  summary, no section.
- **Live, manual** (done 2026-10-13): sweeps against the DGX vLLM from a dev machine and from the
  `pipeline-llm` image through its own entrypoint (read-only, uid 1000, no key; the `llm_cost`
  line with `stage=board_summary` in `docker logs`); SummLlama and BART/LED
  locally. The DGX and SummLlama write a usable paragraph; BART-base, which follows no
  instructions, repeats itself and stops mid-sentence.

## Rollout & Monitoring

- **Phase 1**: the job, the store, `/brief`, the sheet, the sidecar and the kill switch (on), with
  the privacy policy updated in the same release.
- **Before the first deploy**: confirm the player host reaches `dgx-llm-1:8003` over the tailnet (a
  read-only `GET /v1/models`), as the pipeline does from the operator host.
- **Monitoring**: `llm_cost` with `stage=board_summary` in Grafana (it stays at $0); per sweep:
  eligible, summarised and skipped counts, by reason (`unchanged`, `too_soon`, `quiet`,
  `user_cap`, `budget`, `model_unreachable`, `switched_off`).

## Open Questions

1. The privacy policy wording: board content goes to a model on our own hardware, not to a third
   party, which is a smaller change than a cloud provider would be.

Decided 2026-10-13: the DGX profile; a zero budget; every user; the kill switch on by default; the
per-user, per-sweep, quiet and gap numbers as in the table (the gap and quiet window shortened the
same day for the beta).

## Built (2026-10-13)

- `src/podcast_scraper/server/board_summaries.py` — the rules, the prompt, the store, one sweep.
  It takes the model call as a function and imports no SDK; the API imports it only to read.
- `infra/deploy/board_summarizer.py` — the sidecar loop; `board-summarizer` in
  `compose/docker-compose.player-public.yml`; the DGX address in `infra/deploy/deploy-player.sh`.
- `providers/common/summary_use_case.py` and the use case in every provider's `summarize`; the
  three templates in `prompts/shared/summarization/`; `board_summary_*` fields in `config.py`.
- `/collections/{id}/brief` carries `summary`; the Brief sheet shows it first, with its date and a
  note when the board changed since.
- The privacy policy says boards are summarised by a model on our own servers.

Not done, and needed before it runs in prod: confirm the player host reaches `dgx-llm-1:8003`
(prod resolves the name; a `GET /v1/models` from the player host has not been run); the first
sweep on prod. Routing the DGX through LiteLLM, as every other model is: #2342.

## References

- Boards and the Brief: `src/podcast_scraper/server/app_board_view.py`,
  `server/routes/app_collections.py`, `web/learning-player/src/components/BoardBrief.vue`
- The pattern: `digest-scheduler` in `compose/docker-compose.player-public.yml`,
  `infra/deploy/digest_scheduler.py`
- Provider factory: `src/podcast_scraper/summarization/factory.py`; cost:
  `src/podcast_scraper/workflow/cost_monitoring.py`; fuse: `utils/llm_call_fuse.py`; the use case:
  `providers/common/summary_use_case.py`;
  guardrails: `providers/guardrails/chat.py`
