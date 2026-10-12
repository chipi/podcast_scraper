# RFC-127: Board Summaries — the first LLM call outside the pipeline

- **Status**: Draft
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
2. **Only when the board changed**, and at most **once a day per board**.
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

- **Image**: the pipeline image (it has `[llm]` and the profiles), not the API image.
- **Mounts**: the corpus **read-only**; the player appdata **read-write** (it writes summaries).
- **Network**: egress to the profile's model endpoint only — the DGX vLLM over the tailnet
  (`vllm_api_base`, `dgx-llm-1:8003`). No inbound port, no docker.sock.
- **Secrets**: whatever the profile's provider needs (`VLLM_API_KEY`, through the compose secrets
  shim as the pipeline gets it). No other provider's key.
- **Loop**: wakes every `BOARD_SUMMARY_INTERVAL_SECONDS` (default 3600), does one sweep, sleeps.
- **DGX availability**: the `:8003` slot is up only in the GPU's research mode. A sweep that cannot
  reach it skips every board, counts `model_unreachable`, and the next sweep tries again.
- **Kill switch**: `board_summaries.json` in the player appdata, read at the start of every sweep,
  as the API reads `access_policy.json`. Absent or `{"enabled": true}` runs; `{"enabled": false}`
  stops new summaries from the next sweep on (within the hour). Summaries already made stay.

The API keeps no keys and gains nothing that can call a model. The worst a UI or API bug can do is
mark a board as changed; the sidecar decides what runs and when.

### 2. When: changed, quiet, and not summarised today

The API already rewrites a board on every add and remove. It also records `content_changed_at` on
the board row; nothing else changes in the API.

A sweep summarises a board when all of these hold:

| Rule | Default | Why |
|---|---|---|
| Its content fingerprint differs from the stored summary's | — | No change, no call |
| No change for `QUIET_MINUTES` | 30 | A burst of adds is one summary, not ten |
| Last summary older than `MIN_HOURS_BETWEEN` | 24 | At most once a day per board |
| It has at least `MIN_EPISODES` episodes | 2 | One episode already has its own Brief |
| The user is under `MAX_PER_USER_PER_DAY` | 5 | One account cannot spend the day's budget |
| The sweep is under `MAX_PER_SWEEP` | 50 | A backlog drains over hours, not in one burst |
| Today's spend is under `DAILY_BUDGET_USD` | **0.00** | Zero, and enforced: see below |
| A sweep makes at most one call at a time | 1 | The DGX also serves the pipeline |

**The zero budget.** Before a call, the job asks the profile's provider what the call would cost
(the same pricing `llm_cost` uses). A local model prices at zero and runs. Any price above zero is
refused while the budget is zero, so moving the profile to a paid model stops the job instead of
spending; raising the budget is a deliberate change. After each call the recorded cost is added to
the day's ledger and checked again.

The **fingerprint** is a hash of the sorted episode slugs, the saved moments' ids, the prompt
version and the model. Adding a show, a topic or a link changes no input, so it calls nothing.

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

### 4. The call: the pipeline's provider, its own stage

The sidecar loads its profile by name (`BOARD_SUMMARY_PROFILE`, default `prod_dgx_full`: the
DGX vLLM, `summary_model` Qwen3-30B-A3B) and builds the provider with
`create_summarization_provider(cfg)`, exactly as the pipeline does. A new provider method takes one
prompt and returns text:

```python
def complete_prose(self, prompt: str, *, stage: str, max_tokens: int) -> str: ...
```

It exists because `complete_text` is JSON-only, 800 tokens and tagged `speaker_detection`. The new
method tags cost and traces as `stage="board_summary"`, runs `check_chat_response`, and goes
through the existing `retry_with_metrics` and `llm_call_fuse`.

### 5. What it writes, and what the Brief shows

`users/<id>/board_summaries.json`, under a FileLock as the board store uses:
`{board_id: {text, fingerprint, model, prompt_version, created_at}}`.

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
4. **Changed, quiet, once a day.** It is cheaper than summarising on every change, and fresher than
   a nightly batch.
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

- **Unit**: the fingerprint; every eligibility rule, each on its own; prompt assembly, including
  that board names, link titles and searches never reach it; output length.
- **Integration**: a sweep over a fixture appdata and the fixture corpus with a stub provider
  returning recorded text. It writes the summary, skips an unchanged board, respects the per-user
  and daily caps, and leaves the last summary when the stub fails. `/brief` carries it, with
  `stale`.
- **Vitest + Playwright**: the sheet shows the summary, the date, and the stale note; with no
  summary, no section.
- **Live, manual**: one sweep against the DGX vLLM from a dev machine on the tailnet, on a copy of
  a board, before the first deploy.

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
per-user, per-sweep, quiet and once-a-day numbers as in the table.

## References

- Boards and the Brief: `src/podcast_scraper/server/app_board_view.py`,
  `server/routes/app_collections.py`, `web/learning-player/src/components/BoardBrief.vue`
- The pattern: `digest-scheduler` in `compose/docker-compose.player-public.yml`,
  `infra/deploy/digest_scheduler.py`
- Provider factory: `src/podcast_scraper/summarization/factory.py`; cost:
  `src/podcast_scraper/workflow/cost_monitoring.py`; fuse: `utils/llm_call_fuse.py`;
  guardrails: `providers/guardrails/chat.py`
