# Handover: give podcast_scraper a DEDICATED OpenRouter route inside the LiteLLM gateway

**For:** an agent/session running as the homelab admin user (`markodragoljevic` on the Mac-mini =
`homelab`), with docker/OrbStack + gateway-config + gateway-master-key access.
**Why a handover:** the podcast-scraper session's SSH user (`claude`) is sandboxed — no docker
daemon access (OrbStack socket is `markodragoljevic`-owned → permission denied), can't read the
litellm config, no passwordless sudo, no gateway master key. So it could not make this change.

## Problem
The shared LiteLLM gateway (`~/Projects/agentic-ai-homelab/infra/litellm`, deployed on the mini,
`http://homelab:4001/v1`) routes **both** the triage fleet (`fleet-triage-*`) **and** the podcast
project (`homelab-*` aliases) through the **same** `os.environ/OPENROUTER_API_KEY`. One OpenRouter
account, one weekly budget. A podcast 100-ep run tripped the fleet's **weekly budget cap**
(`OpenrouterException: "Budget limit exceeded (weekly limit)"`) at ~88/105 episodes.

## Goal
Isolate podcast on its **own** OpenRouter connection (separate key = separate budget), **still
through the gateway** (do NOT bypass LiteLLM — the operator requires the gateway for observability,
fallback, and key management). The triage fleet keeps the old key (operator is raising its limit).

## The change (all on the gateway side)
1. **Secret** — on the mini, add to the gateway's deployed `.env` (the file `docker-compose.yml`
   loads; `.env.example` says "copy to `.env` on the mini, chmod 600"):
   ```
   OPENROUTER_API_KEY_PODCAST=<operator provides — a key from podcast's OWN OpenRouter workspace/org,
                                NOT the triage org, so the budget is truly separate>
   ```
   The operator has this value. (One was pasted earlier in the podcast session and is being rotated —
   use the fresh one the operator gives you; never commit it.)

2. **Config** — in `~/Projects/agentic-ai-homelab/infra/litellm/config.yaml`, add **dedicated
   podcast aliases** that route via the new key. Create at least the deepseek-flash control (needed
   now); add the rest of the finale models for later. Pattern:
   ```yaml
     - model_name: podcast-flash-0731
       litellm_params:
         model: openrouter/deepseek/deepseek-v4-flash-0731
         api_key: os.environ/OPENROUTER_API_KEY_PODCAST
     # later finale models (same api_key):
     - model_name: podcast-qwen37-flash
       litellm_params: { model: openrouter/qwen/qwen3.7-flash,        api_key: os.environ/OPENROUTER_API_KEY_PODCAST }
     - model_name: podcast-glm47-flash
       litellm_params: { model: openrouter/z-ai/glm-4.7-flash,        api_key: os.environ/OPENROUTER_API_KEY_PODCAST }
     - model_name: podcast-pro
       litellm_params: { model: openrouter/deepseek/deepseek-v4-pro,  api_key: os.environ/OPENROUTER_API_KEY_PODCAST }
     - model_name: podcast-kimi
       litellm_params: { model: openrouter/moonshotai/kimi-k2.6,      api_key: os.environ/OPENROUTER_API_KEY_PODCAST }
     - model_name: podcast-glm
       litellm_params: { model: openrouter/z-ai/glm-5.2,              api_key: os.environ/OPENROUTER_API_KEY_PODCAST }
   ```
   Leave `fleet-triage-*` on `OPENROUTER_API_KEY` (unchanged). Do NOT repoint the existing
   `homelab-*` aliases — other homelab projects use them; that's why podcast gets its own `podcast-*`.

3. **CONFIG DRIFT — verify against the LIVE gateway first.** The repo `config.yaml` is **stale**:
   the deployed gateway serves aliases NOT in the file (e.g. `homelab-flash-0731`,
   `homelab-qwen37-flash`, `homelab-glm47-flash`) — added out-of-band (admin API `/model/new` or a
   different deployed config). Before editing, reconcile: `curl -s http://homelab:4001/v1/model/info
   -H "Authorization: Bearer $LITELLM_MASTER_KEY"` (or the admin UI) to see the live model_list, and
   make sure your new `podcast-*` entries land in the **actually-deployed** config, not just the repo
   file. If models are managed via the admin API + Postgres rather than `config.yaml`, add the
   `podcast-*` models the same way (`POST /model/new` with the literal or env `api_key`).

4. **Virtual-key allowlist** — the podcast app authenticates with the `proj-podcast-bakeoff` virtual
   key (`LITELLM_API_KEY` in the podcast repo `.env`). Add the new `podcast-*` model names to that
   key's allowed-models list (`POST /key/update` with the master key), or the app gets a 401/not-
   permitted on the new aliases.

5. **Reload** the gateway so it picks up the new `.env` var + config (e.g. `docker compose up -d` /
   restart the litellm service in `infra/litellm`). NOTE: brief downtime affects ALL initiatives on
   this gateway — coordinate with the operator.

## Verify (done = all green)
- `curl http://homelab:4001/v1/model/info` lists `podcast-flash-0731`.
- A gateway chat call as the podcast virtual key succeeds and bills the **new** OpenRouter account:
  ```
  curl -s http://homelab:4001/v1/chat/completions \
    -H "Authorization: Bearer $PODCAST_VIRTUAL_KEY" -H "Content-Type: application/json" \
    -d '{"model":"podcast-flash-0731","messages":[{"role":"user","content":"ok"}],"max_tokens":20,
         "reasoning":{"enabled":false},"provider":{"order":["novita","deepinfra"]}}'
  ```
  Expect a 200 with content, and the spend showing on podcast's OpenRouter dashboard (not triage's).
- Tell the operator when it's up; the podcast session will re-run the deepseek finale through the
  gateway (`--profile bakeoff_litellm_deepseek_flash`, already pointed at `podcast-flash-0731`).

## What the podcast side already did (so you don't redo it)
- The app profile `config/profiles/bakeoff_litellm_deepseek_flash.yaml` is already switched to the
  `podcast-flash-0731` alias (gateway, `http://homelab:4001/v1`, `LITELLM_API_KEY`), reasoning off,
  provider order novita→deepinfra. It just needs the gateway alias to exist.
- No app-side OpenRouter key; podcast talks ONLY to the gateway (as required).
