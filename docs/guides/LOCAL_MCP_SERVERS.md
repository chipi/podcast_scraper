# Running both MCP servers locally (zero-config)

The platform exposes **two** MCP servers — **content** (Close Listening: search/explore the
corpus) and **observability** (`podcast_obs`: how prod/homelab is doing). Both run locally over
**stdio** (local trust — no tokens, no ports, no edge), pointed at whatever `make serve` is
serving. This is the same design that runs them on prod, just without the auth/edge apparatus
(which only guards the networked `http` transport).

## Prerequisite

```bash
make serve        # local api (:8000) + viewer — the content MCP reads this corpus (./output)
```

Your Mac is on the tailnet, so the observability MCP reads the homelab backends
(`homelab:3000` Grafana, `homelab:9428` VictoriaLogs) **directly** via MagicDNS — no extra
wiring. Sources without a token just report "not configured", so it works out of the box for
API-health + logs; drop `PODCAST_OBS_GRAFANA_TOKEN` / `SENTRY_AUTH_TOKEN` into a git-ignored
`.env.obs.dev` to light up alerts/errors.

## Claude Code — nothing to do

This repo ships a project **`.mcp.json`**. Open the repo in Claude Code, approve the project MCP
servers once, and **both** connect automatically:

- `podcast-content` → `make serve-mcp`
- `podcast-observability` → `make serve-obs`

The `make` targets hold all the config (PYTHONPATH, the `make serve` corpus, the local
`config/observability.local.yaml`), so there is nothing to edit.

## Claude Desktop / Cursor — paste this

These clients keep MCP config in their own file. Point them at the same `make` launchers (set
`cwd` to this repo so `make` resolves):

```jsonc
{
  "mcpServers": {
    "podcast-content":       { "command": "make", "args": ["serve-mcp"], "cwd": "/path/to/podcast_scraper" },
    "podcast-observability": { "command": "make", "args": ["serve-obs"], "cwd": "/path/to/podcast_scraper" }
  }
}
```

## Run one by hand

```bash
make serve-mcp    # content MCP on stdio (waits for a client on stdin/stdout)
make serve-obs    # observability MCP on stdio
```

Both are **stdio-only** and **echo-free** (stdout carries the JSON-RPC), so a client can spawn
them directly. They are never a public surface — the networked, admin-gated deployment is the
prod `ops.`/`mcp.` vhosts (see the prod runbook).

## Point observability somewhere else

`config/observability.local.yaml` has a `local` target (localhost api + homelab reads). Add more
targets (or edit it) and select with `--target`:

```bash
make obs-summary                                   # glance at the default (local) target
PODCAST_OBS_CONFIG=config/observability.yaml \
  .venv/bin/python -m podcast_obs summary --target prod
```
