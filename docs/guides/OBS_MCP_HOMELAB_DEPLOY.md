# Running the observability MCP control plane on homelab

Goal: an always-on `podcast_obs` MCP server on the homelab that **any agentic tool** (Claude
Desktop/Code, Cursor, …) can connect to over the tailnet to analyze what the pipeline and API are
doing — no public exposure, read-only by default.

The flow is: **CI builds + publishes the image → homelab pulls it → `tailscale serve` fronts it as
HTTPS on the tailnet → agents connect.**

## Why this shape

- **On homelab**, because the backends it reads (VictoriaLogs/Metrics/Traces, GlitchTip, Langfuse,
  Umami) already live there and the tailnet terminates there. The control plane sits next to its data.
- **HTTP transport** (`serve --transport http`, port 8848) so a remote agent can reach it — stdio is
  local-spawn only.
- **Tailnet, not public.** The container binds `127.0.0.1:8848`; `tailscale serve` exposes it as
  HTTPS inside the tailnet. Tailnet membership is the auth, exactly like the Victoria backends and
  the operator Umami in [ADR-126](../adr/ADR-126-operator-analytics-umami-and-podcast-telemetry-ladder.md).
- **Read-only.** `PODCAST_OBS_ALLOW_WRITES` is unset, so the two mutating tools (enrichment
  re-enable/cancel) refuse — any agent can *analyze* but never touch a job.

## 1. Build + publish the image (CI)

`.github/workflows/obs-image.yml` builds `docker/observability/Dockerfile` and pushes a **multi-arch**
(amd64 + arm64 — the homelab is arm64) image to `ghcr.io/<owner>/podcast-obs`:

- Automatically on merge to `main` when `src/podcast_obs/**` or `docker/observability/**` change.
- On demand via **Actions → "Build & publish podcast-obs image" → Run workflow**.

Tags published: `:latest` (default branch) and `:sha-<full-git-sha>` (pin this in prod).

## 2. Deploy on homelab

On the homelab host (on the tailnet, with Docker):

```bash
# in a checkout of this repo, or just copy docker/observability/ over
cd docker/observability
cp .env.homelab.example .env.homelab      # fill the OPTIONAL read tokens (gitignored)

# GHCR is public for this image; if private, first:  echo $GH_PAT | docker login ghcr.io -u <you> --password-stdin
docker compose -f docker-compose.homelab.yml --env-file .env.homelab pull
docker compose -f docker-compose.homelab.yml --env-file .env.homelab up -d
docker compose -f docker-compose.homelab.yml logs -f   # watch it come up
```

With **no** tokens set it still serves logs/metrics/traces/cost/pipeline_stage token-free. Tokens
unlock the extra pivots (GlitchTip issue links, Langfuse LLM traces, Umami usage). See
`.env.homelab.example`.

## 3. Expose over the tailnet as HTTPS

On the homelab host (not in the container):

```bash
tailscale serve --bg --https=443 --set-path /obs http://127.0.0.1:8848
tailscale serve status        # confirm the mapping
```

Agents now reach the MCP endpoint at **`https://homelab.<your-tailnet>.ts.net/obs`** — no public
exposure, and the container's port never leaves localhost.

## 4. Connect an agentic tool

Point the tool's MCP client at the URL (streamable-HTTP):

- **Claude Code:** `claude mcp add --transport http podcast-obs https://homelab.<tailnet>.ts.net/obs`
- **Claude Desktop / Cursor / Windsurf:** add a remote MCP server with that URL. Clients that only
  speak stdio bridge to it with `npx mcp-remote https://homelab.<tailnet>.ts.net/obs`.

The tool's host just needs to be on your tailnet. The agent then has `obs_surface`,
`obs_investigate`, `obs_analytics`, `obs_events`, `obs_metrics`, `obs_traces`, etc.

## 5. Verify

```bash
# from any tailnet device — MCP handshake + tool list over the wire
npx mcp-remote https://homelab.<tailnet>.ts.net/obs --help   # or drive it from your agent:
#   obs_surface(surface="pipeline")        -> live: [cost, logs, pipeline_stage, traces]
#   obs_investigate(run_id="run-...")      -> cost/trace/logs/pipeline_stage joined by run_id
#   obs_analytics()                        -> Umami user actions (once the website id + creds are set)
```

## Security posture

- **No public exposure** — localhost bind + `tailscale serve`; tailnet is the auth.
- **Read-only** — mutating tools refuse unless `PODCAST_OBS_ALLOW_WRITES=1` (do not set it here).
- **Least privilege** — the image runs as a non-root user; the container reads backends, writes
  nothing. Secrets are read-scoped tokens supplied via `--env-file`, never baked into the image.

## Update / rollback

```bash
# update to the newest published image
docker compose -f docker-compose.homelab.yml --env-file .env.homelab pull
docker compose -f docker-compose.homelab.yml --env-file .env.homelab up -d

# rollback: pin the previous good tag and re-up
OBS_IMAGE=ghcr.io/chipi/podcast-obs:sha-<previous-sha> \
  docker compose -f docker-compose.homelab.yml --env-file .env.homelab up -d
```
