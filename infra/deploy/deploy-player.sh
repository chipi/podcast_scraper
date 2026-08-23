#!/usr/bin/env bash
# infra/deploy/deploy-player.sh — bring up the PUBLIC consumer player (#1163 / ADR-116).
#
# Runs ON the VPS as the deploy user (invoked by deploy-player.yml after the runner joins
# the tailnet). Standalone from the operator stack: a LOW-PRIVILEGE app-only backend (no
# docker.sock, no provider keys — PODCAST_SERVE_APP_ONLY) + the player PWA, fronted by the
# shared Caddy edge (ADR-114). The operator / kg-gi surface is untouched.
#
# Required env (staged by the workflow from GH secrets, or set manually):
#   PLAYER_DOMAIN                   public player domain (vhost + health)
#   PODCAST_CORPUS_VOLUME           operator stack's external corpus volume name
#   APP_SESSION_SECRET              (secret) session signing key
#   APP_OAUTH_GOOGLE_CLIENT_ID      Google OAuth client id
#   APP_OAUTH_GOOGLE_CLIENT_SECRET  (secret) Google OAuth client secret
# Optional (remote MCP, RFC-112 — unset = MCP off, mcp vhost skipped):
#   INTERNAL_MCP_TOKEN             (secret) shared secret gating the verify seam + mcp service
#   APP_MCP_ALLOWED_ORIGINS        optional browser Origin allow-list (claude.ai is server-side)
#   (APP_MCP_ISSUER_URL / APP_MCP_RESOURCE_URL are derived from PLAYER_DOMAIN)
#
# Exit: 0 ok / 1 compose up failed / 2 vhost/reload failed / 3 health failed
set -euo pipefail

REPO_DIR=/srv/podcast-scraper
cd "$REPO_DIR"

# Player env — separate from the operator .env; 0600, secrets never committed. The
# workflow scp's a pre-staged .env.player (secrets NOT passed inline over ssh); a manual
# run builds it here from the environment. Either way it holds the config below.
PLAYER_ENV="$REPO_DIR/.env.player"
umask 077
if [ -f "$PLAYER_ENV" ]; then
  # Staged by the deploy workflow — source it for PLAYER_DOMAIN / PODCAST_CORPUS_VOLUME.
  set -a
  # shellcheck disable=SC1090
  . "$PLAYER_ENV"
  set +a
else
  : "${PLAYER_DOMAIN:?set PLAYER_DOMAIN}"
  : "${PODCAST_CORPUS_VOLUME:?set PODCAST_CORPUS_VOLUME to the operator stack corpus volume}"
  {
    echo "PLAYER_DOMAIN=${PLAYER_DOMAIN}"
    echo "PODCAST_CORPUS_VOLUME=${PODCAST_CORPUS_VOLUME}"
    echo "APP_OAUTH_PROVIDER=google"
    echo "APP_OAUTH_GOOGLE_CLIENT_ID=${APP_OAUTH_GOOGLE_CLIENT_ID:-}"
    echo "APP_OAUTH_GOOGLE_CLIENT_SECRET=${APP_OAUTH_GOOGLE_CLIENT_SECRET:-}"
    echo "APP_SESSION_SECRET=${APP_SESSION_SECRET:-}"
    echo "PLAYER_PREVIEW_COOKIE=${PLAYER_PREVIEW_COOKIE:-}"
    echo "APP_SIGNUP_MODE=${APP_SIGNUP_MODE:-allowlist}"
    echo "APP_ALLOWED_EMAILS=${APP_ALLOWED_EMAILS:-}"
    echo "APP_ALLOWED_DOMAINS=${APP_ALLOWED_DOMAINS:-}"
    echo "APP_ADMIN_EMAILS=${APP_ADMIN_EMAILS:-}"
    # OTEL traces (ADR-119). Default OFF for a manual run; set both to enable.
    echo "OTEL_TRACES_EXPORTER=${OTEL_TRACES_EXPORTER:-none}"
    echo "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-}"
    echo "PLAYER_PORT=8092"
    # Remote MCP (RFC-112). INTERNAL_MCP_TOKEN gates the internal verify seam AND the `mcp`
    # service; EMPTY = the whole MCP surface stays inert (verify 503 → every connect 401), which
    # is the safe default. Set it (GH secret PLAYER_INTERNAL_MCP_TOKEN) to turn MCP on. Issuer +
    # resource are DERIVED from the domain (no secret): the AS is the apex, the resource the mcp
    # subdomain.
    echo "INTERNAL_MCP_TOKEN=${INTERNAL_MCP_TOKEN:-}"
    echo "APP_MCP_ISSUER_URL=https://${PLAYER_DOMAIN}"
    echo "APP_MCP_RESOURCE_URL=https://mcp.${PLAYER_DOMAIN}"
    echo "APP_MCP_ALLOWED_ORIGINS=${APP_MCP_ALLOWED_ORIGINS:-}"
    echo "MCP_PORT=8009"
    # Observability MCP (#56): its own vhost/port + config path; reuses INTERNAL_MCP_TOKEN +
    # APP_MCP_ISSUER_URL above. Mirror the workflow render so a manual bootstrap boots obs too
    # (config path relative to compose/, tokens optional → those sources degrade).
    echo "OBS_MCP_RESOURCE_URL=https://obs.${PLAYER_DOMAIN}"
    echo "OBS_MCP_PORT=8848"
    echo "OBS_CONFIG_HOST_PATH=../observability.yaml"
    echo "PODCAST_OBS_GRAFANA_TOKEN=${PODCAST_OBS_GRAFANA_TOKEN:-}"
    echo "PODCAST_OBS_GITHUB_TOKEN=${PODCAST_OBS_GITHUB_TOKEN:-}"
    echo "SENTRY_AUTH_TOKEN=${SENTRY_AUTH_TOKEN:-}"
  } >"$PLAYER_ENV"
fi
chmod 600 "$PLAYER_ENV"

: "${PLAYER_DOMAIN:?PLAYER_DOMAIN missing from .env.player and env}"
: "${PODCAST_CORPUS_VOLUME:?PODCAST_CORPUS_VOLUME missing from .env.player and env}"
# Coming-soon gate cookie secret — substituted into player.caddy below. REQUIRED: an empty
# value would ship `cl_preview=` as the gate, which is guessable → the gate opens for
# anyone. Fail loudly rather than deploy a broken gate.
: "${PLAYER_PREVIEW_COOKIE:?PLAYER_PREVIEW_COOKIE missing from .env.player and env (coming-soon gate cookie secret)}"

# OTEL traces (ADR-119) reach the homelab VictoriaTraces OTLP ingest via the Tailscale
# MagicDNS name `homelab`, which Docker's embedded DNS can't resolve — so the player
# containers get a static `extra_hosts` entry. Resolve the tailnet IP FRESH here (never
# hardcoded, mirrors deploy.sh) and write it into .env.player for compose interpolation.
# NON-fatal: if it can't resolve, the compose default (loopback) applies and OTEL export
# just fails silently — the app keeps serving; only traces are lost.
HL_IP="$(tailscale ip -4 homelab 2>/dev/null | head -1 || true)"
if [ -n "$HL_IP" ]; then
  if grep -qE '^HOMELAB_TAILNET_IP=' "$PLAYER_ENV"; then
    sed -i "s#^HOMELAB_TAILNET_IP=.*#HOMELAB_TAILNET_IP=$HL_IP#" "$PLAYER_ENV"
  else
    echo "HOMELAB_TAILNET_IP=$HL_IP" >>"$PLAYER_ENV"
  fi
  echo "[$(date -u +%FT%TZ)] resolved homelab tailnet IP for OTEL traces: $HL_IP"
else
  echo "WARN: could not resolve 'homelab' tailnet IP; player OTEL traces will not reach VictoriaTraces" >&2
fi

# Delivery seam (#1412 / ADR-145): the homelab delivery worker polls this player-api's
# /internal/outbox/* over the tailnet (token-gated). Publish the low-privilege player-api on
# the BOX'S OWN tailnet IP so the worker can reach it — tailnet-only, NOT the public edge.
# Resolve fresh (never hardcoded); loopback default = no exposure if it can't resolve.
BOX_IP="$(tailscale ip -4 2>/dev/null | head -1 || true)"
if [ -n "$BOX_IP" ]; then
  if grep -qE '^PLAYER_OUTBOX_LISTEN=' "$PLAYER_ENV"; then
    sed -i "s#^PLAYER_OUTBOX_LISTEN=.*#PLAYER_OUTBOX_LISTEN=$BOX_IP#" "$PLAYER_ENV"
  else
    echo "PLAYER_OUTBOX_LISTEN=$BOX_IP" >>"$PLAYER_ENV"
  fi
  echo "[$(date -u +%FT%TZ)] published player-api outbox on tailnet IP ${BOX_IP}:8099 (worker → /internal/outbox/*)"
else
  echo "WARN: could not resolve the box tailnet IP; player-api outbox stays loopback-only (delivery worker cannot reach it)" >&2
fi

# Pin the api image to a CURRENT sha — NEVER the literal :main tag. CI stopped updating
# :main 2026-05-28, so it is 8 weeks stale: pre-ADR-116, with no /api/app/* consumer
# surface, which makes the player 404 every API call (prod incident 2026-07-23). The
# deploy workflow stages PODCAST_IMAGE_TAG (newest published sha from main) into
# .env.player. For a manual run where it is still unset, fall back to the SAME engine the
# operator stack is running ("one engine, two surfaces", ADR-116) by reading its live api
# container. Refuse to deploy if neither resolves — do not silently ship stale :main.
if [ -z "${PODCAST_IMAGE_TAG:-}" ]; then
  op_img=$(docker inspect compose-api-1 --format '{{.Config.Image}}' 2>/dev/null || true)
  PODCAST_IMAGE_TAG="${op_img##*:}"
  case "${PODCAST_IMAGE_TAG:-}" in
    sha-*) echo "[$(date -u +%FT%TZ)] pinned PODCAST_IMAGE_TAG=${PODCAST_IMAGE_TAG} (from running operator api)" ;;
    *) echo "ERROR: PODCAST_IMAGE_TAG unset and could not resolve the operator api image (got: '${PODCAST_IMAGE_TAG:-}'); set PODCAST_IMAGE_TAG=sha-<7> explicitly — refusing to deploy stale :main" >&2; exit 1 ;;
  esac
fi
export PODCAST_IMAGE_TAG

# CRITICAL: run the player-public stack under its OWN compose project (`-p player`),
# NOT the default (`compose`, derived from the compose/ dir) which is the OPERATOR
# stack's project. Without this, `up --remove-orphans` reconciles the operator project
# — removing the operator viewer and replacing the operator api with the player's
# app-only image (prod incident 2026-07-23). `-p player` isolates it: shared read-only
# corpus volume, separate containers/network/lifecycle.
COMPOSE=(docker compose -p player --env-file "$PLAYER_ENV" -f compose/docker-compose.player-public.yml)

# ADR-115 Option A secret delivery (mirrors the operator stack). When
# PLAYER_SECRETS_VIA_FILES=1 the player runtime secrets are delivered as files in host
# tmpfs /dev/shm/player-secrets/ (staged from GH Secrets by deploy-player.yml, never on
# disk), mounted via the secrets overlay -> /run/secrets/*, and exported by the image's
# baked shim. Default off = today's .env.player env-var behaviour. Fail loudly if the flag
# is on but the files never arrived (exit 5) — never boot with silently-missing secrets.
if [ "${PLAYER_SECRETS_VIA_FILES:-}" = "1" ]; then
  if [ ! -d /dev/shm/player-secrets ] || [ -z "$(ls -A /dev/shm/player-secrets 2>/dev/null)" ]; then
    echo "ERROR: PLAYER_SECRETS_VIA_FILES=1 but /dev/shm/player-secrets/ is empty/missing — refusing to boot the player with missing secrets." >&2
    exit 5
  fi
  COMPOSE+=(-f compose/docker-compose.player-secrets.yml)
  echo "[$(date -u +%FT%TZ)] secrets: file-mounted from /dev/shm/player-secrets ($(ls -1 /dev/shm/player-secrets | wc -l | tr -d ' ') files)"
fi

# Ensure the host bind-mount source for per-user data exists and is writable by the
# container's non-root ``podcast`` uid (1000) before compose up. #3.
APPDATA_DIR="${PLAYER_APPDATA_HOST_PATH:-/srv/podcast-scraper/player-appdata}"
install -d -m 0750 "$APPDATA_DIR" 2>/dev/null || mkdir -p "$APPDATA_DIR"
chown -R 1000:1000 "$APPDATA_DIR" 2>/dev/null || sudo -n chown -R 1000:1000 "$APPDATA_DIR" || true

# NO --build. The app image is PUBLISHED now (stack-test publish job) and pinned by
# PODCAST_IMAGE_TAG like every other container here. Building on the box is what made the
# player unpinnable: `up --build` rebuilt the UI from whatever this checkout happened to be,
# so a deploy pinned to sha-X could serve a UI built from something else entirely.
#
# A sha with no published app image now FAILS here rather than silently building one. That is
# deliberate: it cannot be rolled back past the cutover sha, and a loud failure naming the
# missing tag beats a green deploy serving an unknown build.
# DEPLOY_SERVICES (#56): blank/whitespace = the whole stack (with --remove-orphans, the normal full
# deploy). A real service list = an INDIVIDUAL deploy of just those (e.g. "obs" or "mcp obs") —
# --no-deps so their already-running deps aren't recreated, and NO --remove-orphans so a scoped
# deploy is strictly additive (never tears down siblings).
# Defense in depth (advisor M2): re-validate the allowlist here too, so a manual invocation (not
# just the workflow's runner-side check) can't pass shell metacharacters. Parse into an array first
# so a whitespace-only value collapses to the full-deploy branch (empty array), not a no-remove-orphans full deploy.
read -ra _SVC <<<"${DEPLOY_SERVICES:-}"
if [ "${#_SVC[@]}" -gt 0 ]; then
  case "${DEPLOY_SERVICES}" in
    *[!a-zA-Z0-9\ _-]*)
      echo "ERROR: DEPLOY_SERVICES may contain only letters/digits/space/underscore/hyphen" >&2
      exit 1 ;;
  esac
  echo "[$(date -u +%FT%TZ)] individual deploy at ${PODCAST_IMAGE_TAG} — services: ${_SVC[*]}"
  "${COMPOSE[@]}" up -d --pull always --no-deps "${_SVC[@]}" || {
    echo "ERROR: docker compose up (services: ${_SVC[*]}) failed" >&2
    exit 1
  }
else
  echo "[$(date -u +%FT%TZ)] pulling + starting player-public at ${PODCAST_IMAGE_TAG}..."
  "${COMPOSE[@]}" up -d --pull always --remove-orphans || {
    echo "ERROR: docker compose up failed" >&2
    exit 1
  }
fi

# Drop the player Caddy vhosts into the shared sites dir (deploy-owned) with the real
# domain, VALIDATE the merged config once, then restart — roll back ALL player drop-ins
# on failure (ADR-114 validate-before-reload contract). The subdomains reuse the same
# `player.example.com` placeholder so one sed rewrites all three:
#   player.caddy            -> ${PLAYER_DOMAIN}             SPA + app-only api (coming-soon gated)
#   player-telemetry.caddy  -> telemetry.${PLAYER_DOMAIN}  GlitchTip ingest (browser error SDK)
#   player-analytics.caddy  -> analytics.${PLAYER_DOMAIN}  Umami tracking
# The `mcp` + `ops` vhosts are included ONLY when MCP is enabled (INTERNAL_MCP_TOKEN set) —
# otherwise the mcp/obs containers aren't serving and publishing a public vhost to a dead upstream
# is pointless. The obs (#56) MCP reuses the same verify seam/token, so it gates identically. Its
# `obs.player.example.com` placeholder is rewritten by the same domain sed below.
PLAYER_VHOSTS=(player player-telemetry player-analytics)
if [ -n "${INTERNAL_MCP_TOKEN:-}" ]; then
  PLAYER_VHOSTS+=(mcp obs)
  echo "[$(date -u +%FT%TZ)] MCP enabled — installing mcp.${PLAYER_DOMAIN} + obs.${PLAYER_DOMAIN} vhosts"
else
  echo "[$(date -u +%FT%TZ)] MCP disabled (INTERNAL_MCP_TOKEN unset) — skipping mcp + obs vhosts"
fi
# Tailnet suffix for the player-telemetry/analytics vhosts' __TAILNET__ upstream (Level 3 #1665):
# everything after the first label of the canonical FQDN (HOST.<TAILNET>.ts.net -> <TAILNET>.ts.net).
# Passed inline by deploy-player.yml (SSH does not inherit the runner env). deploy-config.yml does the
# same sed; this path is the FULL player deploy, so it must substitute it too or it clobbers the live
# TLS vhosts with a literal __TAILNET__ (syntactically valid -> `caddy adapt` passes -> silent break).
# `:-` keeps this safe under `set -u` when run by hand without the var; the guard then explains.
TAILNET_SUFFIX="${PROD_TAILNET_FQDN:-}"
TAILNET_SUFFIX="${TAILNET_SUFFIX#*.}"
case "$TAILNET_SUFFIX" in
  *.*) : ;;
  *) echo "ERROR: TAILNET_SUFFIX='${TAILNET_SUFFIX}' (from PROD_TAILNET_FQDN='${PROD_TAILNET_FQDN:-<unset>}') has no dot — expected HOST.<TAILNET>.ts.net; refusing to ship broken telemetry vhosts" >&2; exit 1 ;;
esac
# Player-owned vhost NAMESPACE = every basename this deploy has EVER managed, including
# DEPRECATED / renamed ones. Any drop-in here that is NOT in the active PLAYER_VHOSTS above is
# removed below, so a rename or a disable leaves no orphan. A stale vhost keeps retrying ACME for
# a dead domain and spams the log; the `ops`->`obs` rename (#56) left exactly such an orphan.
# SAFETY: list ONLY player-owned names here — NEVER operator/orrery. `/etc/caddy/sites/` is shared
# across projects and those vhosts have other owners; touching them would break another surface.
# When you rename/retire a player vhost, KEEP its old name in this list so its drop-in gets swept.
PLAYER_MANAGED_VHOSTS=(player player-telemetry player-analytics mcp obs ops)
echo "[$(date -u +%FT%TZ)] installing player Caddy vhosts for ${PLAYER_DOMAIN}..."
for v in "${PLAYER_VHOSTS[@]}"; do
  # Three substitutions: the shared `player.example.com` placeholder -> real domain (all vhosts),
  # __PREVIEW_COOKIE__ -> the gate cookie secret (only player.caddy carries it), and __TAILNET__ ->
  # the tailnet suffix (only player-telemetry/analytics carry it). No-op where a placeholder is
  # absent. Different sed delimiters so neither value's characters can clash with the delimiter.
  sed -e "s/player\.example\.com/${PLAYER_DOMAIN}/g" \
      -e "s|__PREVIEW_COOKIE__|${PLAYER_PREVIEW_COOKIE}|g" \
      -e "s|__TAILNET__|${TAILNET_SUFFIX}|g" \
      "infra/caddy/${v}.caddy" >"/etc/caddy/sites/${v}.caddy"
  # Fail loud if any templating placeholder survived — a new __TOKEN__ in a .caddy file without a
  # matching sed rule would otherwise ship the literal to prod (caddy adapt may still pass on it).
  if grep -nE '__[A-Z0-9_]+__' "/etc/caddy/sites/${v}.caddy"; then
    echo "ERROR: unsubstituted placeholder in ${v}.caddy (see match above) — add a sed rule in deploy-player.sh" >&2
    rm -f "/etc/caddy/sites/${v}.caddy"; exit 1
  fi
  # umask 077 makes the `>` land 0600/deploy-owned; the `caddy` user (User=caddy) cannot
  # read a 0600 file -> import "permission denied" -> restart fails (prod incident
  # 2026-07-23). Match the 0644 sibling vhosts so the caddy user can read the drop-in.
  chmod 0644 "/etc/caddy/sites/${v}.caddy"
done
# Sweep orphaned player drop-ins: any managed vhost we did NOT install this run (MCP disabled, or
# a deprecated/renamed name like `ops`). Runs BEFORE `caddy adapt` so the removal is validated and
# picked up by the restart below. Scoped to PLAYER_MANAGED_VHOSTS, so it can never remove an
# operator/orrery vhost sharing this dir.
for v in "${PLAYER_MANAGED_VHOSTS[@]}"; do
  case " ${PLAYER_VHOSTS[*]} " in *" ${v} "*) continue ;; esac  # still active this run — keep it
  if [ -e "/etc/caddy/sites/${v}.caddy" ]; then
    rm -f "/etc/caddy/sites/${v}.caddy"
    echo "[$(date -u +%FT%TZ)] swept orphaned player vhost drop-in: ${v}.caddy"
  fi
done
_rollback_player_vhosts() { for v in "${PLAYER_VHOSTS[@]}"; do rm -f "/etc/caddy/sites/${v}.caddy"; done; }
# Validate with `caddy adapt` (Caddyfile -> JSON, reports real config/syntax errors)
# — NOT `caddy validate`, which also PROVISIONS (opens the caddy-owned access.log) and
# false-fails with "permission denied" when run as the deploy user, even on a valid
# config (prod incident 2026-07-23). adapt does not touch the log writer.
if ! caddy adapt --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null 2>&1; then
  echo "ERROR: Caddy config invalid after adding player vhosts; rolling back" >&2
  caddy adapt --config /etc/caddy/Caddyfile --adapter caddyfile 2>&1 | head -5 >&2
  _rollback_player_vhosts
  exit 2
fi
# RESTART, not reload: the base Caddyfile sets `admin off` (T-02), so admin-API-based
# `caddy reload` fails — a vhost change needs a restart (task #27). If caddy doesn't
# come back active, roll back the vhosts + restart to the last-good config. (A missing LE
# cert for a not-yet-DNS'd telemetry/analytics subdomain is NON-fatal — caddy still starts
# and retries issuance in the background.)
if ! sudo -n /usr/bin/systemctl restart caddy || ! systemctl is-active --quiet caddy; then
  echo "ERROR: caddy failed to restart with player vhosts; rolling back" >&2
  _rollback_player_vhosts
  sudo -n /usr/bin/systemctl restart caddy || true
  exit 2
fi

# Ship the player stack's container logs to Grafana/Loki via the shared node Alloy
# (ADR-121): drop player.alloy into the deploy-writable config.d + hot-reload Alloy
# (`docker kill -s HUP alloy`, no sudo — deploy is in the docker group). NON-fatal: a
# logging hiccup must not fail the player deploy.
ALLOY_DIR=/opt/vps-observability/config.d
if [ -d "$ALLOY_DIR" ]; then
  echo "[$(date -u +%FT%TZ)] installing player.alloy log rules + reloading Alloy..."
  cp infra/observability/player.alloy "$ALLOY_DIR/player.alloy"
  chmod 0644 "$ALLOY_DIR/player.alloy"
  docker kill -s HUP alloy >/dev/null 2>&1 \
    || echo "WARN: could not HUP alloy — player logs may lag until its next reload" >&2
else
  echo "WARN: $ALLOY_DIR absent — skipping player.alloy (node Alloy not deployed here?)" >&2
fi

echo "[$(date -u +%FT%TZ)] health check (app-only backend, in-container)..."
ok=0
for _ in $(seq 1 30); do
  code=$("${COMPOSE[@]}" exec -T api curl -fsS -o /dev/null -w '%{http_code}' \
    http://127.0.0.1:8000/api/health 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then
    ok=1
    break
  fi
  sleep 2
done
[ "$ok" = "1" ] || {
  echo "ERROR: player backend /api/health did not return 200" >&2
  exit 3
}

# ADR-115 Option A: when delivering via files, assert the secrets actually reached the
# player-api container as non-empty /run/secrets/* (mirrors the operator exit-6 gate) —
# a green /api/health must not mask a keyless app (bad DSN / no session secret / OAuth
# broken). Only checks the flag-on path.
if [ "${PLAYER_SECRETS_VIA_FILES:-}" = "1" ]; then
  _missing=""
  for s in app_oauth_google_client_secret app_session_secret podcast_sentry_dsn_api; do
    if ! "${COMPOSE[@]}" exec -T api sh -c "[ -s /run/secrets/$s ]" 2>/dev/null; then
      _missing="$_missing $s"
    fi
  done
  if [ -n "$_missing" ]; then
    echo "ERROR: PLAYER_SECRETS_VIA_FILES=1 but the player-api container is missing non-empty /run/secrets:${_missing}." >&2
    exit 6
  fi
  echo "[$(date -u +%FT%TZ)] secrets: verified /run/secrets present + non-empty in player-api"
fi
# MCP reachability (RFC-112) — NON-fatal. When enabled, confirm the mcp container answers the
# public RFC 9728 discovery doc (200) and that its bearer gate is live (a token-less MCP POST → 401).
# A failure here does not fail the player deploy; it only flags that the MCP surface needs a look.
if [ -n "${INTERNAL_MCP_TOKEN:-}" ]; then
  echo "[$(date -u +%FT%TZ)] MCP reachability check (in-container :8009)..."
  meta=$("${COMPOSE[@]}" exec -T mcp curl -fsS \
    http://127.0.0.1:8009/.well-known/oauth-protected-resource 2>/dev/null || echo "")
  disc=$([ -n "$meta" ] && echo 200 || echo 000)
  gate=$("${COMPOSE[@]}" exec -T mcp curl -sS -o /dev/null -w '%{http_code}' \
    -X POST http://127.0.0.1:8009/mcp 2>/dev/null || echo 000)
  # Consistency: the discovery `resource` must equal APP_MCP_RESOURCE_URL and its
  # authorization_servers must point at the apex issuer — the wiring most likely to drift and the
  # exact drift that would silently defeat aud-binding (review M2). Best-effort grep, non-fatal.
  want_res="https://mcp.${PLAYER_DOMAIN}"
  want_iss="https://${PLAYER_DOMAIN}"
  consistent=no
  if echo "$meta" | grep -q "\"$want_res\"" && echo "$meta" | grep -q "\"$want_iss\""; then
    consistent=yes
  fi
  if [ "$disc" = "200" ] && [ "$gate" = "401" ] && [ "$consistent" = "yes" ]; then
    echo "[$(date -u +%FT%TZ)] MCP up: discovery 200, gate 401, metadata consistent — https://${want_res#https://}"
  else
    echo "WARN: MCP surface not fully verified (discovery=$disc, token-less gate=$gate, metadata-consistent=$consistent; want 200/401/yes). Check the mcp container + APP_MCP_ISSUER_URL/APP_MCP_RESOURCE_URL match ${want_iss} / ${want_res}." >&2
  fi
fi

# Observability MCP (#56) reachability — NON-fatal, same shape as the mcp probe but python (the obs
# image is python:3.12-slim, no curl) and admin-gated. Skipped when MCP is off or obs wasn't in a
# scoped deploy (the exec fails → odisc=000 → WARN). Confirms discovery 200, token-less gate 401,
# and that the discovery metadata names the ops resource + apex issuer.
if [ -n "${INTERNAL_MCP_TOKEN:-}" ]; then
  echo "[$(date -u +%FT%TZ)] obs MCP reachability check (in-container :8848)..."
  ometa=$("${COMPOSE[@]}" exec -T obs python -c \
    "import urllib.request as u; print(u.urlopen('http://127.0.0.1:8848/.well-known/oauth-protected-resource', timeout=5).read().decode())" 2>/dev/null || echo "")
  odisc=$([ -n "$ometa" ] && echo 200 || echo 000)
  ogate=$("${COMPOSE[@]}" exec -T obs python -c '
import urllib.request as u, urllib.error as e
try:
    u.urlopen(u.Request("http://127.0.0.1:8848/mcp", method="POST", data=b""), timeout=5); print(200)
except e.HTTPError as x: print(x.code)
except Exception: print(0)
' 2>/dev/null || echo 000)
  owant_res="https://obs.${PLAYER_DOMAIN}"
  owant_iss="https://${PLAYER_DOMAIN}"
  oconsistent=no
  if echo "$ometa" | grep -q "\"$owant_res\"" && echo "$ometa" | grep -q "\"$owant_iss\""; then
    oconsistent=yes
  fi
  if [ "$odisc" = "200" ] && [ "$ogate" = "401" ] && [ "$oconsistent" = "yes" ]; then
    echo "[$(date -u +%FT%TZ)] obs MCP up: discovery 200, admin gate 401, metadata consistent — https://${owant_res#https://}"
  else
    echo "WARN: obs MCP surface not fully verified (discovery=$odisc, token-less gate=$ogate, metadata-consistent=$oconsistent; want 200/401/yes). Check the obs container + OBS_MCP_RESOURCE_URL=${owant_res}, and that observability.yaml mounted (H1)." >&2
  fi
fi

echo "[$(date -u +%FT%TZ)] player-public up + healthy; vhost live for https://${PLAYER_DOMAIN}"
