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
    echo "PLAYER_PORT=8092"
  } >"$PLAYER_ENV"
fi
chmod 600 "$PLAYER_ENV"

: "${PLAYER_DOMAIN:?PLAYER_DOMAIN missing from .env.player and env}"
: "${PODCAST_CORPUS_VOLUME:?PODCAST_CORPUS_VOLUME missing from .env.player and env}"

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

# Ensure the host bind-mount source for per-user data exists and is writable by the
# container's non-root ``podcast`` uid (1000) before compose up. #3.
APPDATA_DIR="${PLAYER_APPDATA_HOST_PATH:-/srv/podcast-scraper/player-appdata}"
install -d -m 0750 "$APPDATA_DIR" 2>/dev/null || mkdir -p "$APPDATA_DIR"
chown -R 1000:1000 "$APPDATA_DIR" 2>/dev/null || sudo -n chown -R 1000:1000 "$APPDATA_DIR" || true

echo "[$(date -u +%FT%TZ)] building + starting player-public..."
"${COMPOSE[@]}" up -d --build --remove-orphans || {
  echo "ERROR: docker compose up failed" >&2
  exit 1
}

# Drop the player Caddy vhosts into the shared sites dir (deploy-owned) with the real
# domain, VALIDATE the merged config once, then restart — roll back ALL player drop-ins
# on failure (ADR-114 validate-before-reload contract). The subdomains reuse the same
# `player.example.com` placeholder so one sed rewrites all three:
#   player.caddy            -> ${PLAYER_DOMAIN}             SPA + app-only api (coming-soon gated)
#   player-telemetry.caddy  -> telemetry.${PLAYER_DOMAIN}  GlitchTip ingest (browser error SDK)
#   player-analytics.caddy  -> analytics.${PLAYER_DOMAIN}  Umami tracking
PLAYER_VHOSTS=(player player-telemetry player-analytics)
echo "[$(date -u +%FT%TZ)] installing player Caddy vhosts for ${PLAYER_DOMAIN}..."
for v in "${PLAYER_VHOSTS[@]}"; do
  sed "s/player\.example\.com/${PLAYER_DOMAIN}/g" "infra/caddy/${v}.caddy" >"/etc/caddy/sites/${v}.caddy"
  # umask 077 makes the `>` land 0600/deploy-owned; the `caddy` user (User=caddy) cannot
  # read a 0600 file -> import "permission denied" -> restart fails (prod incident
  # 2026-07-23). Match the 0644 sibling vhosts so the caddy user can read the drop-in.
  chmod 0644 "/etc/caddy/sites/${v}.caddy"
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
echo "[$(date -u +%FT%TZ)] player-public up + healthy; vhost live for https://${PLAYER_DOMAIN}"
