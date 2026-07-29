#!/usr/bin/env bash
# infra/deploy/deploy-operator.sh — bring up the PUBLIC operator surface (RFC-108 / epic #1320).
#
# Runs ON the VPS as the deploy user (invoked by deploy-operator.yml after the runner joins
# the tailnet). Standalone from the privileged operator stack: a LOW-PRIVILEGE operator-read
# backend (no docker.sock, no provider keys — PODCAST_SERVE_OPERATOR_PUBLIC) + the gi-kg-viewer
# SPA, fronted by the shared Caddy edge (ADR-114). The privileged operator / kg-gi surface
# (tailnet-only) is untouched.
#
# Required env (staged by the workflow from GH secrets, or set manually):
#   OPERATOR_DOMAIN                 public operator domain (vhost + health)
#   PODCAST_CORPUS_VOLUME           operator stack's external corpus volume name
#   APP_SESSION_SECRET              (secret) session signing key
#   APP_OAUTH_GOOGLE_CLIENT_ID      Google OAuth client id
#   APP_OAUTH_GOOGLE_CLIENT_SECRET  (secret) Google OAuth client secret
#   APP_ADMIN_EMAILS                operator differentiator — these emails become admin/creator
#
# Exit: 0 ok / 1 compose up failed / 2 vhost/reload failed / 3 health failed
set -euo pipefail

REPO_DIR=/srv/podcast-scraper
cd "$REPO_DIR"

# Operator env — separate from the privileged operator .env; 0600, secrets never committed.
# The workflow scp's a pre-staged .env.operator (secrets NOT passed inline over ssh); a manual
# run builds it here from the environment. Either way it holds the config below.
OPERATOR_ENV="$REPO_DIR/.env.operator"
umask 077
if [ -f "$OPERATOR_ENV" ]; then
  # Staged by the deploy workflow — source it for OPERATOR_DOMAIN / PODCAST_CORPUS_VOLUME.
  set -a
  # shellcheck disable=SC1090
  . "$OPERATOR_ENV"
  set +a
else
  : "${OPERATOR_DOMAIN:?set OPERATOR_DOMAIN}"
  : "${PODCAST_CORPUS_VOLUME:?set PODCAST_CORPUS_VOLUME to the operator stack corpus volume}"
  {
    echo "OPERATOR_DOMAIN=${OPERATOR_DOMAIN}"
    echo "PODCAST_CORPUS_VOLUME=${PODCAST_CORPUS_VOLUME}"
    echo "APP_OAUTH_PROVIDER=google"
    echo "APP_OAUTH_GOOGLE_CLIENT_ID=${APP_OAUTH_GOOGLE_CLIENT_ID:-}"
    echo "APP_OAUTH_GOOGLE_CLIENT_SECRET=${APP_OAUTH_GOOGLE_CLIENT_SECRET:-}"
    echo "APP_SESSION_SECRET=${APP_SESSION_SECRET:-}"
    echo "APP_ADMIN_EMAILS=${APP_ADMIN_EMAILS:-}"
    echo "OPERATOR_PREVIEW_COOKIE=${OPERATOR_PREVIEW_COOKIE:-}"
    echo "APP_SIGNUP_MODE=${APP_SIGNUP_MODE:-allowlist}"
    echo "APP_ALLOWED_EMAILS=${APP_ALLOWED_EMAILS:-}"
    echo "APP_ALLOWED_DOMAINS=${APP_ALLOWED_DOMAINS:-}"
    # OTEL traces (ADR-119). Default OFF for a manual run; set both to enable.
    echo "OTEL_TRACES_EXPORTER=${OTEL_TRACES_EXPORTER:-none}"
    echo "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:-}"
    echo "OPERATOR_PORT=8093"
  } >"$OPERATOR_ENV"
fi
chmod 600 "$OPERATOR_ENV"

: "${OPERATOR_DOMAIN:?OPERATOR_DOMAIN missing from .env.operator and env}"
: "${PODCAST_CORPUS_VOLUME:?PODCAST_CORPUS_VOLUME missing from .env.operator and env}"
# Coming-soon gate cookie secret — substituted into operator.caddy below. REQUIRED: an empty
# value would ship `cl_op_preview=` as the gate, which is guessable → the gate opens for
# anyone. Fail loudly rather than deploy a broken gate.
: "${OPERATOR_PREVIEW_COOKIE:?OPERATOR_PREVIEW_COOKIE missing from .env.operator and env (coming-soon gate cookie secret)}"

# OTEL traces (ADR-119) reach the homelab VictoriaTraces OTLP ingest via the Tailscale
# MagicDNS name `homelab`, which Docker's embedded DNS can't resolve — so the operator
# containers get a static `extra_hosts` entry. Resolve the tailnet IP FRESH here (never
# hardcoded, mirrors deploy.sh) and write it into .env.operator for compose interpolation.
# NON-fatal: if it can't resolve, the compose default (loopback) applies and OTEL export
# just fails silently — the app keeps serving; only traces are lost.
HL_IP="$(tailscale ip -4 homelab 2>/dev/null | head -1 || true)"
if [ -n "$HL_IP" ]; then
  if grep -qE '^HOMELAB_TAILNET_IP=' "$OPERATOR_ENV"; then
    sed -i "s#^HOMELAB_TAILNET_IP=.*#HOMELAB_TAILNET_IP=$HL_IP#" "$OPERATOR_ENV"
  else
    echo "HOMELAB_TAILNET_IP=$HL_IP" >>"$OPERATOR_ENV"
  fi
  echo "[$(date -u +%FT%TZ)] resolved homelab tailnet IP for OTEL traces: $HL_IP"
else
  echo "WARN: could not resolve 'homelab' tailnet IP; operator OTEL traces will not reach VictoriaTraces" >&2
fi

# Pin the api image to a CURRENT sha — NEVER the literal :main tag. CI stopped updating
# :main 2026-05-28, so it is stale: pre-ADR-116, with no /api/app/* consumer surface.
# The deploy workflow stages PODCAST_IMAGE_TAG (newest published sha from main) into
# .env.operator. For a manual run where it is still unset, fall back to the SAME engine the
# privileged operator stack is running ("one engine, two surfaces", ADR-116) by reading its
# live api container. Refuse to deploy if neither resolves — do not silently ship stale :main.
if [ -z "${PODCAST_IMAGE_TAG:-}" ]; then
  op_img=$(docker inspect compose-api-1 --format '{{.Config.Image}}' 2>/dev/null || true)
  PODCAST_IMAGE_TAG="${op_img##*:}"
  case "${PODCAST_IMAGE_TAG:-}" in
    sha-*) echo "[$(date -u +%FT%TZ)] pinned PODCAST_IMAGE_TAG=${PODCAST_IMAGE_TAG} (from running operator api)" ;;
    *) echo "ERROR: PODCAST_IMAGE_TAG unset and could not resolve the operator api image (got: '${PODCAST_IMAGE_TAG:-}'); set PODCAST_IMAGE_TAG=sha-<7> explicitly — refusing to deploy stale :main" >&2; exit 1 ;;
  esac
fi
export PODCAST_IMAGE_TAG

# CRITICAL: run the operator-public stack under its OWN compose project (`-p operator`),
# NOT the default (`compose`, derived from the compose/ dir) which is the PRIVILEGED
# operator stack's project. Without this, `up --remove-orphans` reconciles the operator
# project — removing privileged containers. `-p operator` isolates it: shared read-only
# corpus volume, separate containers/network/lifecycle.
COMPOSE=(docker compose -p operator --env-file "$OPERATOR_ENV" -f compose/docker-compose.operator-public.yml)

# ADR-115 Option A secret delivery (mirrors the player stack). When
# OPERATOR_SECRETS_VIA_FILES=1 the operator runtime secrets are delivered as files in host
# tmpfs /dev/shm/operator-secrets/ (staged from GH Secrets by deploy-operator.yml, never on
# disk), mounted via the secrets overlay -> /run/secrets/*, and exported by the image's
# baked shim. Default off = today's .env.operator env-var behaviour. Fail loudly if the flag
# is on but the files never arrived (exit 5) — never boot with silently-missing secrets.
if [ "${OPERATOR_SECRETS_VIA_FILES:-}" = "1" ]; then
  if [ ! -d /dev/shm/operator-secrets ] || [ -z "$(ls -A /dev/shm/operator-secrets 2>/dev/null)" ]; then
    echo "ERROR: OPERATOR_SECRETS_VIA_FILES=1 but /dev/shm/operator-secrets/ is empty/missing — refusing to boot the operator surface with missing secrets." >&2
    exit 5
  fi
  COMPOSE+=(-f compose/docker-compose.operator-secrets.yml)
  echo "[$(date -u +%FT%TZ)] secrets: file-mounted from /dev/shm/operator-secrets ($(ls -1 /dev/shm/operator-secrets | wc -l | tr -d ' ') files)"
fi

# Ensure the host bind-mount source for per-user data exists and is writable by the
# container's non-root ``podcast`` uid (1000) before compose up. #3.
APPDATA_DIR="${OPERATOR_APPDATA_HOST_PATH:-/srv/podcast-scraper/operator-appdata}"
install -d -m 0750 "$APPDATA_DIR" 2>/dev/null || mkdir -p "$APPDATA_DIR"
chown -R 1000:1000 "$APPDATA_DIR" 2>/dev/null || sudo -n chown -R 1000:1000 "$APPDATA_DIR" || true

echo "[$(date -u +%FT%TZ)] building + starting operator-public..."
"${COMPOSE[@]}" up -d --build --remove-orphans || {
  echo "ERROR: docker compose up failed" >&2
  exit 1
}

# Drop the operator Caddy vhost into the shared sites dir (deploy-owned) with the real
# domain substituted, VALIDATE the merged config once, then restart — roll back on failure
# (ADR-114 validate-before-reload contract).
# operator.caddy hardcodes `operator.closelistening.app` (the real domain), so the domain
# sed is a no-op unless OPERATOR_DOMAIN differs — kept for parity with the player pattern.
# The __OPERATOR_PREVIEW_COOKIE__ substitution IS required: the caddy file carries the
# placeholder that must be swapped for the live gate secret.
echo "[$(date -u +%FT%TZ)] installing operator Caddy vhost for ${OPERATOR_DOMAIN}..."
sed -e "s/operator\.example\.com/${OPERATOR_DOMAIN}/g" \
    -e "s|__OPERATOR_PREVIEW_COOKIE__|${OPERATOR_PREVIEW_COOKIE}|g" \
    "infra/caddy/operator.caddy" >"/etc/caddy/sites/operator.caddy"
# umask 077 makes the `>` land 0600/deploy-owned; the `caddy` user (User=caddy) cannot
# read a 0600 file -> import "permission denied" -> restart fails (prod incident
# 2026-07-23). Match the 0644 sibling vhosts so the caddy user can read the drop-in.
chmod 0644 "/etc/caddy/sites/operator.caddy"
_rollback_operator_vhost() { rm -f "/etc/caddy/sites/operator.caddy"; }
# Validate with `caddy adapt` (Caddyfile -> JSON, reports real config/syntax errors)
# — NOT `caddy validate`, which also PROVISIONS (opens the caddy-owned access.log) and
# false-fails with "permission denied" when run as the deploy user, even on a valid
# config (prod incident 2026-07-23). adapt does not touch the log writer.
if ! caddy adapt --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null 2>&1; then
  echo "ERROR: Caddy config invalid after adding operator vhost; rolling back" >&2
  caddy adapt --config /etc/caddy/Caddyfile --adapter caddyfile 2>&1 | head -5 >&2
  _rollback_operator_vhost
  exit 2
fi
# RESTART, not reload: the base Caddyfile sets `admin off` (T-02), so admin-API-based
# `caddy reload` fails — a vhost change needs a restart (task #27). If caddy doesn't
# come back active, roll back the vhost + restart to the last-good config. (A missing LE
# cert for a not-yet-DNS'd domain is NON-fatal — caddy still starts and retries issuance
# in the background.)
if ! sudo -n /usr/bin/systemctl restart caddy || ! systemctl is-active --quiet caddy; then
  echo "ERROR: caddy failed to restart with operator vhost; rolling back" >&2
  _rollback_operator_vhost
  sudo -n /usr/bin/systemctl restart caddy || true
  exit 2
fi

# Ship the operator stack's container logs to Grafana/Loki via the shared node Alloy
# (ADR-121): drop operator.alloy into the deploy-writable config.d + hot-reload Alloy
# (`docker kill -s HUP alloy`, no sudo — deploy is in the docker group). NON-fatal: a
# logging hiccup must not fail the operator deploy. (operator.alloy exists as of ADR-129 —
# renamed from podcast.alloy; the operator surface owns its log drop-in. Guard kept anyway.)
ALLOY_DIR=/opt/vps-observability/config.d
if [ -d "$ALLOY_DIR" ]; then
  if [ -f infra/observability/operator.alloy ]; then
    echo "[$(date -u +%FT%TZ)] installing operator.alloy log rules + reloading Alloy..."
    cp infra/observability/operator.alloy "$ALLOY_DIR/operator.alloy"
    chmod 0644 "$ALLOY_DIR/operator.alloy"
    docker kill -s HUP alloy >/dev/null 2>&1 \
      || echo "WARN: could not HUP alloy — operator logs may lag until its next reload" >&2
  else
    echo "WARN: infra/observability/operator.alloy absent — skipping Alloy rules (not created yet)" >&2
  fi
else
  echo "WARN: $ALLOY_DIR absent — skipping operator.alloy (node Alloy not deployed here?)" >&2
fi

echo "[$(date -u +%FT%TZ)] health check (operator-read backend, in-container)..."
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
  echo "ERROR: operator backend /api/health did not return 200" >&2
  exit 3
}

# ADR-115 Option A: when delivering via files, assert the secrets actually reached the
# operator-api container as non-empty /run/secrets/* (mirrors the player exit-6 gate) —
# a green /api/health must not mask a keyless app (bad DSN / no session secret / OAuth
# broken). Only checks the flag-on path.
if [ "${OPERATOR_SECRETS_VIA_FILES:-}" = "1" ]; then
  _missing=""
  for s in app_oauth_google_client_secret app_session_secret podcast_sentry_dsn_api; do
    if ! "${COMPOSE[@]}" exec -T api sh -c "[ -s /run/secrets/$s ]" 2>/dev/null; then
      _missing="$_missing $s"
    fi
  done
  if [ -n "$_missing" ]; then
    echo "ERROR: OPERATOR_SECRETS_VIA_FILES=1 but the operator-api container is missing non-empty /run/secrets:${_missing}." >&2
    exit 6
  fi
  echo "[$(date -u +%FT%TZ)] secrets: verified /run/secrets present + non-empty in operator-api"
fi
echo "[$(date -u +%FT%TZ)] operator-public up + healthy; vhost live for https://${OPERATOR_DOMAIN}"
