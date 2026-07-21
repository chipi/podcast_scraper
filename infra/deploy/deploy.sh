#!/usr/bin/env bash
# infra/deploy/deploy.sh — invoked by .github/workflows/deploy-prod.yml after
# the GHA runner has joined the tailnet. Runs ON the VPS as the deploy user.
#
# Pulls latest GHCR images, rolls the compose stack, smoke-tests /api/health
# from inside the api container (GH-745: api port 8000 is not published on
# the host; host loopback :8000 is a false negative).
#
# Exit codes:
#   0  success — stack pulled + up + healthy
#   1  pull failed
#   2  compose up failed
#   3  /api/health did not return 200 within the timeout
#   4  DEPLOY_GIT_SHA / DEPLOY_IMAGE_SHA validation failed (format mismatch or
#      git/image SHA disagree on the 7-char prefix)
set -euo pipefail

REPO_DIR=/srv/podcast-scraper
# vps-prod overlay (#713) layers prod nginx template + landing page on
# top of the stack + prod overlays. Codespaces deliberately omits it.
STACK_FILES=(-f compose/docker-compose.stack.yml -f compose/docker-compose.prod.yml -f compose/docker-compose.vps-prod.yml)

cd "$REPO_DIR"

# ``docker-compose.prod.yml`` requires ``PODCAST_DOCKER_PROJECT_DIR`` for api
# volume interpolation. PROD_RUNBOOK stages it in ``.env``; drill GHA may only
# append ``PODCAST_RELEASE``, so ensure the bind-mount path exists before compose.
if [ ! -f .env ]; then
  install -m 600 /dev/null .env
fi
if ! grep -qE '^PODCAST_DOCKER_PROJECT_DIR=' .env; then
  echo "PODCAST_DOCKER_PROJECT_DIR=$REPO_DIR" >> .env
  chmod 600 .env
fi
# Same pattern as ``.devcontainer/start.sh`` (Codespaces): prod compose binds
# corpus_data to ``PODCAST_CORPUS_HOST_PATH`` on the host → ``/app/output`` in
# containers. VPS default matches cloud-init (``/srv/podcast-scraper/corpus``).
if ! grep -qE '^PODCAST_CORPUS_HOST_PATH=' .env; then
  echo "PODCAST_CORPUS_HOST_PATH=$REPO_DIR/corpus" >> .env
  chmod 600 .env
fi
mkdir -p "$REPO_DIR/corpus"

# Resolve the o11y backend (homelab) tailnet IP for the vps-prod `extra_hosts`.
# Docker's embedded DNS can't resolve Tailscale MagicDNS names, so the api/pipeline
# containers need a static ``homelab:<ip>`` /etc/hosts entry to reach VictoriaTraces
# (traces) + GlitchTip (errors). Resolved fresh each deploy — the IP is stable but
# never hardcoded in the repo. If it can't resolve, traces/errors just won't ship
# (telemetry never breaks the app); warn loudly.
HL_IP="$(tailscale ip -4 homelab 2>/dev/null | head -1 || true)"
if [ -n "$HL_IP" ]; then
  if grep -qE '^HOMELAB_TAILNET_IP=' .env; then
    sed -i "s#^HOMELAB_TAILNET_IP=.*#HOMELAB_TAILNET_IP=$HL_IP#" .env
  else
    echo "HOMELAB_TAILNET_IP=$HL_IP" >> .env
    chmod 600 .env
  fi
else
  echo "[$(date -u +%FT%TZ)] WARN: could not resolve 'homelab' tailnet IP; o11y traces/errors may not ship from containers." >&2
fi

# Compose resolves the project directory from the first `-f` path (`compose/`), so
# it does not load `/srv/podcast-scraper/.env` by default — pass explicitly (VPS + GHA).
COMPOSE=(docker compose --env-file .env)

_set_env_kv() {
  local key="$1"
  local value="$2"
  if grep -qE "^${key}=" .env; then
    sed -i "s|^${key}=.*|${key}=${value}|" .env
  else
    echo "${key}=${value}" >> .env
  fi
  chmod 600 .env
}

DEPLOY_GIT_SHA="${DEPLOY_GIT_SHA:-}"
DEPLOY_IMAGE_SHA="${DEPLOY_IMAGE_SHA:-}"

if [ -n "$DEPLOY_GIT_SHA" ] && [ -n "$DEPLOY_IMAGE_SHA" ]; then
  _git7="${DEPLOY_GIT_SHA:0:7}"
  _img7="${DEPLOY_IMAGE_SHA:0:7}"
  if [ "$_git7" != "$_img7" ]; then
    echo "ERROR: DEPLOY_GIT_SHA (${DEPLOY_GIT_SHA}) and DEPLOY_IMAGE_SHA (${DEPLOY_IMAGE_SHA}) must match (7-char prefix)" >&2
    exit 4
  fi
fi

if [ -n "$DEPLOY_IMAGE_SHA" ]; then
  if ! [[ "$DEPLOY_IMAGE_SHA" =~ ^[a-f0-9]{7,40}$ ]]; then
    echo "ERROR: DEPLOY_IMAGE_SHA must match ^[a-f0-9]{7,40}$ (got: ${DEPLOY_IMAGE_SHA})" >&2
    exit 4
  fi
  _set_env_kv PODCAST_IMAGE_TAG "sha-${DEPLOY_IMAGE_SHA}"
  echo "[$(date -u +%FT%TZ)] pinned PODCAST_IMAGE_TAG=sha-${DEPLOY_IMAGE_SHA}"
fi

# Refresh repo so compose files match the deployed git ref (not always origin/main).
if [ -n "$DEPLOY_GIT_SHA" ]; then
  if ! [[ "$DEPLOY_GIT_SHA" =~ ^[a-f0-9]{7,40}$ ]]; then
    echo "ERROR: DEPLOY_GIT_SHA must match ^[a-f0-9]{7,40}$ (got: ${DEPLOY_GIT_SHA})" >&2
    exit 4
  fi
  # GitHub doesn't honor ``git fetch origin <SHA>`` for arbitrary commit
  # SHAs (only for SHAs that happen to be a branch/tag tip). Fetch all
  # branch refs shallow so the SHA is reachable locally, then reset.
  # First-deploy chicken-and-egg: this branch + the matching deploy-prod.yml
  # change land together; the FIRST deploy after this commit still runs
  # the OLD deploy.sh (the one being replaced), so a redeploy is needed
  # to actually exercise the new code path.
  git fetch --depth=50 origin "+refs/heads/*:refs/remotes/origin/*"
  git reset --hard "$DEPLOY_GIT_SHA"
else
  git fetch --depth=50 origin main
  git reset --hard origin/main
fi

# Repair ``/usr/local/sbin/podcast-tailscale-serve.sh`` from the repo when cloud-init
# embedded a broken copy (Terraform ``$$((`` edge case). Requires sudoers ``install`` rule.
CANONICAL_TS_SERVE="$REPO_DIR/infra/cloud-init/podcast-tailscale-serve.sh"
if [ -f "$CANONICAL_TS_SERVE" ] && command -v sudo >/dev/null 2>&1; then
  install -m 0644 "$CANONICAL_TS_SERVE" /tmp/podcast-tailscale-serve.ci
  if ! sudo -n /usr/bin/install -m 0755 -o root -g root /tmp/podcast-tailscale-serve.ci \
    /usr/local/sbin/podcast-tailscale-serve.sh; then
    echo "[$(date -u +%FT%TZ)] WARN: could not refresh podcast-tailscale-serve.sh from repo (sudo install)." >&2
  fi
fi

echo "[$(date -u +%FT%TZ)] pulling latest images..."
if ! "${COMPOSE[@]}" "${STACK_FILES[@]}" pull; then
  echo "ERROR: docker compose pull failed" >&2
  exit 1
fi

echo "[$(date -u +%FT%TZ)] rolling stack..."
if ! "${COMPOSE[@]}" "${STACK_FILES[@]}" up -d --remove-orphans; then
  echo "ERROR: docker compose up failed" >&2
  exit 2
fi

# Container-local smoke check (authoritative on the VPS). The workflow also
# probes externally over Tailscale; a failure here aborts SSH early.
# Do not curl http://127.0.0.1:8000 on the host — compose only exposes :8000
# inside the Docker network. See docs/guides/PROD_RUNBOOK.md
# "API health checks by context (GH-745)".
echo "[$(date -u +%FT%TZ)] waiting for /api/health inside api container (up to 60s)..."
for i in $(seq 1 12); do
  if "${COMPOSE[@]}" "${STACK_FILES[@]}" exec -T api \
    curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    echo "[$(date -u +%FT%TZ)] /api/health OK after $((i * 5))s (container-local)"
    if command -v sudo >/dev/null 2>&1 && [ -x /usr/local/sbin/podcast-tailscale-serve.sh ]; then
      # MagicDNS HTTPS (:443) for tailnet peers (GHA stack-test, laptops). Do not
      # swallow failures — drill-stack-playwright and PROD_RUNBOOK rely on serve.
      ts_ok=0
      for a in 1 2 3; do
        if sudo -n /usr/local/sbin/podcast-tailscale-serve.sh; then
          ts_ok=1
          break
        fi
        echo "[$(date -u +%FT%TZ)] podcast-tailscale-serve.sh attempt $a failed; retrying in 5s..." >&2
        sleep 5
      done
      if [ "$ts_ok" != 1 ]; then
        echo "WARNING: podcast-tailscale-serve.sh failed after 3 attempts (MagicDNS HTTPS may be down). Deploy continues; fix script on host or reprovision cloud-init. See infra/cloud-init/prod.user-data." >&2
      fi
    fi
    exit 0
  fi
  sleep 5
done

echo "ERROR: /api/health did not return 200 within 60s (container-local exec probe)." >&2
echo "ERROR: If tailnet curl still works, you were likely misled by host loopback :8000 — api is not published there (GH-745)." >&2
echo "ERROR: From this host, try viewer proxy: curl -fsS http://127.0.0.1:\${VIEWER_PORT:-8080}/api/health (prod nginx allows /api/health without auth)." >&2
echo "ERROR: From your laptop, use https://<prod-tailnet-fqdn>/api/health (see PROD_RUNBOOK.md)." >&2
docker logs --tail 50 compose-api-1 >&2 || true
exit 3
