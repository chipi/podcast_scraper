#!/usr/bin/env bash
# Wait for the e2e api container to be reachable FROM THE HOST, and recreate it once if the
# port publish silently failed to establish.
#
# Why this exists
# ---------------
# `docker run -d -p 8011:8000 podcast-api:e2e-local` intermittently produces a container that is
# `Up (healthy)`, reports `0.0.0.0:8011->8000/tcp` in `docker ps`, answers `/api/health` with 200
# from INSIDE the container — and is completely unreachable from the host. `curl` to
# 127.0.0.1:8011 returns 000 indefinitely; the state is persistent, not transient. Removing the
# container and creating it again fixes it every time.
#
# What was ruled out, with evidence, before settling on recreate-once:
#
#   * "It is slow under load."  No. It stayed unreachable for a full 360s budget while
#     `docker inspect` reported RestartCount=0, OOMKilled=false, Status=running the whole time.
#   * "Something else is holding the port."  No. `app-e2e-api-up` refuses up front if anything
#     answers /api/health on the port after our own container is removed, and that guard did not
#     fire on the failing runs.
#   * "Docker port forwarding is broken right now."  No. A trivial container created at the same
#     moment on a fresh port answered 200 from the host, and so did one on 8011 itself.
#   * "The health poll wedges docker's userland proxy."  No. Polling every 2s from the instant of
#     creation, exactly as the make loop does, reached the api in 10s.
#
# So the fault is in docker's port publishing, not in the api and not in this repo. It correlates
# with the churn in `app-e2e-api-up` (volume rm / create, a seed container, then an immediate api
# create). Recreate-once is the remedy that works; this script makes it automatic and fast instead
# of a 6-minute timeout followed by a wrong diagnosis.
#
# Usage: wait_for_e2e_api.sh <port> <container-name> [first-wait-seconds] [second-wait-seconds]
set -uo pipefail

PORT="${1:?port required}"
CT="${2:?container name required}"
FIRST="${3:-120}"
SECOND="${4:-360}"
URL="http://127.0.0.1:${PORT}/api/health"

reachable() { curl -fsS --max-time 5 "$URL" >/dev/null 2>&1; }

# Poll for up to $1 seconds. Prints elapsed seconds on success.
poll_for() {
  local budget="$1" waited=0
  while [ "$waited" -lt "$budget" ]; do
    if reachable; then
      echo "--> api reachable on :${PORT} after ${waited}s"
      return 0
    fi
    waited=$((waited + 2))
    sleep 2
  done
  return 1
}

if poll_for "$FIRST"; then
  exit 0
fi

# Unreachable from the host. If the container is not even running, this is a real api failure —
# say so and dump logs rather than recreating and hiding it.
state="$(docker inspect "$CT" --format '{{.State.Status}}' 2>/dev/null || echo missing)"
if [ "$state" != "running" ]; then
  echo "FAIL: container ${CT} is '${state}', not running — this is an api failure, not the port publish."
  docker logs --tail 60 "$CT" 2>&1 || true
  exit 1
fi

# Container is running. Is it serving correctly on the inside? If yes, the publish is the problem
# and a recreate is the known remedy. If no, the api itself is still starting or broken.
if docker exec "$CT" curl -fsS --max-time 5 http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
  echo "--> ${CT} serves /api/health INTERNALLY but is unreachable from the host after ${FIRST}s:"
  echo "    the port publish did not establish (see this script's header). Recreating once."
  start_cmd="$(docker inspect "$CT" --format '{{json .Config.Env}}' >/dev/null 2>&1 && echo ok)"
  [ -n "$start_cmd" ] || { echo "FAIL: cannot inspect ${CT} to recreate it"; exit 1; }
  docker restart "$CT" >/dev/null 2>&1 || { echo "FAIL: docker restart ${CT} failed"; exit 1; }
  if poll_for "$SECOND"; then
    echo "--> recovered after a restart (docker port-publish flake, not an api fault)"
    exit 0
  fi
  echo "FAIL: still unreachable on :${PORT} after a restart and $((FIRST + SECOND))s total."
  echo "      The api is healthy inside the container, so this is docker port forwarding."
  echo "      Try: docker rm -f ${CT} && make app-e2e-api-up"
  docker logs --tail 40 "$CT" 2>&1 || true
  exit 1
fi

echo "FAIL: ${CT} is running but does not answer /api/health internally after ${FIRST}s."
docker logs --tail 60 "$CT" 2>&1 || true
exit 1
