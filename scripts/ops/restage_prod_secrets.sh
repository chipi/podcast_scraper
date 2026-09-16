#!/usr/bin/env bash
# Stage the PUBLIC-SURFACE tmpfs secret dirs (/dev/shm/{operator,player}-secrets) onto prod.
#
# Deliberately does NOT handle /dev/shm/podcast-secrets. That one has a canonical composite
# action — .github/actions/stage-prod-secrets — used by ten workflows, and it stages ELEVEN
# files including deepgram_api_key and litellm_api_key. Duplicating its list here would be a
# data-loss bug waiting to happen: staging does `rm -rf` then swap, so a copy that drifted to
# nine files would silently replace a complete directory with an incomplete one and take out
# Deepgram and LiteLLM. The control plane goes through the action; this script covers only the
# two surfaces that have no action and were previously inlined in their deploy workflows.
#
# Usage:  SELECTED="operator player" restage_prod_secrets.sh deploy@host
#
# Values never touch disk on the runner: staged under /dev/shm, pushed, then atomically
# swapped into place so a half-written dir is never visible to a starting container.
#
# See #2080.
set -euo pipefail

SSH_TARGET="${1:?usage: restage_prod_secrets.sh <ssh-target>}"
SELECTED="${SELECTED:-operator player}"

SSHF=(-i "${SSH_PROD_IDENTITY:?SSH_PROD_IDENTITY not set}"
      -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o BatchMode=yes)

# A blank secret is worse than a missing one: the container starts and fails later in a way
# nobody traces back to here. Refuse instead.
require() {
    if [ -z "${2:-}" ]; then
        echo "::error::secret $1 is empty — refusing to stage. Set it in repo secrets."
        exit 1
    fi
}

# push_dir <remote-dir-name> <local-staging-dir>
push_dir() {
    local dir="$1" tmp="$2" n
    n=$(find "$tmp" -maxdepth 1 -type f | wc -l | tr -d ' ')
    chmod 400 "$tmp"/*
    ssh "${SSHF[@]}" "$SSH_TARGET" \
        "rm -rf /dev/shm/${dir}.staged && mkdir -p /dev/shm/${dir}.staged && chmod 700 /dev/shm/${dir}.staged"
    scp "${SSHF[@]}" "$tmp"/* "${SSH_TARGET}:/dev/shm/${dir}.staged/"
    ssh "${SSHF[@]}" "$SSH_TARGET" \
        "set -euo pipefail; chmod 400 /dev/shm/${dir}.staged/*; \
         rm -rf /dev/shm/${dir}; mv /dev/shm/${dir}.staged /dev/shm/${dir}; chmod 700 /dev/shm/${dir}"
    rm -rf "$tmp"
    # Mirrors the canonical action's check: an empty file is a live failure, not a warning.
    local empties
    empties=$(ssh "${SSHF[@]}" "$SSH_TARGET" \
        "find /dev/shm/${dir} -mindepth 1 -size 0 | wc -l | tr -d ' '")
    if [ "$empties" != "0" ]; then
        echo "::error::${dir}: ${empties} staged file(s) are EMPTY — a blank secret fails later, not now"
        exit 1
    fi
    echo "::notice::staged ${dir} (${n} files, 0 empty)"
}

for s in $SELECTED; do
    case "$s" in
        podcast)
            echo "::notice::podcast-secrets is staged by .github/actions/stage-prod-secrets, not here — skipping"
            ;;
        operator)
            require PLAYER_GOOGLE_CLIENT_SECRET "${PLAYER_GOOGLE_CLIENT_SECRET:-}"
            require PLAYER_APP_SESSION_SECRET   "${PLAYER_APP_SESSION_SECRET:-}"
            t=$(mktemp -d -p /dev/shm restage.XXXXXX)
            printf '%s' "${PLAYER_GOOGLE_CLIENT_SECRET}" > "$t/app_oauth_google_client_secret"
            printf '%s' "${PLAYER_APP_SESSION_SECRET}"   > "$t/app_session_secret"
            printf '%s' "${PROD_SENTRY_DSN_API:-}"       > "$t/podcast_sentry_dsn_api"
            push_dir operator-secrets "$t"
            ;;
        player)
            require PLAYER_GOOGLE_CLIENT_SECRET "${PLAYER_GOOGLE_CLIENT_SECRET:-}"
            require PLAYER_APP_SESSION_SECRET   "${PLAYER_APP_SESSION_SECRET:-}"
            t=$(mktemp -d -p /dev/shm restage.XXXXXX)
            printf '%s' "${PLAYER_GOOGLE_CLIENT_SECRET}"    > "$t/app_oauth_google_client_secret"
            printf '%s' "${PLAYER_APP_SESSION_SECRET}"      > "$t/app_session_secret"
            printf '%s' "${PROD_SENTRY_DSN_PLAYER_API:-}"   > "$t/podcast_sentry_dsn_api"
            printf '%s' "${PLAYER_INTERNAL_OUTBOX_TOKEN:-}" > "$t/internal_outbox_token"
            push_dir player-secrets "$t"
            ;;
        *)
            echo "::warning::unknown surface '$s' — skipped"
            ;;
    esac
done
