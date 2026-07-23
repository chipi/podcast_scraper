#!/usr/bin/env bash
# player-allowlist.sh — manage who may sign in to the public player (closelistening.app).
#
# The backend gates account creation on an allowlist (server/app_access.py): only emails in
# the GH variable PLAYER_ALLOWED_EMAILS (or domains in PLAYER_ALLOWED_DOMAINS) may sign in.
# This script edits that variable; a player deploy then applies it to the running container.
#
# Usage:
#   scripts/ops/player-allowlist.sh list
#   scripts/ops/player-allowlist.sh add    alice@example.com [bob@example.com ...]
#   scripts/ops/player-allowlist.sh remove alice@example.com
#   scripts/ops/player-allowlist.sh add --deploy alice@example.com   # also trigger the deploy
#
# --deploy dispatches deploy-player.yml (still requires the prod-environment approval in the
# Actions UI). Without --deploy the change only takes effect on the next player deploy.
#
# Requires: gh (authenticated), run from the repo root or anywhere inside it.
set -euo pipefail

VAR="PLAYER_ALLOWED_EMAILS"

_get() { gh variable get "$VAR" 2>/dev/null || true; }

# Normalise a comma/space list -> lowercase, trimmed, de-duplicated, comma-joined (sorted).
_normalise() {
  tr ',' '\n' | tr '[:upper:]' '[:lower:]' | sed 's/[[:space:]]//g' \
    | grep -E '.+@.+\..+' | sort -u | paste -sd, -
}

cmd="${1:-list}"; shift || true

case "$cmd" in
  list)
    cur="$(_get)"
    echo "PLAYER_ALLOWED_EMAILS = ${cur:-<empty — nobody can sign in>}"
    ;;
  add)
    deploy=false
    [ "${1:-}" = "--deploy" ] && { deploy=true; shift; }
    [ "$#" -ge 1 ] || { echo "usage: $0 add [--deploy] <email> [email ...]" >&2; exit 2; }
    new="$( { _get; printf ',%s' "$@"; } | _normalise )"
    gh variable set "$VAR" --body "$new"
    echo "PLAYER_ALLOWED_EMAILS = $new"
    if $deploy; then
      gh workflow run deploy-player.yml --ref production -f confirm=PLAYER_DEPLOY -f override_image_sha=
      echo "-> deploy dispatched; approve the 'prod' environment in the Actions UI to apply."
    else
      echo "-> not applied yet: run a player deploy (or re-run with --deploy)."
    fi
    ;;
  remove)
    [ "${1:-}" = "--deploy" ] && { deploy=true; shift; } || deploy=false
    [ "$#" -ge 1 ] || { echo "usage: $0 remove [--deploy] <email> [email ...]" >&2; exit 2; }
    drop="$(printf '%s\n' "$@" | tr '[:upper:]' '[:lower:]' | sed 's/[[:space:]]//g')"
    new="$( _get | tr ',' '\n' | grep -vxF -f <(printf '%s\n' "$drop") | _normalise )"
    gh variable set "$VAR" --body "$new"
    echo "PLAYER_ALLOWED_EMAILS = ${new:-<empty>}"
    if $deploy; then
      gh workflow run deploy-player.yml --ref production -f confirm=PLAYER_DEPLOY -f override_image_sha=
      echo "-> deploy dispatched; approve the 'prod' environment in the Actions UI to apply."
    else
      echo "-> not applied yet: run a player deploy (or re-run with --deploy)."
    fi
    ;;
  *)
    echo "usage: $0 {list|add [--deploy] <email>...|remove [--deploy] <email>...}" >&2
    exit 2
    ;;
esac
