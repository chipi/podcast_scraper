#!/usr/bin/env bash
# Create + finalize a Sentry release and mark a deploy for an environment (#803 D2).
#
# This draws the "release boundary" on the Sentry error timeline so "errors started after
# sha-X" is visible. deploy-prod.yml calls it after a successful deploy; kept reusable so it can
# be exercised manually with a throwaway version.
#
# Env (required): SENTRY_AUTH_TOKEN (org Internal Integration with Release: Admin).
# Optional: SENTRY_API (default https://sentry.io/api/0).
# Args: --org O --version V [--environment E (default prod)] [--projects a,b,c]
set -euo pipefail

org=""
version=""
environment="prod"
projects=""
api="${SENTRY_API:-https://sentry.io/api/0}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --org) org="$2"; shift 2 ;;
    --version) version="$2"; shift 2 ;;
    --environment) environment="$2"; shift 2 ;;
    --projects) projects="$2"; shift 2 ;;
    *) echo "mark_sentry_release: unknown arg: $1" >&2; exit 2 ;;
  esac
done

: "${SENTRY_AUTH_TOKEN:?SENTRY_AUTH_TOKEN is required}"
if [ -z "$org" ] || [ -z "$version" ]; then
  echo "mark_sentry_release: --org and --version are required" >&2
  exit 2
fi

projects_json="$(python3 - "$projects" <<'PY'
import json
import sys

print(json.dumps([p.strip() for p in sys.argv[1].split(",") if p.strip()]))
PY
)"

released_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 1. Create (or update) the release with its projects.
curl -fsS -H "Authorization: Bearer $SENTRY_AUTH_TOKEN" -H "Content-Type: application/json" \
  -X POST "$api/organizations/$org/releases/" \
  -d "{\"version\":\"$version\",\"projects\":$projects_json}" >/dev/null

# 2. Finalize it (sets dateReleased).
curl -fsS -H "Authorization: Bearer $SENTRY_AUTH_TOKEN" -H "Content-Type: application/json" \
  -X PUT "$api/organizations/$org/releases/$version/" \
  -d "{\"dateReleased\":\"$released_at\"}" >/dev/null

# 3. Mark a deploy for the environment — this is the boundary marker on the error timeline.
curl -fsS -H "Authorization: Bearer $SENTRY_AUTH_TOKEN" -H "Content-Type: application/json" \
  -X POST "$api/organizations/$org/releases/$version/deploys/" \
  -d "{\"environment\":\"$environment\"}" >/dev/null

echo "mark_sentry_release: marked $version (env=$environment, projects=[$projects])"
