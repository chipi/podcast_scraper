#!/usr/bin/env bash
# Append a before/after snapshot of prod's secret dirs + container roster to the job summary.
#
# Usage:  restage_prod_report.sh <ssh-target> <label>
#
# The container roster matters as much as the dirs: the 2026-09-15 incident presented as two
# nginx containers crashlooping, and only the api containers' Exited(127) explained it. Seeing
# both in one place is what makes the summary actionable.
#
# See #2080.
set -euo pipefail

SSH_TARGET="${1:?usage: restage_prod_report.sh <ssh-target> <label>}"
LABEL="${2:-State}"

SSHF=(-i "${SSH_PROD_IDENTITY:?SSH_PROD_IDENTITY not set}"
      -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o BatchMode=yes)

REMOTE='
for d in podcast-secrets operator-secrets player-secrets; do
  if [ -d "/dev/shm/$d" ]; then
    printf "  %-18s present (%s files)\n" "$d" "$(ls -1 "/dev/shm/$d" | wc -l | tr -d " ")"
  else
    printf "  %-18s MISSING\n" "$d"
  fi
done
echo
docker ps -a --format "  {{.Names}}: {{.Status}}" | sort
'

out="$(ssh "${SSHF[@]}" "$SSH_TARGET" "$REMOTE" 2>&1 || echo "  (unreachable)")"

{
    echo "### ${LABEL}"
    echo '```'
    printf '%s\n' "$out"
    echo '```'
} >> "${GITHUB_STEP_SUMMARY:-/dev/stdout}"

printf '%s\n' "$out"
