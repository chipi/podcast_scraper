#!/usr/bin/env bash
# no-secrets-in-image.sh
#
# Runtime assertion: confirm a built Docker image contains no .env files,
# private keys, certificates, or other common secret patterns. Used as a
# pre-publish gate in the stack-test workflow's ``publish`` job (gated on
# ``stack-test`` success) before pushing to GHCR.
#
# Belt-and-suspenders against ``.dockerignore`` drift: if a developer adds
# a new secret file pattern that .dockerignore doesn't know about, this
# script will refuse to publish the image.
#
# Usage:
#     scripts/check/no-secrets-in-image.sh <image-tag>
#     scripts/check/no-secrets-in-image.sh ghcr.io/chipi/podcast-scraper-stack-api:sha-abcdef0
#
# Exit codes:
#     0 — no secrets found (image safe to publish)
#     1 — usage error
#     2 — secrets found in image (blocks publish)
#
# This script intentionally has no dependencies beyond ``docker`` and
# POSIX shell — it must run in the bare GHA runner environment without
# any setup step beyond ``docker login``.

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <image-tag>" >&2
    echo "Example: $0 podcast-scraper-stack-api:latest" >&2
    exit 1
fi

IMAGE="$1"

# Patterns that signal accidental secret leakage. Mirrors .dockerignore's
# Environment-and-secrets block; both must stay in sync. Path separator is
# explicit so we don't false-positive on dirs named ``env`` etc.
#
# We use a single ``find`` invocation with ``-o`` alternation so the runtime
# cost is one filesystem walk, not N walks.
PATTERN_ARGS=(
    -name '.env'
    -o -name '.env.*'
    -o -name '*.key'
    -o -name '*.pem'
    -o -name '*.crt'
    -o -name '*.p12'
    -o -name '*.pfx'
    -o -name 'id_rsa'
    -o -name 'id_rsa.*'
    -o -name 'id_ed25519'
    -o -name 'id_ed25519.*'
    -o -name 'credentials'
    -o -name 'credentials.*'
    -o -name 'secrets'
    -o -name 'secrets.*'
)

# Excluded paths: directories where these patterns are legitimate and not
# operator secrets. ``/etc/ssl/certs`` is the standard Debian CA bundle
# (every base image has *.pem files here — they're public CA certs, not
# secrets). ``/proc`` and ``/sys`` are kernel virtual filesystems and can
# return spurious matches under find.
EXCLUDE_PATHS=(
    -not -path '/proc/*'
    -not -path '/sys/*'
    -not -path '/etc/ssl/*'
    -not -path '/usr/share/ca-certificates/*'
    -not -path '/usr/lib/ssl/*'
    -not -path '/usr/local/share/ca-certificates/*'
    -not -path '/etc/ca-certificates/*'
)

# Run find inside the image. ``2>/dev/null`` suppresses ``Permission denied``
# noise from /proc/* etc. (we're already excluding those paths but find
# still tries to read them once).
echo "Scanning image '${IMAGE}' for secret-pattern files..."
MATCHES="$(docker run --rm --entrypoint sh "${IMAGE}" -c "find / ${EXCLUDE_PATHS[*]} \\( ${PATTERN_ARGS[*]} \\) -type f 2>/dev/null" || true)"

if [ -z "${MATCHES}" ]; then
    echo "OK: no secret-pattern files found in '${IMAGE}'."
    exit 0
fi

echo "" >&2
echo "FAIL: image '${IMAGE}' contains files matching secret patterns:" >&2
echo "${MATCHES}" | sed 's/^/  /' >&2
echo "" >&2
echo "Each path above must be either:" >&2
echo "  1. Removed from the build context via .dockerignore, or" >&2
echo "  2. Explicitly allow-listed in this script if the path is genuinely" >&2
echo "     not a secret (e.g., a public CA bundle the base image ships)." >&2
echo "" >&2
echo "Do not bypass this check — it gates publish to GHCR." >&2
exit 2
