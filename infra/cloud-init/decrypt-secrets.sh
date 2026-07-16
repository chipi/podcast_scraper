#!/usr/bin/env bash
# Shared multi-tenant secret decrypt (ADR-115). Decrypts a tenant's sops/age
# secrets file into a per-tenant tmpfs dir under /run/secrets, one 0400 file per
# top-level key. The VPS age identity (/etc/vps-secrets/age.key) decrypts any
# tenant's file — tenants encrypt to the published VPS recipient public key.
#
# Usage: decrypt-secrets.sh <tenant> <enc-file>
#   <tenant>   lowercase [a-z0-9-]; names the /run/secrets/<tenant> dir
#   <enc-file> path to the tenant's sops-encrypted YAML (in its repo checkout)
#
# Invoked by a tenant's deploy via the narrow sudoers grant. /run is tmpfs, so
# decrypted secrets never touch persistent disk and vanish on reboot.
#
# Injected into cloud-init via file() (NOT inline) so its ${..} shell syntax is
# not mangled by Terraform templatefile — same precedent as the tailscale-serve
# wrappers.
set -euo pipefail

AGE_KEY=/etc/vps-secrets/age.key
tenant="${1:?usage: decrypt-secrets.sh <tenant> <enc-file>}"
enc="${2:?usage: decrypt-secrets.sh <tenant> <enc-file>}"

case "$tenant" in
  *[!a-z0-9-]* | '') echo "decrypt-secrets: invalid tenant '$tenant'" >&2; exit 2 ;;
esac
[ -f "$AGE_KEY" ] || { echo "decrypt-secrets: missing $AGE_KEY (stage it first)" >&2; exit 3; }
[ -f "$enc" ] || { echo "decrypt-secrets: missing enc file '$enc'" >&2; exit 4; }

# Dir 0700 root: only root can traverse it on the host (host-side protection).
# Files 0444: the container (which may run as a non-root uid, e.g. pipeline's
# ``podcast``) bind-mounts individual files and must be able to read them; the
# 0700 dir still blocks any non-root host process from reaching them by path.
dest="/run/secrets/${tenant}"
install -d -m 0700 -o root -g root "$dest"

umask 077
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
SOPS_AGE_KEY_FILE="$AGE_KEY" sops --decrypt --output-type json "$enc" >"$tmp"

# One 0400 file per top-level key; value written raw (no trailing newline via -j).
for key in $(jq -r 'keys[]' "$tmp"); do
  case "$key" in
    *[!a-zA-Z0-9_]*) echo "decrypt-secrets: skipping odd key '$key'" >&2; continue ;;
  esac
  jq -rj --arg k "$key" '.[$k]' "$tmp" >"${dest}/${key}"
  chmod 0444 "${dest}/${key}"
done

echo "decrypt-secrets: wrote $(find "$dest" -maxdepth 1 -type f | wc -l | tr -d ' ') secret(s) to $dest"
