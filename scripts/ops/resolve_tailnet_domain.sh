#!/usr/bin/env bash
# Resolve the tailnet MagicDNS domain (e.g. `<tailnet>.ts.net`) for the machine we are on.
#
# WHY THIS EXISTS: ops scripts need to build tailnet URLs (`https://vlogs.<tailnet>.ts.net`), and
# the obvious way to do that is to hardcode the domain. That put the operator's tailnet name in
# three tracked files, which `identifier-denylist` forbids (CONTRIBUTING.md § "No operator
# identifiers in the repo") — but a `<TAILNET>` placeholder is worse than the leak, because the
# scripts stop resolving and fail at the moment you need them.
#
# So: derive it. Every source below is something the machine already knows.
#
# ORDER, and why each rung is there:
#   1. $TAILNET_DOMAIN         explicit override; the escape hatch, and what tests pin.
#   2. $PROD_TAILNET_FQDN      prod. Already injected by prod-ops-health.yml from
#                              `vars.PROD_TAILNET_FQDN` — no new configuration to create or
#                              forget. We take everything after the first label.
#   3. tailscale status        the authoritative answer when a REAL CLI is present. On macOS the
#                              App Store build lives inside the .app bundle; /usr/local/bin/
#                              tailscale here is a 3-line shim that only implements `ip` and
#                              silently exits 0 for everything else — which is exactly why this
#                              rung must not be trusted to fail loudly, and why rung 4 exists.
#   4. resolv.conf search      MagicDNS writes the tailnet into the resolver search path. Verified
#                              present on both a dev Mac and Linux, needs no tailscale binary at
#                              all, and is what makes dev zero-config on EITHER dev machine.
#
# TEST SEAMS (same convention as resolve_prod_tailnet_host.sh's TAILSCALE_STATUS_JSON_PATH):
#   TAILSCALE_STATUS_JSON_PATH  read rung 3's JSON from a file instead of running the CLI
#   RESOLV_CONF_PATH            read rung 4 from here instead of /etc/resolv.conf
# Without these a test cannot exercise the failure path: blanking PATH to disable the CLI also
# breaks `/usr/bin/env bash`, so the script never runs and the test asserts on the wrong error.
#
# stdout: the domain, no leading dot, no trailing dot. exit 0 on success, 1 if nothing resolved.
set -uo pipefail

_strip_dots() {
  local s="${1#.}"
  while [ "${s%.}" != "$s" ]; do s="${s%.}"; done
  printf '%s' "$s"
}

_try_cli_rung() {
  # macOS note: /usr/local/bin/tailscale here is a 3-line shim implementing only `ip`, which exits
  # 0 with empty output for `status`. That is why an empty answer must fall through to the next
  # candidate rather than be treated as "not on a tailnet".
  local ts out
  for ts in tailscale /Applications/Tailscale.app/Contents/MacOS/tailscale; do
    command -v "$ts" >/dev/null 2>&1 || [ -x "$ts" ] || continue
    out=$("$ts" status --json 2>/dev/null \
      | grep -oE '"MagicDNSSuffix"[[:space:]]*:[[:space:]]*"[^"]+"' \
      | head -1 | sed -E 's/.*:[[:space:]]*"([^"]+)".*/\1/')
    if [ -n "$out" ]; then
      _strip_dots "$out"
      return 0
    fi
  done
  return 1
}

resolve_tailnet_domain() {
  local d=""

  # 1. explicit
  if [ -n "${TAILNET_DOMAIN:-}" ]; then
    _strip_dots "$TAILNET_DOMAIN"
    return 0
  fi

  # 2. prod: derive from the FQDN the workflow already provides
  if [ -n "${PROD_TAILNET_FQDN:-}" ] && [ "${PROD_TAILNET_FQDN#*.}" != "$PROD_TAILNET_FQDN" ]; then
    _strip_dots "${PROD_TAILNET_FQDN#*.}"
    return 0
  fi

  # 3. a real tailscale CLI, if one is reachable (or a fixture standing in for it)
  if [ -n "${TAILSCALE_STATUS_JSON_PATH:-}" ]; then
    if [ -r "$TAILSCALE_STATUS_JSON_PATH" ]; then
      d=$(grep -oE '"MagicDNSSuffix"[[:space:]]*:[[:space:]]*"[^"]+"' "$TAILSCALE_STATUS_JSON_PATH" \
        | head -1 | sed -E 's/.*:[[:space:]]*"([^"]+)".*/\1/')
      if [ -n "$d" ]; then
        _strip_dots "$d"
        return 0
      fi
    fi
  else
    _try_cli_rung && return 0
  fi

  # 4. MagicDNS search domain — no tailscale binary required
  local resolv="${RESOLV_CONF_PATH:-/etc/resolv.conf}"
  if [ -r "$resolv" ]; then
    d=$(awk '/^search/{for (i = 2; i <= NF; i++) if ($i ~ /\.ts\.net$/) { print $i; exit }}' \
      "$resolv")
    if [ -n "$d" ]; then
      _strip_dots "$d"
      return 0
    fi
  fi

  return 1
}

# Sourced (`. resolve_tailnet_domain.sh`) → expose the function only.
# Executed → print the domain, or explain what to do when nothing resolved.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  if ! resolve_tailnet_domain; then
    echo "resolve_tailnet_domain: could not determine the tailnet domain." >&2
    echo "  Tried: \$TAILNET_DOMAIN, \$PROD_TAILNET_FQDN, 'tailscale status --json'," >&2
    echo "  and the .ts.net search domain in /etc/resolv.conf." >&2
    echo "  Are you on the tailnet? Otherwise set TAILNET_DOMAIN=<tailnet>.ts.net." >&2
    exit 1
  fi
  echo
fi
