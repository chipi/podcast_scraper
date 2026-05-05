#!/usr/bin/env bash
# Resolve the live prod VPS MagicDNS hostname for SSH/HTTPS (GH-744).
#
# Tailscale may rename a replacement VPS to prod-podcast-1.<tailnet> when the
# canonical prod-podcast name is still held by an offline orphan. GitHub
# Actions keeps vars.PROD_TAILNET_FQDN as the operator-intended canonical name;
# this script picks the first online host in a safe candidate order.
#
# Env:
#   PROD_TAILNET_FQDN (required) — e.g. prod-podcast.tail-xxxx.ts.net
#   TAILSCALE_STATUS_JSON_PATH (optional) — read status JSON from this file
#     instead of `tailscale status --json` (tests / dry-run without tailscaled).
#
# stdout: resolved FQDN (lowercase, no trailing dot). stderr: diagnostics.
# exit 0 on success, 1 on failure.

set -euo pipefail

PRIMARY_RAW="${PROD_TAILNET_FQDN:?Set PROD_TAILNET_FQDN (e.g. prod-podcast.tail-xxxx.ts.net).}"

norm() {
  local s
  s=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
  while [[ "$s" == *. ]]; do
    s=${s%.}
  done
  printf '%s' "$s"
}

PRIMARY_N=$(norm "$PRIMARY_RAW")

if [[ "$PRIMARY_N" != *.* ]]; then
  echo "resolve_prod_tailnet_host: expected an FQDN (at least one dot), got: ${PRIMARY_RAW}" >&2
  exit 1
fi

STEM="${PRIMARY_N%%.*}"
DOMAIN="${PRIMARY_N#*.}"

if [[ ! "$STEM" =~ ^prod-podcast(-[0-9]+)?$ ]]; then
  echo "resolve_prod_tailnet_host: hostname must match prod-podcast or prod-podcast-N (got stem: ${STEM})" >&2
  exit 1
fi

if [[ -n "${TAILSCALE_STATUS_JSON_PATH:-}" ]]; then
  if [[ ! -f "$TAILSCALE_STATUS_JSON_PATH" ]]; then
    echo "resolve_prod_tailnet_host: TAILSCALE_STATUS_JSON_PATH is not a file: ${TAILSCALE_STATUS_JSON_PATH}" >&2
    exit 1
  fi
  TS_JSON=$(cat "$TAILSCALE_STATUS_JSON_PATH")
elif command -v tailscale >/dev/null 2>&1; then
  TS_JSON=$(tailscale status --json)
else
  echo "resolve_prod_tailnet_host: tailscale CLI not found; set TAILSCALE_STATUS_JSON_PATH for offline tests." >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "resolve_prod_tailnet_host: jq is required." >&2
  exit 1
fi

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
printf '%s' "$TS_JSON" >"$tmp"

resolved=$(
  jq -r \
    --arg primary "$PRIMARY_N" \
    --arg domain "$DOMAIN" \
    '
    def raw_nodes:
      [(.Peer // {}) | to_entries[] | .value]
      + (if (.Self | type) == "object" then [.Self] else [] end);

    def dns_tail:
      (.DNSName // "") | sub("\\.$"; "") | ascii_downcase;

    (raw_nodes | map(select(.Online == true)) | map(dns_tail) | map(select(length > 0))) as $dns_on
    | ($dns_on | map(select(startswith("prod-podcast") and endswith("." + $domain)))) as $prod_on
    | (
        [$primary]
        + (if ("prod-podcast.\($domain)") != $primary then ["prod-podcast.\($domain)"] else [] end)
        + [range(1; 10) | "prod-podcast-\(.).\($domain)" | select(. != $primary)]
      ) as $candidates
    | ($candidates | map(select(. as $c | $prod_on | index($c) != null)) | first) as $hit
    | if $hit == null then
        ($prod_on | join(", ")) as $listed
        | "NO_MATCH|\($listed)"
      else
        "OK|\($hit)"
      end
    ' "$tmp"
)

if [[ "${resolved}" == OK\|* ]]; then
  out="${resolved#OK|}"
  if [[ "${out}" != "${PRIMARY_N}" ]]; then
    echo "resolve_prod_tailnet_host: live host ${out} differs from PROD_TAILNET_FQDN ${PRIMARY_N} (suffix drift; GH-744)." >&2
  fi
  printf '%s\n' "$out"
  exit 0
fi

listed="${resolved#NO_MATCH|}"
echo "::error::No online prod VPS MagicDNS host matched candidates derived from PROD_TAILNET_FQDN=${PRIMARY_N}." >&2
if [[ -n "${listed}" ]]; then
  echo "::error::Online prod-podcast*.${DOMAIN} seen in tailscale status: ${listed}" >&2
else
  echo "::error::No online peers matched prod-podcast*.${DOMAIN}. Join the tailnet or fix TS_AUTHKEY." >&2
fi
echo "::error::Update repo variable PROD_TAILNET_FQDN to the live hostname, or remove stale machines in Tailscale admin." >&2
exit 1
