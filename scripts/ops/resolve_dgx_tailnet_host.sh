#!/usr/bin/env bash
# Resolve the live DGX MagicDNS hostname for Ollama / embedding shim (RFC-089).
#
# Env:
#   DGX_TAILNET_FQDN (required) — e.g. dgx-llm-1.tail-xxxx.ts.net
#   TAILSCALE_STATUS_JSON_PATH (optional) — read status JSON from this file
#     instead of `tailscale status --json` (tests / dry-run without tailscaled).
#
# stdout: resolved FQDN (lowercase, no trailing dot). stderr: diagnostics.
# exit 0 on success, 1 on failure.

set -euo pipefail

PRIMARY_RAW="${DGX_TAILNET_FQDN:?Set DGX_TAILNET_FQDN (e.g. dgx-llm-1.tail-xxxx.ts.net).}"

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
  echo "resolve_dgx_tailnet_host: expected an FQDN (at least one dot), got: ${PRIMARY_RAW}" >&2
  exit 1
fi

STEM="${PRIMARY_N%%.*}"
DOMAIN="${PRIMARY_N#*.}"

if [[ ! "$STEM" =~ ^dgx-llm(-[0-9]+)?$ ]]; then
  echo "resolve_dgx_tailnet_host: hostname must match dgx-llm or dgx-llm-N (got stem: ${STEM})" >&2
  exit 1
fi

if [[ -n "${TAILSCALE_STATUS_JSON_PATH:-}" ]]; then
  if [[ ! -f "$TAILSCALE_STATUS_JSON_PATH" ]]; then
    echo "resolve_dgx_tailnet_host: TAILSCALE_STATUS_JSON_PATH is not a file: ${TAILSCALE_STATUS_JSON_PATH}" >&2
    exit 1
  fi
  TS_JSON=$(cat "$TAILSCALE_STATUS_JSON_PATH")
elif command -v tailscale >/dev/null 2>&1; then
  TS_JSON=$(tailscale status --json)
else
  echo "resolve_dgx_tailnet_host: tailscale CLI not found; set TAILSCALE_STATUS_JSON_PATH for offline tests." >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "resolve_dgx_tailnet_host: jq is required." >&2
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
    | ($dns_on | map(select(startswith("dgx-llm") and endswith("." + $domain)))) as $dgx_on
    | (
        [$primary]
        + (if ("dgx-llm.\($domain)") != $primary then ["dgx-llm.\($domain)"] else [] end)
        + [range(1; 10) | "dgx-llm-\(.).\($domain)" | select(. != $primary)]
      ) as $candidates
    | ($candidates | map(select(. as $c | $dgx_on | index($c) != null)) | first) as $hit
    | if $hit == null then
        ($dgx_on | join(", ")) as $listed
        | "NO_MATCH|\($listed)"
      else
        "OK|\($hit)"
      end
    ' "$tmp"
)

if [[ "${resolved}" == OK\|* ]]; then
  out="${resolved#OK|}"
  if [[ "${out}" != "${PRIMARY_N}" ]]; then
    echo "resolve_dgx_tailnet_host: live host ${out} differs from DGX_TAILNET_FQDN ${PRIMARY_N} (suffix drift)." >&2
  fi
  printf '%s\n' "$out"
  exit 0
fi

listed="${resolved#NO_MATCH|}"
echo "::error::No online DGX MagicDNS host matched candidates from DGX_TAILNET_FQDN=${PRIMARY_N}." >&2
if [[ -n "${listed}" ]]; then
  echo "::error::Online dgx-llm*.${DOMAIN} seen in tailscale status: ${listed}" >&2
else
  echo "::error::No online peers matched dgx-llm*.${DOMAIN}. Join the tailnet or fix device tag." >&2
fi
echo "::error::Update DGX_TAILNET_FQDN or remove stale machines in Tailscale admin." >&2
exit 1
