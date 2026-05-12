#!/usr/bin/env bash
# Resolve the live DR drill VPS MagicDNS hostname for SSH/HTTPS (mirrors
# resolve_prod_tailnet_host.sh for stem dr-podcast / dr-podcast-N).
#
# Env:
#   DRILL_TAILNET_FQDN (required) — e.g. dr-podcast.tail-xxxx.ts.net
#   TAILSCALE_STATUS_JSON_PATH (optional) — read status JSON from this file
#     instead of `tailscale status --json` (tests / dry-run without tailscaled).
#
# stdout: resolved FQDN (lowercase, no trailing dot). stderr: diagnostics.
# exit 0 on success, 1 on failure.

set -euo pipefail

PRIMARY_RAW="${DRILL_TAILNET_FQDN:?Set DRILL_TAILNET_FQDN (e.g. dr-podcast.tail-xxxx.ts.net).}"

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
  echo "resolve_drill_tailnet_host: expected an FQDN (at least one dot), got: ${PRIMARY_RAW}" >&2
  exit 1
fi

STEM="${PRIMARY_N%%.*}"
DOMAIN="${PRIMARY_N#*.}"

if [[ ! "$STEM" =~ ^dr-podcast(-[0-9]+)?$ ]]; then
  echo "resolve_drill_tailnet_host: hostname must match dr-podcast or dr-podcast-N (got stem: ${STEM})" >&2
  exit 1
fi

if [[ -n "${TAILSCALE_STATUS_JSON_PATH:-}" ]]; then
  if [[ ! -f "$TAILSCALE_STATUS_JSON_PATH" ]]; then
    echo "resolve_drill_tailnet_host: TAILSCALE_STATUS_JSON_PATH is not a file: ${TAILSCALE_STATUS_JSON_PATH}" >&2
    exit 1
  fi
  TS_JSON=$(cat "$TAILSCALE_STATUS_JSON_PATH")
elif command -v tailscale >/dev/null 2>&1; then
  TS_JSON=$(tailscale status --json)
else
  echo "resolve_drill_tailnet_host: tailscale CLI not found; set TAILSCALE_STATUS_JSON_PATH for offline tests." >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "resolve_drill_tailnet_host: jq is required." >&2
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

    # MagicDNS ``DNSName`` can lag ``HostName`` right after ``tailscale up``; derive
    # ``<host>.<magicdns-domain>`` when ``HostName`` matches dr-podcast / dr-podcast-N.
    def drill_fqdn_for_peer:
      ((.DNSName // "") | sub("\\.$"; "") | ascii_downcase) as $dn
      | ((.HostName // "") | ascii_downcase) as $hn
      | if ($dn != "" and ($dn | startswith("dr-podcast")) and ($dn | endswith("." + $domain))) then $dn
        elif ($hn | test("^dr-podcast(-[0-9]+)?$")) then "\($hn).\($domain)"
        else empty end;

    # Workspace ``drill`` tags the VPS with ``tag:dr-drill``; MagicDNS can differ
    # from ``dr-podcast*`` if the host registered under another stable name.
    def drill_fqdn_for_tagged_drill_peer:
      select((.Tags // []) | index("tag:dr-drill") != null)
      | ((.DNSName // "") | sub("\\.$"; "") | ascii_downcase) as $dn
      | ((.HostName // "") | ascii_downcase) as $hn
      | if ($dn != "" and ($dn | endswith("." + $domain))) then $dn
        elif ($hn != "") then "\($hn).\($domain)"
        else empty end;

    (
      [raw_nodes[] | select(.Online == true) | drill_fqdn_for_peer]
      + [raw_nodes[] | select(.Online == true) | drill_fqdn_for_tagged_drill_peer]
      | map(select(. != null and . != ""))
      | unique
    ) as $drill_unique
    | (
        [$primary]
        + (if ("dr-podcast.\($domain)") != $primary then ["dr-podcast.\($domain)"] else [] end)
        + [range(1; 10) | "dr-podcast-\(.).\($domain)" | select(. != $primary)]
      ) as $candidates
    | ($candidates | map(select(. as $c | $drill_unique | index($c) != null)) | first) as $hit
    | if $hit != null then
        "OK|\($hit)"
      elif ($drill_unique | length) == 1 then
        "OK|\($drill_unique[0])"
      else
        ($drill_unique | join(", ")) as $listed
        | "NO_MATCH|\($listed)"
      end
    ' "$tmp"
)

if [[ "${resolved}" == OK\|* ]]; then
  out="${resolved#OK|}"
  if [[ "${out}" != "${PRIMARY_N}" ]]; then
    echo "resolve_drill_tailnet_host: live host ${out} differs from DRILL_TAILNET_FQDN ${PRIMARY_N} (suffix drift)." >&2
  fi
  printf '%s\n' "$out"
  exit 0
fi

listed="${resolved#NO_MATCH|}"

# One-line roster: ALL peers matching dr-podcast* under this MagicDNS suffix (any
# Online flag). Helps GH Actions "host not on tailnet yet" loops where only
# ``Online:false`` orphaned names exist or the domain suffix is wrong vs vars.
diag=$(
  jq -r \
    --arg domain "$DOMAIN" '
    def raw_nodes: [(.Peer // {}) | to_entries[] | .value];
    def dns_tail: (.DNSName // "") | sub("\\.$"; "") | ascii_downcase;
    def hn_tail: ((.HostName // "") | ascii_downcase);
    (raw_nodes
      | map(
          . as $p
          | ($p | dns_tail) as $d
          | ($p | hn_tail) as $h
          | select(
              ($d != "" and ($d | startswith("dr-podcast")) and ($d | endswith("." + $domain)))
              or ($h | test("^dr-podcast(-[0-9]+)?$"))
            )
          | "[dns=\($d) host=\($h) Online=\($p.Online)]"
        )
      | join(" ")) // ""
    ' "$tmp"
)
echo "::notice::resolve_drill: dr-podcast* peers (any Online) for .${DOMAIN}: ${diag:-none}" >&2
tag_diag=$(
  jq -r \
    --arg domain "$DOMAIN" '
    def raw_nodes: [(.Peer // {}) | to_entries[] | .value];
    def dns_tail: (.DNSName // "") | sub("\\.$"; "") | ascii_downcase;
    def hn_tail: ((.HostName // "") | ascii_downcase);
    (raw_nodes
      | map(
          . as $p
          | select((.Tags // []) | index("tag:dr-drill") != null)
          | ($p | dns_tail) as $d
          | ($p | hn_tail) as $h
          | "[dns=\($d) host=\($h) Online=\($p.Online)]"
        )
      | join(" ")) // ""
    ' "$tmp"
)
echo "::notice::resolve_drill: tag:dr-drill peers (any Online) for .${DOMAIN}: ${tag_diag:-none}" >&2

echo "::error::No online drill VPS MagicDNS host matched candidates from DRILL_TAILNET_FQDN=${PRIMARY_N}." >&2
if [[ -n "${listed}" ]]; then
  echo "::error::Online dr-podcast*.${DOMAIN} seen in tailscale status: ${listed}" >&2
else
  echo "::error::No online peers matched dr-podcast*.${DOMAIN}. Join the tailnet or fix TS_AUTHKEY." >&2
fi
echo "::error::Update repo variable DRILL_TAILNET_FQDN to the live hostname, or remove stale machines in Tailscale admin." >&2
exit 1
