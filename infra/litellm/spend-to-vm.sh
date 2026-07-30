#!/usr/bin/env bash
# spend-to-vm.sh — push the prod LiteLLM gateway's per-key spend to homelab VictoriaMetrics
# (#1357 §metrics, option a — ADR-142). The gateway's own Prometheus endpoint is
# enterprise-gated, so we read the metered truth straight from its Postgres and push it,
# exactly like infra/observability/container-metrics does for container inventory.
#
# Runs ON the box (systemd timer litellm-spend-push.timer). deploy@ is in the docker group,
# so it reads the DB via `docker exec litellm-postgres psql` (no psql needed on the host) and
# pushes to homelab:8428 over the tailnet (host resolves MagicDNS `homelab`; ACL tag:prod →
# homelab-host:8428 already granted). Emits (box="prod"):
#   litellm_key_spend_usd{box,key_alias}       — lifetime spend metered on the virtual key
#   litellm_key_max_budget_usd{box,key_alias}  — the hard budget wall (0 = unset)
#   litellm_key_budget_burn_ratio{box,key_alias} — spend/budget (0 when no budget)
set -uo pipefail

VM_URL="${VM_URL:-http://homelab:8428/api/v1/import/prometheus}"
PG_CONTAINER="${PG_CONTAINER:-litellm-postgres}"

# Per-key spend + budget from the virtual-key table. Tab-separated, no headers/alignment.
sql='SELECT coalesce(key_alias,'"'"'(unaliased)'"'"'), coalesce(spend,0), coalesce(max_budget,0)
     FROM "LiteLLM_VerificationToken" WHERE key_alias IS NOT NULL;'

rows="$(docker exec "$PG_CONTAINER" psql -U litellm -d litellm -At -F $'\t' -c "$sql" 2>/dev/null || true)"
if [ -z "$rows" ]; then
  echo "spend-to-vm: no key rows (gateway not up, no keys minted yet, or DB unreachable) — nothing to push" >&2
  exit 0
fi

metrics="$(printf '%s\n' "$rows" | awk -F'\t' '
  NF>=3 {
    alias=$1; spend=$2+0; budget=$3+0;
    gsub(/\\/,"\\\\",alias); gsub(/"/,"\\\"",alias);
    printf "litellm_key_spend_usd{box=\"prod\",key_alias=\"%s\"} %g\n", alias, spend;
    printf "litellm_key_max_budget_usd{box=\"prod\",key_alias=\"%s\"} %g\n", alias, budget;
    burn = (budget>0) ? spend/budget : 0;
    printf "litellm_key_budget_burn_ratio{box=\"prod\",key_alias=\"%s\"} %g\n", alias, burn;
  }')"

if [ -z "$metrics" ]; then
  echo "spend-to-vm: rows present but produced no metrics — skipping" >&2
  exit 0
fi

if printf '%s\n' "$metrics" | curl -s -m 10 -o /dev/null --data-binary @- "$VM_URL"; then
  n="$(printf '%s\n' "$rows" | grep -c . || true)"
  echo "spend-to-vm: pushed spend for ${n} key(s) to ${VM_URL}"
else
  echo "spend-to-vm: push to ${VM_URL} failed (homelab VM unreachable?) — will retry next tick" >&2
  exit 1
fi
