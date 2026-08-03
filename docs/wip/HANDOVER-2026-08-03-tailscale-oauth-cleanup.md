# Handover — Tailscale OAuth migration cleanup (2026-08-03)

Session close-out for the Tailscale `TS_API_KEY` → OAuth migration. **The functional
work is DONE and on `main`.** What remains is operator-only console hygiene that no
agent can do via API. This file exists so it survives session clearance.

## Done (verified, on `main` — do NOT redo)

- `TS_API_KEY` → OAuth migration for the tofu provider + ~9 workflows (#1388, `a6cd6d62`).
- Teardown `gha-deployer` cleanup OAuth token-exchange fix (`0fedf3d2`).
- **ACL tag self-ownership** for `tag:prod` + `tag:dr-drill` in `tailscale/policy.hujson`
  (`afab42bb`) — the real fix; without it the OAuth client can't mint a single-tag key.
  See **[[ADR-143]]** for the full-set-vs-subset mechanism.
- Docs updated (ADR-143, PROD_RUNBOOK "Tailscale credentials", RFC-082 Decision 2 note,
  stale-ref sweep) — `e66e28d1`.
- DR drill re-run **green end-to-end** (run `30833592422`).
- Credential cleanup so far: deleted the orphaned `tag:dr-drill` auth key
  `kf3e7pM9ts11CNTRL`; deleted GH secrets `TS_API_KEY` + `TS_AUTHKEY`; operator revoked
  all old OAuth clients except the three keepers; all three keepers verified authenticating.

## KEEP — the only 3 OAuth clients we need (do not delete)

| GH secret | Client ID | Purpose |
| --- | --- | --- |
| `TS_INFRA_OAUTH_*` | `ka9ZX4e6QJ11CNTRL` | terraform `tailscale` provider (mints keys, sweeps devices) |
| `TS_ACL_OAUTH_*` | `knKTVm1J4321CNTRL` | `policy.hujson` GitOps apply |
| `TS_OAUTH_*` | `kZTri7oyRE11CNTRL` (assumed — confirm, see item 2) | GHA runner join (`tag:gha-deployer`) |

Shared tailnet (`tail6d0ed4.ts.net`): never touch credentials for `tag:dgx-llm-host`,
`tag:homelab-host`, or orrery.

## TODO — operator-only (console; API can't see these)

**1. Console credential sweep** — [admin/settings/keys](https://login.tailscale.com/admin/settings/keys), low priority (all dead/expired):
- **API access tokens tab** → delete the old Personal token behind the retired `TS_API_KEY`
  (expired 2026-08-03) + any other podcast API tokens (keep tokens used for other tooling).
- **Auth keys tab** → delete leftover static keys with podcast descriptions /
  `tag:prod` / `tag:dr-drill` / `tag:gha-deployer` (CI now uses ephemeral minted keys).
  Keep `dgx-llm-host` / `homelab-host` / orrery keys.
- OAuth clients tab is already clean (only the 3 keepers remain).

**2. Confirm `TS_OAUTH_CLIENT_ID` == `kZTri7oyRE11CNTRL`.** Can't read the secret value.
Self-confirms on the next `deploy-prod` / `backup-corpus-prod` / drill run — watch the
Tailscale-join step. Or trigger `backup-corpus-prod` (non-destructive) to force it.

**3. (Optional, deferred by ADR-143) dedicated `tag:infra-minter`.** Cleaner security split
(CI identity ≠ prod identity → drop the tag self-ownership). Do only when the infra client
is next recreated. Not needed for correctness.

## If Tailscale breaks in ~3 months (rotation / expiry)

OAuth clients don't expire, so this should not recur. If `tofu apply` ever fails with
`requested tags [...] are invalid or not permitted (400)`: the ACL self-ownership is
missing — **do NOT recreate the client** (that dead end cost hours). Fix `tagOwners` in
`policy.hujson`. Full diagnosis + a copy-paste mint probe: PROD_RUNBOOK
"Tailscale credentials (OAuth clients)".
