# ADR-143: Migrate Tailscale auth to OAuth clients; self-own tags for provider key-minting

- **Status**: Accepted — executed 2026-08-03 (provider + workflows on OAuth; `policy.hujson`
  tag self-ownership applied to the live tailnet; DR drill green end-to-end)
- **Date**: 2026-08-03
- **Authors**: Marko Dragoljevic, Claude (Opus 4.8), advisor (Fable 5)
- **Related**: [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md) Decision 2
  (Tailscale auth wall — this ADR supersedes its "OAuth is Premium-gated, use Personal
  tokens" premise), [ADR-128](ADR-128-decouple-tailnet-acl-from-hetzner-tofu.md) (ACL ships
  via the GitOps action — the vehicle that applies the tagOwners change here)
- **Tracking**: PR #1388 + hotfixes `0fedf3d2`, `afab42bb`

## Context

RFC-082 Decision 2 chose **Personal API tokens** for Tailscale auth because, at the time,
OAuth clients were gated to Tailscale Premium+ tiers. On the Personal Free plan that meant
two credentials, **both expiring ≤90 days**:

- `TS_AUTHKEY` — device-join auth key (GHA runner + VPS `tailscale up`).
- `TS_API_KEY` — the terraform `tailscale` provider's management token (mints per-server
  tailnet keys, managed the ACL).

The `TS_API_KEY` token lapsed on 2026-08-03 (HTTP 401 "API token invalid") and broke every
infra + tailscale-management workflow — the exact calendar-rotation failure mode RFC-082
warned about. Tailscale has since **lifted the Free-plan OAuth gating** (OAuth clients are
now creatable and functional on this tailnet — verified: the operator created clients and
they mint keys). OAuth clients **do not expire**, which removes the root cause.

Two device-join concerns had *already* moved to OAuth in a prior pass (2026-07-25,
`TS_OAUTH`, `tag:gha-deployer`) and to tofu-minted keys (the VPS join uses
`tailscale_tailnet_key.prod`, not a static secret). So `TS_AUTHKEY` was already orphaned;
only the provider's `TS_API_KEY` remained on an expiring token.

## Decision

1. **Authenticate the terraform `tailscale` provider with an OAuth client**
   (`TS_INFRA_OAUTH_CLIENT_ID`/`_SECRET`; scopes `auth_keys` + `devices:core` + `dns` +
   `policy_file`; bound to `tag:prod` + `tag:dr-drill`). Workflows that hit the raw Tailscale
   REST API exchange the client for a short-lived token (`POST /oauth/token`) at run time.
2. **Self-own `tag:prod` and `tag:dr-drill`** in `tailscale/policy.hujson`:
   `"tag:prod": ["autogroup:admin", "tag:prod"]` (same for `tag:dr-drill`).
3. **Retire `TS_API_KEY` and `TS_AUTHKEY`** (secrets + any live keys deleted).

**Why the self-ownership is required (the non-obvious core of this ADR).** Tailscale allows
an OAuth client to mint an auth key carrying its **full** tag set unconditionally, but a
strict **subset** (one of its tags) only when the client's tags include an *owner* of the
requested tag. `TS_INFRA_OAUTH` carries BOTH tags, but the provider mints **one tag per
workspace** (prod→`tag:prod`, drill→`tag:dr-drill`) — a subset — so every mint was rejected
`400 "requested tags […] are invalid or not permitted"`. `tagOwners` listed only
`autogroup:admin`, which the *Personal token* satisfied (it acts as the admin user) but an
OAuth client does not. Listing each tag as its own owner makes the single-tag subset legal,
because the client (bearing `tag:prod`) is then an owner of `tag:prod`.

## Consequences

**Positive**
- No more expiring Tailscale credentials for infra; the 2026-08-03 outage class is gone.
- One credential model (OAuth) across provider, device-join, and ACL — three single-purpose
  clients (`TS_INFRA_OAUTH`, `TS_OAUTH`, `TS_ACL_OAUTH`).
- Fewer secrets: `TS_API_KEY` + `TS_AUTHKEY` deleted.

**Negative / trade-off**
- Tag self-ownership means **any bearer of `tag:prod` may mint further `tag:prod` keys** —
  that now includes the prod VPS itself (it carries `tag:prod`), not just the CI client.
  Accepted: the prod VPS is already prod-trusted, so the marginal escalation is small.
  The cleaner split is deferred (see Alternatives).

**Neutral**
- The failure surfaces only when the provider actually **mints** a key (a real
  `tofu apply` / DR drill), not on `tofu plan` ("No changes"). Validate migrations with a
  drill or a direct mint probe — see PROD_RUNBOOK "Tailscale credentials" for the one-liner.

## Alternatives considered

- **Revert the provider to a Personal API token.** Restores minting immediately but
  reinstates the ≤90-day expiry that caused the outage. Rejected.
- **Mint both tags on every key.** Would satisfy the full-set rule, but the joining VPS
  would then carry `tag:prod` and inherit every prod ACL grant. Security-wrong. Rejected.
- **Recreate the OAuth client** (chased for hours on 2026-08-03). The client was correctly
  configured the whole time; a fresh clean client failed identically. The blocker was the
  ACL, not the client. The diagnostic lesson: **an OAuth client can always mint its full
  tag set but not a subset** — test the exact tag set the provider requests, not tags
  individually.
- **Dedicated `tag:infra-minter`** on the client, listed as owner of `tag:prod`/`tag:dr-drill`
  (CI identity ≠ prod identity; drop the self-ownership entries). Cleaner, but needs a client
  recreation + a tag definition. **Deferred** — do it the next time the client is touched.
