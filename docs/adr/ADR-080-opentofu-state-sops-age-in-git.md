# ADR-080: OpenTofu State Encrypted In-Repo (sops + age)

- **Status**: Accepted
- **Date**: 2026-05-08
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)

## Context & Problem Statement

OpenTofu state contains sensitive resource metadata. It must not live as plaintext in the
repository or in casual shell history. At hobby scale the project still wants **auditable** state
changes (for example `git diff` showing that state moved) without adopting Terraform Cloud or
another paid control plane unless necessary.

## Decision

We store OpenTofu state **encrypted in git** using **Mozilla sops** with **age** keys:

- Committed blobs such as **`infra/terraform/terraform.tfstate.enc`** (default / prod workspace) and
  **`infra/terraform/terraform.tfstate.enc.drill`** (workspace **`drill`**).
- The **`infra/tofu`** wrapper decrypts to a short-lived plaintext file for OpenTofu, runs the
  requested command, re-encrypts when state changes, and shreds plaintext on exit.

The age **private** key is operator-held (for example 1Password); the age **public** key is in
**`infra/.sops.yaml`**. GitHub Actions uses **`TFSTATE_AGE_KEY`** for the same decrypt/encrypt loop in
CI.

## Rationale

- **Zero new SaaS** for state at current team size.
- **History in git** gives a coarse audit trail that state changed, without exposing secrets in
  cleartext.
- **sops + age** is a small, well-understood toolchain aligned with RFC-082 Decision 5.

## Alternatives Considered

1. **Terraform Cloud / HCP Terraform remote state** — Deferred; adds accounts, RBAC, and cost for
   a single-operator footprint. Migration path remains `tofu state push` if needed later.
2. **Hetzner Object Storage backend** — Valid later upgrade for locking and non-git storage; rejected
   as the initial default to minimize moving parts.
3. **Plaintext state in repo** — Forbidden; violates security and review norms.

## Consequences

- **Positive**: No plaintext state in `main`; local and CI share one pattern.
- **Negative**: No remote **state locking**; two concurrent applies could corrupt state — acceptable
   only with explicit operator discipline (RFC-082 calls this out for team-of-1).
- **Neutral**: Rotating or splitting age keys (for example drill-only) is an operational exercise,
   not a format change.

## Implementation Notes

- **Paths**: `infra/tofu`, `infra/.sops.yaml`, `infra/terraform/terraform.tfstate.enc`,
  `infra/terraform/terraform.tfstate.enc.drill`

## References

- [RFC-082 — Decision 5 — State storage](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)
- [`infra/README.md` (repo root)](https://github.com/chipi/podcast_scraper/blob/main/infra/README.md) — state encryption model
