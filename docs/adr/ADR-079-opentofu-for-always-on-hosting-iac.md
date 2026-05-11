# ADR-079: OpenTofu for Always-On Hosting IaC

- **Status**: Accepted
- **Date**: 2026-05-08
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)

## Context & Problem Statement

Always-on production hosting needs declarative infrastructure: Hetzner compute and networking,
Tailscale registration and (optionally) ACL material, and reproducible first-boot bootstrap. The
tooling must stay compatible with the Terraform provider ecosystem the project already references in
RFC-082, while avoiding unnecessary license and vendor lock-in for a hobby-scale operator.

## Decision

We standardize on **OpenTofu** as the IaC engine for `infra/terraform/`, with:

- **`hetznercloud/hcloud`** for server, firewall, SSH key attachment, and related Hetzner objects.
- **`tailscale/tailscale`** for tailnet keys and optional ACL resources co-managed with code.

Operators invoke OpenTofu through the repo **`infra/tofu`** wrapper (decrypt/encrypt state; see
[ADR-080](ADR-080-opentofu-state-sops-age-in-git.md)).

## Rationale

- **Drop-in Terraform compatibility** — existing provider blocks and modules work unchanged.
- **Open-source engine** — avoids HashiCorp Terraform BSL concerns for CI and local applies.
- **Single stack** — one language (HCL), one plan/apply UX for cloud + Tailscale.

## Alternatives Considered

1. **HashiCorp Terraform (BSL)** — Rejected as the default engine; OpenTofu covers the same surface
   with a clearer OSS license story for long-lived automation.
2. **Pulumi / CDK** — Rejected for this layer; adds a second language/runtime for a small footprint
   (one VPS, one tailnet); revisit if infra grows past single-host patterns.
3. **Manual cloud consoles + ad-hoc SSH** — Rejected; fails reproducibility and reviewability goals in
   RFC-082.

## Consequences

- **Positive**: Plans and applies are reviewable in PRs; same toolchain locally and in GitHub Actions.
- **Negative**: Operators must install OpenTofu (or use CI) and understand workspace/state rules for
   drill vs prod ([ADR-081](ADR-081-drill-opentofu-workspace-tailscale-acl-ownership.md)).

## Implementation Notes

- **Paths**: `infra/terraform/*.tf`, `infra/tofu`, `infra/README.md`
- **CI**: `.github/workflows/infra-ci.yml`, `.github/workflows/infra-apply.yml`,
  `.github/workflows/drill-infra-*.yml`

## References

- [RFC-082 — Decision 5 — IaC](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)
