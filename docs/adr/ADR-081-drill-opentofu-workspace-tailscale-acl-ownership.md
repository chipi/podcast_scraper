# ADR-081: Drill OpenTofu Workspace and Tailscale ACL Ownership

- **Status**: Accepted
- **Date**: 2026-05-08
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)

## Context & Problem Statement

Disaster-recovery drills need a **throwaway** Hetzner footprint and distinct tailnet identities
without risking applies against **production** OpenTofu state or resources. The tailnet-wide
**`tailscale_acl`** object must not be contended by two OpenTofu states, or plans will fight in CI
and at apply time.

## Decision

1. **Separate OpenTofu workspace** named **`drill`** for all drill-only Hetzner resources, with
   **encrypted state** in **`infra/terraform/terraform.tfstate.enc.drill`** (see
   [ADR-080](ADR-080-opentofu-state-sops-age-in-git.md)).
2. **Hetzner API token for drill** is scoped to the drill project only (**`HCLOUD_TOKEN_DRILL`** in
   Actions), never reused for prod applies.
3. **Exactly one workspace owns `tailscale_acl`**: the **default / prod** OpenTofu workspace. The
   **`drill`** workspace sets **`manage_tailscale_acl = false`** so drill state does not manage the
   tailnet ACL resource; ACL changes still land through prod **`infra-apply.yml`** and
   **`tailscale/policy.hujson`**.

## Rationale

- **Blast-radius isolation** — a mistaken `tofu apply` in **`drill`** cannot delete prod servers
  because they are not in that state file.
- **Single writer for ACL** — avoids two-state drift on the same Tailscale API object.
- **Repeatable drills** — destroy workspace **`drill`** and re-apply without touching prod state.

## Alternatives Considered

1. **Second git repo for drill** — Rejected; duplicates provider versions, modules, and review
   workflow without adding isolation beyond a workspace + token split.
2. **Same workspace, different tfvars** — Rejected; one state file mixing prod and drill resources
   raises destroy and plan scope errors.
3. **Drill manages its own tailnet / ACL** — Rejected for this project; one operator tailnet with
   tags (`tag:dr-drill`, `tag:prod`) is the chosen model in RFC-082.

## Consequences

- **Positive**: GitHub workflows can plan/apply/destroy **`drill`** independently (see
  **`drill-infra-*`** and **`dr-drill-exercise.yml`**).
- **Negative**: Operators must remember ACL edits flow through **prod** infra apply, then drill jobs
  consume the updated policy.

## Implementation Notes

- **Paths**: `infra/terraform/` (workspace **`drill`**, `terraform.drill.ci.tfvars`, drill tfvars
  example), `tailscale/policy.hujson`
- **Docs**: [DR_DRILL_RUNBOOK.md](../guides/DR_DRILL_RUNBOOK.md), [`infra/README.md` (repo root)](https://github.com/chipi/podcast_scraper/blob/main/infra/README.md)

## References

- [RFC-082 — DR drill workspace notes](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)
- [ADR-080: OpenTofu state encrypted in-repo](ADR-080-opentofu-state-sops-age-in-git.md)
