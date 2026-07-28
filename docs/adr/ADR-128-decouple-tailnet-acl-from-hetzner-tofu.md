# ADR-128: Decouple the tailnet ACL from the Hetzner prod OpenTofu state

- **Status**: Accepted — executed 2026-07-28 (ACL removed from tofu state via `state rm` +
  from config; the GitOps action owns the apply)
- **Date**: 2026-07-28
- **Authors**: Marko Dragoljevic, Claude (Opus 4.8)
- **Related**: [RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md) Decision 2
  (Tailscale auth wall — this ADR amends how the ACL half of it ships),
  [ADR-114](ADR-114-shared-multi-tenant-public-edge-caddy.md) (shared edge; the ACL grants
  that front it)
- **Tracking**: (add issue/epic on adoption)

## Context

The prod OpenTofu state (`infra/terraform/`) manages **two different kinds of resource**
in one root + one state:

- **Hetzner-prod-specific** (rare changes, destructive-capable): `hcloud_server.prod`,
  `hcloud_firewall.main`, `hcloud_network[_subnet].main`, `hcloud_volume.corpus` +
  attachment, `hcloud_storage_box.audio_archive`, `hcloud_ssh_key.operator`, and
  `tailscale_tailnet_key.prod` (the VPS's own join key — legitimately prod-specific).
- **`tailscale_acl.main`** — the **tailnet-WIDE** policy (`tailscale/policy.hujson`). It
  governs the entire tailnet — the homelab mini, the DGX, GHA runners, orrery, *and* the
  prod VPS — not just this Hetzner estate.

`tailscale_acl` is the odd one out: a **tailnet-global, frequently-edited, low-risk**
config bolted onto an **estate-specific, rarely-edited, high-risk** state. Consequences of
the coupling, all observed:

1. **Every ACL edit is a full-estate apply.** Adding one port grant (e.g. `:8443` for
   Langfuse, 2026-07-28) requires the gated `infra-apply.yml` that reconciles the whole
   Hetzner estate. The `allow_destructive`-empty guard blocks destructive plans, but a
   one-line policy change still runs through the heaviest, highest-blast-radius apply path.
2. **Single-owner contortions.** Because the ACL is tailnet-wide, a *second* state can't
   also manage it. The `manage_tailscale_acl` flag exists **solely** to let the DR-drill
   workspace set it `false` so it "cannot fight prod for the same tailnet ACL"
   (`variables.tf:118`, `terraform.drill.ci.tfvars`). That flag is a workaround for a
   coupling that shouldn't exist.
3. **Provider-quirk plumbing.** The `tailscale_acl` resource always reports
   "will be updated in-place" even when before == after, so `infra-drift.yml` carries a
   bespoke filter (`select(.address != "tailscale_acl.main[0]")`) to avoid holding the
   drift issue permanently open. `infra/recover.sh` carries a special-case
   `tofu import 'tailscale_acl.main[0]' acl` for apply conflicts.

The ACL is **policy config**, not infrastructure. Terraform is the wrong lifecycle for it.

## Decision (D1)

**Move `tailscale_acl` out of OpenTofu entirely and manage `tailscale/policy.hujson` with
Tailscale's native GitOps ACL action** (`tailscale/gitops-acl-action`):

- **On PR** touching `tailscale/policy.hujson` → the action runs in **test** mode
  (validate + dry-run diff, posted for review). Replaces the ACL portion of the
  `infra-ci.yml` plan comment.
- **On merge to `main`** (or a dedicated `workflow_dispatch`) → the action runs in
  **apply** mode, syncing the file to the tailnet. A fast, ACL-only sync — no Hetzner in
  the plan, no full-estate apply.

`tailscale_tailnet_key.prod` **stays in tofu** (it is genuinely prod-specific — the key
the VPS uses to join the tailnet). Only the tailnet-wide `tailscale_acl` leaves.

`policy.hujson` remains the single source of truth and the review gate (RFC-082 Decision 2's
"ACL changes ship as PRs" is preserved — only the *mechanism* that applies them changes,
from `tofu apply` to the GitOps action).

## Consequences

**Positive**

- ACL changes stop triggering a full-estate Hetzner apply. Blast radius of a port grant
  drops from "whole prod estate" to "the ACL only."
- Hetzner `tofu apply` only ever touches Hetzner — plans are smaller, reviews are clearer,
  and the drift check no longer needs the `tailscale_acl` quirk filter.
- Removes the `manage_tailscale_acl` flag, the drill-workspace ACL suppression, and the
  `recover.sh` ACL-import special case — net simpler.
- Idiomatic: Tailscale's first-class GitOps tooling owns tailnet policy; tofu owns cloud
  infra. Clean separation of concerns.

**Negative**

- A **state migration** (`terraform state rm tailscale_acl.main[0]`) is required — a
  destructive state operation that must be done carefully (state backup first; the *live*
  ACL is untouched, only tofu's tracking of it is removed). See the migration plan.
- One new mechanism to understand (the GitOps action) instead of "everything is tofu."
- The apply moves from the human-gated `infra-apply.yml` (typed confirm + `environment:
  prod`) to the GitOps action. The action must be gated equivalently (branch protection on
  `main` for the source; and either apply-on-merge or a dispatch-gated apply) so an ACL
  change still can't reach the tailnet without review.

**Neutral**

- `policy.hujson` path, format, and PR-review workflow are unchanged for authors.
- DR-drill: with the ACL out of tofu, the drill workspace no longer needs
  `manage_tailscale_acl=false`; the drill simply never runs the GitOps apply.

## Alternatives considered

- **A — Separate OpenTofu state for `tailscale_acl`.** Move the resource to its own tofu
  root + state + a lightweight ACL-only plan/apply workflow. Decouples the blast radius
  and keeps everything in tofu (familiar). Rejected as the primary because it still uses
  the wrong lifecycle for policy config, keeps the provider-quirk plumbing, and adds a
  second state to manage — the GitOps action removes the ACL from tofu's problem space
  entirely. Kept as the fallback if the GitOps action proves unsuitable.
- **B — Status quo + rely on `allow_destructive`.** The existing guards do prevent a
  destructive Hetzner change from riding along an ACL apply. Rejected: it doesn't address
  the core mismatch (full-estate apply for a policy tweak) and keeps all the plumbing.
- **C — OAuth vs API-key for the GitOps action.** The action supports both. Prefer a
  scoped OAuth client (acl:write) over the broad `TS_API_KEY`; decide at implementation.

## Migration

State migration + workflow cutover steps live in
[`docs/wip/2026-07-28-tailscale-acl-tofu-decouple-migration.md`](../wip/2026-07-28-tailscale-acl-tofu-decouple-migration.md).
No state is touched until that plan is reviewed and approved (rule 4).
