# Migration plan — move `tailscale_acl` from OpenTofu → Tailscale GitOps

Companion to [ADR-128](../adr/ADR-128-decouple-tailnet-acl-from-hetzner-tofu.md). Concrete,
ordered, reversible steps. **Nothing here runs until the ADR is accepted + this plan is
approved.** The one destructive step is `terraform state rm` (rule 4 — per-instance approval,
state backed up first). The *live* tailnet ACL is never at risk: `state rm` only stops tofu
from tracking it; `policy.hujson` stays the source of truth throughout.

## Guiding invariant
At every step, exactly **one** mechanism owns the live tailnet ACL, and it always applies
the current `tailscale/policy.hujson`. We add the new owner (GitOps action) and prove it
works **before** removing the old owner (tofu) from state.

---

## Phase 0 — Prep (no changes to prod)
1. **Scoped credential.** Create a Tailscale **OAuth client** with `acl` write scope
   (preferred over the broad `TS_API_KEY`). Store as GH secret `TS_ACL_OAUTH_CLIENT_ID` /
   `TS_ACL_OAUTH_SECRET`. (Fallback: reuse `TS_API_KEY` if OAuth is deferred.)
2. **Confirm the action + policy format.** `tailscale/gitops-acl-action` consumes a HuJSON
   policy file. Verify `tailscale/policy.hujson` validates against the action's `test` mode
   locally / in a scratch branch.

## Phase 1 — Add the new owner in TEST-only mode (safe, additive)
3. **New workflow `.github/workflows/tailscale-acl.yml`:**
   - `on: pull_request` (paths: `tailscale/policy.hujson`) → action `test` (validate +
     dry-run diff comment). **No apply yet.**
   - `on: workflow_dispatch` → action `test` (manual check).
   - Do **not** add the `apply`/merge trigger yet.
4. **Open a no-op PR** touching `policy.hujson` (whitespace/comment) to confirm the `test`
   run posts a clean diff. At this point tofu still owns the apply; the action only tests.

## Phase 2 — Cut over the apply (the ownership handoff)
> Do Phase 2 as its own PR, reviewed, so the handoff is a single reviewable diff.

5. **Back up tofu state** (out-of-band, before any `state rm`):
   ```sh
   # decrypt + copy the current encrypted state artifact aside, timestamped
   cp infra/terraform/terraform.tfstate.enc \
      infra/terraform/terraform.tfstate.enc.pre-acl-decouple-$(date -u +%Y%m%d)
   ```
   Keep this until Phase 3 is verified stable.
6. **`terraform state rm 'tailscale_acl.main[0]'`** on the prod state (the destructive step,
   rule 4). Removes tofu's tracking only — the live ACL is unchanged. Re-encrypt + persist
   state via the normal path. **Verify:** next `tofu plan` shows **no** `tailscale_acl`
   resource and no attempt to destroy the live ACL.
7. **Remove the tofu resource + its plumbing** (same PR):
   - `infra/terraform/tailscale.tf` — delete `resource "tailscale_acl" "main"` + the
     `moved` block.
   - `infra/terraform/variables.tf` — delete `manage_tailscale_acl`.
   - `infra/terraform/terraform.drill.ci.tfvars` + `terraform.drill.tfvars.example` —
     drop the `manage_tailscale_acl = false` line.
   - `infra/recover.sh` — remove the `tofu import 'tailscale_acl.main[0]' acl` special case.
   - `.github/workflows/infra-drift.yml` — remove the `tailscale_acl` quirk filter
     (`select(.address != "tailscale_acl.main[0]")`) — now moot.
   - `.github/workflows/infra-ci.yml` — its tofu plan no longer includes the ACL; keep it
     for the Hetzner resources. (The ACL diff now comes from the new `tailscale-acl.yml`.)
8. **Enable the apply trigger** on `tailscale-acl.yml`: on merge to `main` (paths:
   `policy.hujson`) → action `apply`. Gate it equivalently to today (branch protection on
   `main` is the review gate; if you want a human click, use `workflow_dispatch` + typed
   confirm for apply instead of auto-on-merge — decide in the PR).

## Phase 3 — Verify the new path end-to-end
9. **Prove GitOps apply works:** merge a *real* trivial ACL change (e.g. re-assert an
   existing grant) and confirm the action's `apply` run syncs it (Tailscale admin console
   shows the update; a `tailscale serve`/`nc` reachability check for an affected grant).
10. **Prove tofu is clean:** `infra-apply` (or `infra-drift`) plan shows **only Hetzner**
    resources, no `tailscale_acl`, no drift.
11. **Docs:** flip ADR-128 to *Accepted*; amend RFC-082 Decision 2 (ACL applies via the
    GitOps action, not tofu); update `infra/README.md` (drop the drill `manage_tailscale_acl`
    note) + `PROD_RUNBOOK.md` (ACL-change procedure → the new workflow).

## Rollback (at any phase, <10 min)
- **Before Phase 2 (state rm):** just delete `tailscale-acl.yml` — nothing else changed.
- **After Phase 2, if the GitOps apply misbehaves:** re-add the tofu resource +
  `tofu import 'tailscale_acl.main[0]' acl` (re-adopts the live ACL into state — this is
  exactly what `recover.sh` did), and disable the `tailscale-acl.yml` apply trigger. The
  live ACL is never lost because `policy.hujson` is unchanged throughout.

## Risk register
| Risk | Mitigation |
|---|---|
| `state rm` corrupts/loses state | Timestamped state backup (step 5) before the op; import path re-adopts (rollback) |
| Two owners briefly (tofu + action) both apply | Phase 1 is TEST-only; the apply trigger is enabled (step 8) only *after* `state rm` (step 6) removed tofu's ownership |
| GitOps action can't reach tailnet / bad creds | Phase 1 `test` mode proves creds before any apply; OAuth scoped to `acl` only |
| ACL change slips in without review | Apply trigger gated on `main` (branch-protected) or `workflow_dispatch` + typed confirm |
