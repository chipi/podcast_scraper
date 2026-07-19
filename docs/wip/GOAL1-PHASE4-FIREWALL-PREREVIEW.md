# Phase 4.1 — firewall open: static pre-review

**Goal-1 go-live, Phase 4.1.** Static source analysis of what `tofu apply` will do
when it opens `:80`/`:443` — done **before** running the live `tofu plan`, so the
live plan is a rubber-stamp against the expectations below.

**Scope + honesty:** this is a read of `infra/terraform/main.tf` at the current
commit. I did **not** run `tofu plan` (needs the prod state + Hetzner creds and is
an infra action). The live plan MUST still be run and diffed against this note
before 4.2. If the live plan disagrees with any expectation here, **stop**.

## What Phase 4.2 changes

`hcloud_firewall.main` gains two inbound TCP rules (already in the repo, not yet
applied to the live firewall):

| Port | Source | Purpose |
|---|---|---|
| `80` | `0.0.0.0/0, ::/0` | ACME HTTP-01 + HTTP→HTTPS redirect (no app content) |
| `443` | `0.0.0.0/0, ::/0` **when `cloudflare_origin_lock = false`** (default) | the public edge (Caddy TLS) |

This is **the exposure moment** — after apply, the box is world-reachable on
`:80`/`:443`. Everything else stays tailnet-only.

## Expected `tofu plan` — and the abort condition

**MUST be:** `~ hcloud_firewall.main` (in-place rule add). Firewall rules are a
mutable attribute; the resource's `id` does not change.

**MUST NOT be:** any `-/+` (destroy-then-create) or `# forces replacement` on
**`hcloud_server.prod`**. A server replace would destroy the box. → **ABORT.**

Why the server is safe (from the source, not a guess):
- `hcloud_server.prod` references the firewall by stable id:
  `firewall_ids = [hcloud_firewall.main.id]` — updating the firewall's *rules*
  doesn't change its `id`, so this attribute is unchanged.
- `lifecycle { ignore_changes = [user_data, ssh_keys] }` — the cloud-init
  edits (edge/hardening) do **not** show as drift and cannot force a replace.
- No `server_type` / `image` / `location` change is part of this step.

So the server should show **no change at all**; only the firewall shows `~`.

## Preconditions to check at apply time

- **Workspace = prod**, not `drill`. (The drill workspace sets
  `enable_delete_protection = false`; prod keeps delete + rebuild protection ON.)
- **`cloudflare_origin_lock`** — default `false` ⇒ `:443` world-open (correct for
  Phase 4; Cloudflare fronting is Phase 6). Do not flip it here.
- **`hcloud_inbound_ssh_troubleshoot_cidrs`** — should be **empty** so no `:22`
  rule is added (SSH stays tailnet-only). If non-empty, confirm it's an
  intentional, time-boxed troubleshoot CIDR before applying.
- Phase 3.2 signed off (`No firewall opens until 3.2 is signed off`).

## Rollback

Remove the two rules and `tofu apply` again (<2 min) — the firewall drops back to
tailnet-only. No server touch, no data risk.
