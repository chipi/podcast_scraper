provider "tailscale" {
  # Free-plan workaround: OAuth clients are gated to Tailscale Premium+ tiers.
  # On Personal Free we authenticate the provider with a Personal API access
  # token instead. See PROD_RUNBOOK.md "Tailscale credentials" + RFC-082 §Decision 2.
  api_key = var.tailscale_api_key
  tailnet = var.tailscale_tailnet
}

# Per-server auth key. Rotated on every `tofu apply` so a leaked key only
# affects the window between issuance and re-apply (typically minutes).
# 1h expiry is enough for cloud-init's `tailscale up` on first boot.
resource "tailscale_tailnet_key" "prod" {
  reusable      = false
  ephemeral     = false
  preauthorized = true
  expiry        = 3600
  tags          = var.tailscale_advertise_tags
  # Tailscale API rejects some punctuation in key descriptions (400). Keep
  # colons out of the description string even though tag values use "tag:name".
  description = format(
    "podcast-scraper-auth-key-%s",
    replace(join("-", var.tailscale_advertise_tags), ":", "-")
  )
}

# NOTE (ADR-128, 2026-07-28): the tailnet ACL is no longer managed here. It moved out of
# OpenTofu to Tailscale's native GitOps action (.github/workflows/tailscale-acl.yml) so a
# one-line ACL grant stops forcing a full-estate apply. `tailscale/policy.hujson` remains
# the source of truth; it is now applied by the GitOps action, not `tofu apply`. The
# `tailscale_acl` resource was removed from state via `tofu state rm` before this deletion
# so tofu neither destroys nor recreates the live policy. Only `tailscale_tailnet_key.prod`
# (the VPS join key — genuinely prod-specific) stays in tofu.
