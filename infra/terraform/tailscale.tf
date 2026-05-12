provider "tailscale" {
  # Free-plan workaround: OAuth clients are gated to Tailscale Premium+ tiers.
  # On Personal Free we authenticate the provider with a Personal API access
  # token instead. See PROD_RUNBOOK.md "Tailscale credentials" + RFC-082 §Decision 2.
  api_key = var.tailscale_api_key
  tailnet = var.tailscale_tailnet
}

# Per-server auth key. Rotated on every `tofu apply` so a leaked key only
# affects the window between issuance and re-apply (typically minutes).
# Prod: 1h expiry, single-use. Drill: 4h + reusable so cloud-init can retry
# ``tailscale up`` without burning the key on transient boot networking.
resource "tailscale_tailnet_key" "prod" {
  # Drill: reusable keys survive a transient ``tailscale up`` failure (cloud-init
  # can retry) without burning a one-shot key. Prod stays single-use.
  reusable      = var.hcloud_environment_label == "drill"
  ephemeral     = false
  preauthorized = true
  expiry        = var.hcloud_environment_label == "drill" ? 14400 : 3600
  tags          = var.tailscale_advertise_tags
  # Tailscale API rejects some punctuation in key descriptions (400). Keep
  # colons out of the description string even though tag values use "tag:name".
  description = format(
    "podcast-scraper-auth-key-%s",
    replace(join("-", var.tailscale_advertise_tags), ":", "-")
  )
}

# Sync the repo's ACL file to the tailnet. Source of truth = tailscale/policy.hujson;
# every `tofu apply` overwrites the live policy. ACL changes ship as PRs per
# RFC-082 Decision 2 + Decisions-made #4.
#
# When manage_tailscale_acl is false (e.g. OpenTofu workspace "drill"), omit this
# resource so a second state cannot fight prod for the same tailnet ACL (#752).
resource "tailscale_acl" "main" {
  count = var.manage_tailscale_acl ? 1 : 0
  acl   = file("${path.module}/../../tailscale/policy.hujson")
}

# State upgrade: ACL resource gained `count`; prod workspace rewrites address
# tailscale_acl.main -> tailscale_acl.main[0] on first plan after this change.
moved {
  from = tailscale_acl.main
  to   = tailscale_acl.main[0]
}
