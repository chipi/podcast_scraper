provider "tailscale" {
  oauth_client_id     = var.tailscale_oauth_client_id
  oauth_client_secret = var.tailscale_oauth_client_secret
  tailnet             = var.tailscale_tailnet
}

# Per-server auth key. Rotated on every `tofu apply` so a leaked key only
# affects the window between issuance and re-apply (typically minutes).
# 1h expiry is enough for cloud-init's `tailscale up` on first boot.
resource "tailscale_tailnet_key" "prod" {
  reusable      = false
  ephemeral     = false
  preauthorized = true
  expiry        = 3600
  tags          = ["tag:prod"]
  description   = "podcast-scraper-prod VPS auth key (per-apply)"
}

# Sync the repo's ACL file to the tailnet. Source of truth = tailscale/policy.hujson;
# every `tofu apply` overwrites the live policy. ACL changes ship as PRs per
# RFC-082 Decision 2 + Decisions-made #4.
resource "tailscale_acl" "main" {
  acl = file("${path.module}/../../tailscale/policy.hujson")
}
