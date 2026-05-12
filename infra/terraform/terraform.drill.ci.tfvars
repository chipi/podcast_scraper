# Non-secret drill defaults for GitHub Actions (workspace `drill`).
# Tokens come from Actions secrets (`HCLOUD_TOKEN_DRILL`, `TS_API_KEY`).
# Keep in sync with terraform.drill.tfvars.example.
#
# Default **hel1** for drill: Falkenstein (fsn1) placement often returns
# ``resource_unavailable`` in CI; apply retries still try nbg1 then fsn1 last.

manage_tailscale_acl     = false
tailscale_advertise_tags = ["tag:dr-drill"]
tailnet_hostname         = "dr-podcast"
hcloud_environment_label = "drill"
location                 = "hel1"
