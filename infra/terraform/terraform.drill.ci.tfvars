# Non-secret drill defaults for GitHub Actions (workspace `drill`).
# Tokens come from Actions secrets (`HCLOUD_TOKEN_DRILL`, `TS_API_KEY`).
# Keep in sync with terraform.drill.tfvars.example.
#
# **hel1** + **cpx32** by default: cheaper shared types are often unavailable in
# CI; cpx32 placement is usually faster than burning retries on cx43/cax31/cx33.
# ``drill-infra-apply.yml`` still retries **nbg1** then **fsn1** with the same
# type on ``resource_unavailable``.

manage_tailscale_acl     = false
tailscale_advertise_tags = ["tag:dr-drill"]
tailnet_hostname         = "dr-podcast"
# Drill servers are ephemeral — disable Hetzner delete/rebuild protection so the
# orchestrator's teardown (`tofu destroy`) can actually remove them. PROD keeps
# protection on (var default true). Without this, destroy hits "server deletion is
# protected" and orphans the drill server (observed 2026-06-17, run 27681998910).
enable_delete_protection = false
hcloud_environment_label = "drill"
# Break-glass: public SSH while Tailscale join is broken or slow (narrow in a private tfvars).
hcloud_inbound_ssh_troubleshoot_cidrs = ["0.0.0.0/0", "::/0"]
location                              = "hel1"
server_type                           = "cpx32"
