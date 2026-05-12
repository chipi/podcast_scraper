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
hcloud_environment_label = "drill"
location                 = "hel1"
server_type              = "cpx32"
