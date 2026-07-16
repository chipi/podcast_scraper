variable "hcloud_token" {
  type        = string
  description = "Hetzner Cloud API token (env: HCLOUD_TOKEN). Read+Write scope, project-scoped to podcast-scraper-prod."
  sensitive   = true
}

variable "tailscale_api_key" {
  type        = string
  description = "Tailscale Personal API access token (Free-plan substitute for OAuth clients). Used by the tailscale Terraform provider to manage ACL + generate per-server auth keys. NOT to be confused with TS_AUTHKEY (the device-join auth key consumed by tailscale/github-action and cloud-init)."
  sensitive   = true
}

variable "tailscale_tailnet" {
  type        = string
  description = "Tailscale tailnet name (e.g. 'example.com' or 'tail-xxxxx.ts.net')."
}

variable "server_type" {
  type        = string
  description = "Hetzner Cloud server type. CX43 (chosen, 8 vCPU shared Intel/AMD, 16 GB, EUR 15.11/mo) per RFC-082 Decision 1. CAX31 (ARM Ampere, EUR 19.95/mo) is the anti-jitter premium swap; CCX23 dedicated AMD is the noisy-neighbor escalation. CPX32 (shared AMD, 4 vCPU / 8 GB) is allowed for drill CI when cost-optimized lines lack placement. Re-verify prices at hetzner.com/cloud before apply."
  default     = "cx43"

  validation {
    condition = contains(
      ["cx43", "cax31", "ccx23", "cx33", "ccx13", "cpx32"],
      var.server_type,
    )
    error_message = "Supported types: cx43 (chosen), cax31 (anti-jitter swap), ccx23 (dedicated escalation), cx33/ccx13 (smaller), cpx32 (drill placement fallback). See RFC-082 Decision 1."
  }
}

variable "location" {
  type        = string
  description = "Hetzner location: fsn1 (Falkenstein) or nbg1 (Nuremberg) for EU per RFC-082."
  # Default changed 2026-05-30 from fsn1 → nbg1: fsn1 ran out of CX43 capacity
  # during the 2026-05-29 wipe-then-apply rebuild, so prod was placed in nbg1
  # using ``override_location``. Pinning the default here so routine applies
  # don't try to "fix" location → server replacement. Migrate back to fsn1
  # later by overriding when capacity returns + planning the migration.
  default = "nbg1"

  validation {
    condition     = contains(["fsn1", "nbg1", "hel1"], var.location)
    error_message = "Supported EU locations: fsn1, nbg1, hel1."
  }
}

variable "ssh_public_key" {
  type        = string
  description = "Operator's SSH public key. Cloud-init bakes it into deploy@<host>:~/.ssh/authorized_keys."
  sensitive   = true
}

variable "additional_authorized_keys" {
  type        = list(string)
  description = <<-EOT
    Extra SSH public keys to append to deploy@<host>:~/.ssh/authorized_keys
    (alongside ssh_public_key). Each entry is a full one-line OpenSSH
    pubkey ("ssh-ed25519 AAAA... comment"). Used to grant cross-repo
    GitHub Actions deploy jobs SSH access without rotating the operator
    key. Public keys, not sensitive — safe to commit.
  EOT
  default = [
    # chipi/orrery deploy workflow (issue #834 / orrery #260).
    # Private half lives in chipi/orrery → Settings → Secrets → PROD_SSH_PRIVATE_KEY.
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDJ/59d9WL+5JeUTX73P9l3jgGnjhE43Hz5keSfw/hf3 gha-orrery-deploy",
  ]
}

variable "ssh_public_key_name" {
  type        = string
  description = "Name to register the SSH key under in Hetzner."
  default     = "operator-laptop"
}

variable "volume_size_gb" {
  type        = number
  description = "Optional detached Volume for corpus bind-mount. 0 = no Volume (boot disk used). RFC-082 'recommended once corpus grows past ~20 GB'."
  default     = 0

  validation {
    condition     = var.volume_size_gb == 0 || (var.volume_size_gb >= 10 && var.volume_size_gb <= 1000)
    error_message = "volume_size_gb must be 0 (disabled) or 10-1000."
  }
}

variable "tailnet_hostname" {
  type        = string
  description = "Hostname the VPS registers with Tailscale. Final URL: https://<tailnet_hostname>.<tailscale_tailnet>/."
  default     = "prod-podcast"
}

variable "manage_tailscale_acl" {
  type        = bool
  description = "When true (default), apply tailscale/policy.hujson via tailscale_acl. Set false for a secondary OpenTofu workspace (e.g. drill) so only one state owns the tailnet-wide ACL — see infra/README.md \"DR drill workspace\"."
  default     = true
}

variable "enable_delete_protection" {
  type        = bool
  description = "Hetzner delete/rebuild protection on the server + corpus volume. True (default) hardens PROD against accidental destroy/rebuild (post-2026-05-30 incident). Set false ONLY for the ephemeral drill workspace so its automated teardown (`tofu destroy`) can actually delete the drill server/volume instead of hitting \"server deletion is protected\" and orphaning infra — see infra/README.md \"DR drill workspace\"."
  default     = true
}

variable "tailscale_advertise_tags" {
  type        = list(string)
  description = "Tailscale tags for tailscale_tailnet_key and for `tailscale up --advertise-tags` in cloud-init. Prod uses [\"tag:prod\"]; drill uses e.g. [\"tag:dr-drill\"] per tailscale/policy.hujson tagOwners."
  default     = ["tag:prod"]

  validation {
    condition = alltrue([
      for t in var.tailscale_advertise_tags : can(regex("^tag:[a-z][a-z0-9_-]*$", t))
    ])
    error_message = "Each tailscale_advertise_tags entry must look like tag:name (lowercase start; letters, digits, hyphen, underscore after tag:)."
  }
}

variable "hcloud_environment_label" {
  type        = string
  description = "Value for hcloud_server label `environment` (Hetzner UI / API). Use `prod` for production; use `drill` for throwaway DR drill stacks in a separate Hetzner project."
  default     = "prod"

  validation {
    condition     = contains(["prod", "drill", "preprod"], var.hcloud_environment_label)
    error_message = "hcloud_environment_label must be prod, drill, or preprod."
  }
}

variable "hcloud_inbound_ssh_troubleshoot_cidrs" {
  type        = list(string)
  description = <<-EOT
    When non-empty, the Hetzner firewall also allows inbound TCP/22 from these CIDRs
    (in addition to UDP/41641 + ICMP). Default empty: SSH is intended only over Tailscale.
    Drill CI sets 0.0.0.0/0 and ::/0 so operators can reach deploy@ via the server's public
    IPv4/IPv6 while debugging Tailscale join or restore; narrow to your /32 in a private tfvars
    file if you want stricter hygiene on long-lived drill VMs.
  EOT
  default     = []
}

variable "grafana_cloud_metrics_remote_write_url" {
  type        = string
  description = "Grafana Cloud Prometheus remote_write URL (…/api/prom/push). If any of the three grafana_cloud_metrics_* vars are empty, cloud-init skips Alloy host metrics."
  default     = ""
}

variable "grafana_cloud_metrics_username" {
  type        = string
  description = "Grafana Cloud Prometheus basic-auth username (numeric instance / stack user id for remote_write)."
  default     = ""
}

variable "grafana_cloud_metrics_password" {
  type        = string
  description = "Grafana Cloud access policy token with metrics:write (or legacy API key) for remote_write basic-auth password. Reused for Loki logs:write (same token needs logs:write scope — see T-11 / ADR-117)."
  default     = ""
  sensitive   = true
}

variable "grafana_cloud_logs_url" {
  type        = string
  description = "Grafana Cloud Loki push URL (…/loki/api/v1/push) for the Alloy security-log pipeline (T-11 / ADR-117: sshd/fail2ban/Caddy). Empty = Alloy ships metrics only."
  default     = ""
}

variable "grafana_cloud_logs_username" {
  type        = string
  description = "Grafana Cloud Loki basic-auth username (numeric Loki instance id). Password reuses grafana_cloud_metrics_password (token needs logs:write)."
  default     = ""
}

variable "cloudflare_origin_lock" {
  type        = bool
  description = <<-EOT
    T-05 / ADR-118. When true, the Hetzner firewall restricts inbound :443 to
    Cloudflare's published ranges (cloudflare_ip_ranges) so attackers can't
    bypass CF by hitting the origin IP directly. :80 stays world-open for ACME
    HTTP-01 (Let's Encrypt validates from its own IPs, not through CF).
    Default false. FLIP TO TRUE ONLY AFTER DNS is orange-clouded and the site is
    verified loading through CF (ADR-118 rollout step 5) — flipping early with
    grey-cloud DNS makes :443 unreachable. Rollback: set false + apply.
  EOT
  default     = false
}

variable "cloudflare_ip_ranges" {
  type        = list(string)
  description = <<-EOT
    Cloudflare proxy IP ranges allowed to reach :443 when cloudflare_origin_lock
    is true. Refresh from https://www.cloudflare.com/ips/ (~yearly) and keep in
    sync with the `trusted_proxies` list in infra/cloud-init/Caddyfile.
  EOT
  default = [
    "173.245.48.0/20", "103.21.244.0/22", "103.22.200.0/22", "103.31.4.0/22",
    "141.101.64.0/18", "108.162.192.0/18", "190.93.240.0/20", "188.114.96.0/20",
    "197.234.240.0/22", "198.41.128.0/17", "162.158.0.0/15", "104.16.0.0/13",
    "104.24.0.0/14", "172.64.0.0/13", "131.0.72.0/22",
    "2400:cb00::/32", "2606:4700::/32", "2803:f800::/32", "2405:b500::/32",
    "2405:8100::/32", "2a06:98c0::/29", "2c0f:f248::/32",
  ]
}
