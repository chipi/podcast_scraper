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
  default     = "nbg1"

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
  default     = "operator-laptop-drift-test"
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
  description = "Grafana Cloud access policy token with metrics:write (or legacy API key) for remote_write basic-auth password."
  default     = ""
  sensitive   = true
}
