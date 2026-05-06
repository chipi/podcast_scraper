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
  description = "Hetzner Cloud server type. CX43 (chosen, 8 vCPU shared Intel/AMD, 16 GB, EUR 15.11/mo) per RFC-082 Decision 1. CAX31 (ARM Ampere, EUR 19.95/mo) is the anti-jitter premium swap; CCX23 dedicated AMD is the noisy-neighbor escalation. Re-verify prices at hetzner.com/cloud before apply."
  default     = "cx43"

  validation {
    condition     = contains(["cx43", "cax31", "ccx23", "cx33", "ccx13"], var.server_type)
    error_message = "Supported types: cx43 (chosen), cax31 (anti-jitter swap), ccx23 (dedicated escalation), cx33/ccx13 (smaller). See RFC-082 Decision 1."
  }
}

variable "location" {
  type        = string
  description = "Hetzner location: fsn1 (Falkenstein) or nbg1 (Nuremberg) for EU per RFC-082."
  default     = "fsn1"

  validation {
    condition     = contains(["fsn1", "nbg1", "hel1"], var.location)
    error_message = "Supported EU locations: fsn1, nbg1, hel1."
  }
}

variable "ssh_public_key" {
  type        = string
  description = "Operator's SSH public key. Cloud-init bakes it into deploy@<host>:~/.ssh/authorized_keys."
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
