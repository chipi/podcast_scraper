variable "hcloud_token" {
  type        = string
  description = "Hetzner Cloud API token (env: HCLOUD_TOKEN). Read+Write scope, project-scoped to podcast-scraper-prod."
  sensitive   = true
}

variable "tailscale_oauth_client_id" {
  type        = string
  description = "Tailscale OAuth client ID for the gha-deployer tag."
  sensitive   = true
}

variable "tailscale_oauth_client_secret" {
  type        = string
  description = "Tailscale OAuth client secret."
  sensitive   = true
}

variable "tailscale_tailnet" {
  type        = string
  description = "Tailscale tailnet name (e.g. 'example.com' or 'tail-xxxxx.ts.net')."
}

variable "server_type" {
  type        = string
  description = "Hetzner Cloud server type. CX33 (interim, amd64, EUR 6.49) or CAX31 (preferred, ARM, EUR 15.99, post-#712 multi-arch publish)."
  default     = "cx33"

  validation {
    condition     = contains(["cx33", "cax31", "ccx13", "ccx23"], var.server_type)
    error_message = "Supported types: cx33, cax31 (preferred once #712 lands), ccx13, ccx23. See RFC-082 Decision 1."
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
