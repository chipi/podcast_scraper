output "server_id" {
  description = "Hetzner Cloud server ID for the prod VPS."
  value       = hcloud_server.prod.id
}

output "ipv4_address" {
  description = "Public IPv4 (used for outbound only; no inbound services exposed publicly)."
  value       = hcloud_server.prod.ipv4_address
}

output "tailnet_url" {
  description = "Full HTTPS URL for the viewer over Tailscale MagicDNS."
  value       = "https://${var.tailnet_hostname}.${var.tailscale_tailnet}/"
}

output "ssh_target" {
  description = "SSH target for deploy-prod.yml (assumes the runner has joined the tailnet via OAuth)."
  value       = "deploy@${var.tailnet_hostname}.${var.tailscale_tailnet}"
}

output "volume_id" {
  description = "Attached Volume ID, or null if no Volume was provisioned."
  value       = var.volume_size_gb > 0 ? hcloud_volume.corpus[0].id : null
}
