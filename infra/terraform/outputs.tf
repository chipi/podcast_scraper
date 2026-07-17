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

output "ssh_break_glass_deploy_ipv4" {
  description = <<-EOT
    Non-Tailscale SSH over the server's public IPv4 (requires hcloud_inbound_ssh_troubleshoot_cidrs
    allowing your source, e.g. drill CI tfvars). Use the private key matching OPERATOR_SSH_PUBLIC_KEY
    / TF_VAR_ssh_public_key. First-boot cloud-init also installs that key for root@ on this IPv4.
  EOT
  value       = format("ssh deploy@%s", hcloud_server.prod.ipv4_address)
}

output "volume_id" {
  description = "Attached Volume ID, or null if no Volume was provisioned."
  value       = var.volume_size_gb > 0 ? hcloud_volume.corpus[0].id : null
}

# #1199 audio archive — feed these into the rclone remote config on prod/laptop:
#   RCLONE_CONFIG_<NAME>_HOST = audio_storage_box_server
#   RCLONE_CONFIG_<NAME>_USER = audio_storage_box_username
#   RCLONE_CONFIG_<NAME>_PASS = rclone obscure <the audio_storage_box_password>
output "audio_storage_box_server" {
  description = "Storage Box FQDN for rclone SFTP host, or null if not provisioned."
  value       = var.audio_storage_box_type != "" ? hcloud_storage_box.audio_archive[0].server : null
}

output "audio_storage_box_username" {
  description = "Storage Box primary username for rclone SFTP, or null if not provisioned."
  value       = var.audio_storage_box_type != "" ? hcloud_storage_box.audio_archive[0].username : null
}
