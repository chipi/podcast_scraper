provider "hcloud" {
  token = var.hcloud_token
}

locals {
  # All three must be non-empty to bake Grafana Alloy + remote_write into cloud-init.
  alloy_host_metrics_enabled = (
    length(trimspace(var.grafana_cloud_metrics_remote_write_url)) > 0 &&
    length(trimspace(var.grafana_cloud_metrics_username)) > 0 &&
    length(trimspace(var.grafana_cloud_metrics_password)) > 0
  )

  # Comma-separated for `tailscale up --advertise-tags=tag:a,tag:b` (RFC-082 / #752 drill).
  tailscale_advertise_tags_cli = join(",", var.tailscale_advertise_tags)
}

# === SSH key registration (operator's laptop pubkey) ===
resource "hcloud_ssh_key" "operator" {
  name       = var.ssh_public_key_name
  public_key = var.ssh_public_key
}

# === Private network ===
resource "hcloud_network" "main" {
  name     = "podcast-scraper-prod"
  ip_range = "10.0.0.0/16"
}

resource "hcloud_network_subnet" "main" {
  network_id   = hcloud_network.main.id
  type         = "cloud"
  network_zone = "eu-central"
  ip_range     = "10.0.1.0/24"
}

# === Firewall: default-deny inbound except Tailscale + ICMP + outbound ===
# Tailnet traffic tunnels through UDP 41641; SSH and viewer/api are reached
# over the tailnet, NOT through the public firewall.
resource "hcloud_firewall" "main" {
  name = "podcast-scraper-prod"

  dynamic "rule" {
    for_each = length(var.hcloud_inbound_ssh_troubleshoot_cidrs) > 0 ? [1] : []
    content {
      direction  = "in"
      protocol   = "tcp"
      port       = "22"
      source_ips = var.hcloud_inbound_ssh_troubleshoot_cidrs
    }
  }

  rule {
    direction  = "in"
    protocol   = "udp"
    port       = "41641"
    source_ips = ["0.0.0.0/0", "::/0"]
  }

  rule {
    direction  = "in"
    protocol   = "icmp"
    source_ips = ["0.0.0.0/0", "::/0"]
  }
}

# === Optional detached Volume for corpus bind-mount ===
resource "hcloud_volume" "corpus" {
  count    = var.volume_size_gb > 0 ? 1 : 0
  name     = "podcast-scraper-corpus"
  size     = var.volume_size_gb
  location = var.location
  format   = "ext4"
}

# === The VPS itself ===
resource "hcloud_server" "prod" {
  name        = var.tailnet_hostname
  image       = "ubuntu-24.04"
  server_type = var.server_type
  location    = var.location

  ssh_keys     = [hcloud_ssh_key.operator.id]
  firewall_ids = [hcloud_firewall.main.id]

  network {
    network_id = hcloud_network.main.id
    ip         = "10.0.1.10"
  }

  user_data = templatefile("${path.module}/../cloud-init/prod.user-data", {
    tailscale_auth_key         = tailscale_tailnet_key.prod.key
    tailnet_hostname           = var.tailnet_hostname
    ssh_public_key             = var.ssh_public_key
    additional_authorized_keys = var.additional_authorized_keys

    tailscale_advertise_tags_cli = local.tailscale_advertise_tags_cli

    alloy_enabled = local.alloy_host_metrics_enabled

    grafana_cloud_metrics_remote_write_url = var.grafana_cloud_metrics_remote_write_url
    grafana_cloud_metrics_username         = var.grafana_cloud_metrics_username
    grafana_cloud_metrics_password         = var.grafana_cloud_metrics_password

    # Shell script body is injected via ``file()`` so ``$`` / ``((`` are never passed through
    # ``templatefile`` twice (avoids broken ``n=$$((n+1))`` on the VPS).
    podcast_tailscale_serve_body = indent(6, chomp(file("${path.module}/../cloud-init/podcast-tailscale-serve.sh")))
    orrery_tailscale_serve_body  = indent(6, chomp(file("${path.module}/../cloud-init/orrery-tailscale-serve.sh")))
  })

  labels = {
    project     = "podcast-scraper"
    environment = var.hcloud_environment_label
    managed-by  = "opentofu"
  }

  # Cloud-init only runs once. Re-applying with a rotated auth key would force
  # server replacement (loses corpus, tailnet identity); ignore the diff so
  # routine `tofu apply` for non-server resources stays cheap.
  #
  # ``ssh_keys`` added 2026-05-29 (#839) after a rotated ``OPERATOR_SSH_PUBLIC_KEY``
  # GH Secret cascaded ``hcloud_ssh_key.operator`` replacement into ``ssh_keys =
  # [...] # forces replacement`` on this resource, destroying prod. Tailscale's
  # serve config + the deploy@ authorized_keys are managed independently of
  # the Hetzner ssh_keys attribute, so ignoring drift here is safe and
  # eliminates the cascade.
  lifecycle {
    ignore_changes = [user_data, ssh_keys]
  }
}

# === Attach optional Volume ===
resource "hcloud_volume_attachment" "corpus" {
  count     = var.volume_size_gb > 0 ? 1 : 0
  volume_id = hcloud_volume.corpus[0].id
  server_id = hcloud_server.prod.id
  automount = true
}
