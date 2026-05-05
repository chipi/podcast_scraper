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
    tailscale_auth_key = tailscale_tailnet_key.prod.key
    tailnet_hostname   = var.tailnet_hostname
    ssh_public_key     = var.ssh_public_key

    alloy_enabled = local.alloy_host_metrics_enabled

    grafana_cloud_metrics_remote_write_url = var.grafana_cloud_metrics_remote_write_url
    grafana_cloud_metrics_username         = var.grafana_cloud_metrics_username
    grafana_cloud_metrics_password         = var.grafana_cloud_metrics_password
  })

  labels = {
    project     = "podcast-scraper"
    environment = "prod"
    managed-by  = "opentofu"
  }

  # Cloud-init only runs once. Re-applying with a rotated auth key would force
  # server replacement (loses corpus, tailnet identity); ignore the diff so
  # routine `tofu apply` for non-server resources stays cheap.
  lifecycle {
    ignore_changes = [user_data]
  }
}

# === Attach optional Volume ===
resource "hcloud_volume_attachment" "corpus" {
  count     = var.volume_size_gb > 0 ? 1 : 0
  volume_id = hcloud_volume.corpus[0].id
  server_id = hcloud_server.prod.id
  automount = true
}
