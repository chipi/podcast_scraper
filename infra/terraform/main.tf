provider "hcloud" {
  token = var.hcloud_token
}

locals {
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

  # Public edge (ADR-114 / #1158): the shared Caddy reverse proxy terminates TLS
  # on :443 and redirects :80 -> https. This is the first world-facing ingress on
  # this box — everything else stays tailnet-only. Adding these is an in-place
  # ``~`` update of the firewall resource; it MUST NOT show a server ``-/+``
  # replace in ``tofu plan`` (verify before apply). Reversible: remove + apply.
  #
  # :80 stays world-open even under the Cloudflare origin lock — Let's Encrypt
  # HTTP-01 challenges validate from LE's own IPs (not through CF), and it only
  # ever serves the ACME challenge + an HTTP->HTTPS redirect, no app content.
  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "80"
    source_ips = ["0.0.0.0/0", "::/0"]
  }

  # T-05 / ADR-118: when cloudflare_origin_lock is true, :443 is reachable only
  # from Cloudflare's ranges — no direct-IP bypass of CF's WAF/rate-limit/DDoS.
  # Default (false) keeps :443 world-open so the edge works before CF is fronted.
  rule {
    direction  = "in"
    protocol   = "tcp"
    port       = "443"
    source_ips = var.cloudflare_origin_lock ? var.cloudflare_ip_ranges : ["0.0.0.0/0", "::/0"]
  }
}

# === Optional detached Volume for corpus bind-mount ===
resource "hcloud_volume" "corpus" {
  count    = var.volume_size_gb > 0 ? 1 : 0
  name     = "podcast-scraper-corpus"
  size     = var.volume_size_gb
  location = var.location
  format   = "ext4"

  # Hetzner-API-level delete guard. Same shape as hcloud_server.prod's
  # delete_protection — Hetzner refuses to delete the volume until this
  # is flipped to false in a prior apply. Corpus is the most expensive
  # thing to recreate (snapshots are weekly at best); worth the
  # two-step ratchet to destroy. PROD keeps this true (default); the drill
  # workspace sets enable_delete_protection=false so its teardown can destroy.
  delete_protection = var.enable_delete_protection

  # Never let a size change plan a destroy+create of the corpus volume (Hetzner
  # treats a size decrease as ForceNew). Resize is done out-of-band via
  # ``hcloud volume resize`` + filesystem expand (review 2026-07-17 low/tf-volume).
  # (``prevent_destroy`` can't be used — it must be a literal and would block the
  # drill workspace's own teardown.)
  lifecycle {
    ignore_changes = [size]
  }
}

# === The VPS itself ===
resource "hcloud_server" "prod" {
  name        = var.tailnet_hostname
  image       = "ubuntu-24.04"
  server_type = var.server_type
  location    = var.location

  # Hetzner-API-level guards (added 2026-05-30 post-incident). Any DELETE
  # or REBUILD request — whether from ``tofu destroy``, an apply that
  # plans a replacement, or a direct Hetzner API call — is rejected by
  # Hetzner with "protected: deletion/rebuild protection enabled" before
  # any destruction happens. To legitimately destroy the server (DR
  # drill, planned migration), first apply a change setting these to
  # false, then run the destroy / wipe-then-apply. PROD keeps these true
  # (var default); the drill workspace sets enable_delete_protection=false
  # so its automated teardown destroys instead of orphaning the server.
  delete_protection  = var.enable_delete_protection
  rebuild_protection = var.enable_delete_protection

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

    # Shell script body is injected via ``file()`` so ``$`` / ``((`` are never passed through
    # ``templatefile`` twice (avoids broken ``n=$$((n+1))`` on the VPS).
    podcast_tailscale_serve_body = indent(6, chomp(file("${path.module}/../cloud-init/podcast-tailscale-serve.sh")))
    orrery_tailscale_serve_body  = indent(6, chomp(file("${path.module}/../cloud-init/orrery-tailscale-serve.sh")))

    # ADR-115: shared secret-decrypt helper injected via file() (its ``$${..}``
    # shell syntax must not pass through templatefile twice — same reason as the
    # tailscale-serve bodies above).
    decrypt_secrets_body = indent(6, chomp(file("${path.module}/../cloud-init/decrypt-secrets.sh")))

    # ADR-114 / #1158: shared Caddy edge base Caddyfile injected via file() — its
    # Caddy ``{..}`` / ``{$..}`` syntax must not be interpreted by templatefile.
    caddy_base_body = indent(6, chomp(file("${path.module}/../cloud-init/Caddyfile")))
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
