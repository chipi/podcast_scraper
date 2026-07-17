#!/usr/bin/env bash
# apply-edge.sh — converge a LIVE box to the ADR-114 edge + #1160 hardening +
# ADR-117 o11y state, WITHOUT a rebuild (GOAL-1 go-live plan Phase 1.1, Option A).
#
# WHY this exists: cloud-init (infra/cloud-init/prod.user-data) is first-boot-only
# and `hcloud_server` ignores user_data drift, so editing the bootstrap does NOT
# change a running box. This script applies the SAME edge/hardening/o11y steps
# imperatively and idempotently to the live host over the tailnet — the
# tailscale-serve precedent (see prod.user-data "Security hardening" note).
#
# SCOPE (mirrors the edge/hardening/o11y sections of prod.user-data ONLY):
#   - SSH key-only hardening drop-in
#   - fail2ban: sshd jail + Caddy-access jail (T-11)
#   - metadata-egress SSRF guard (T-07 / review H8 — folds in Phase 3.3)
#   - Caddy shared edge engine + base Caddyfile (ADR-114) incl. ADR-118 CF real-IP
#   - Grafana Alloy host metrics + security logs -> Grafana Cloud (ADR-117)
#
# EXPLICITLY OUT OF SCOPE (do NOT let this script do these):
#   - Opening the firewall (80/443) — that is Phase 4, Terraform, the exposure
#     moment. This script leaves the box tailnet-only.
#   - docker / tailscale / repo checkout / corpus mount / the app stack — those
#     are already present on a bootstrapped box; this script asserts, never
#     reinstalls, and never touches the corpus or `podcast-scraper.service`.
#   - Secrets cutover (ADR-115) — Phase 3.4, operator-owned.
#
# The inline snippets below are kept BYTE-IDENTICAL to prod.user-data. If you
# change one, change both (there is no shared source for the small heredocs; the
# Caddyfile IS shared — this script copies it from the repo checkout).
#
# Usage:
#   sudo ./apply-edge.sh [--dry-run] [--repo-dir DIR] [--with-alloy]
#   DRY_RUN=1 sudo -E ./apply-edge.sh
#
#   --dry-run       Print what would change; make no changes. (or DRY_RUN=1)
#   --repo-dir DIR  Repo checkout to source the Caddyfile from (default /srv/podcast-scraper).
#   --with-alloy    Force Alloy enable even without creds present (config still
#                   needs /etc/alloy/grafana-cloud.env — normally set in Phase 2.1).
#
# Idempotent: safe to re-run. Each step converges only on drift.
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
REPO_DIR="${REPO_DIR:-/srv/podcast-scraper}"
WITH_ALLOY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --repo-dir) REPO_DIR="${2:?--repo-dir needs a path}"; shift ;;
    --with-alloy) WITH_ALLOY=1 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

# --- output helpers ----------------------------------------------------------
_c() { printf '\033[%sm%s\033[0m' "$1" "$2"; }
info()   { echo "$(_c '1;34' '[edge]') $*"; }
change() { echo "$(_c '1;33' '[edge]') $*"; }
ok()     { echo "$(_c '1;32' '[edge]') $*"; }
skip()   { echo "$(_c '2'    '[edge]') $*  (already converged)"; }
warn()   { echo "$(_c '1;31' '[edge]') $*" >&2; }

# run a command, or just print it under --dry-run
run() {
  if [ "$DRY_RUN" = 1 ]; then echo "  would run: $*"; else "$@"; fi
}

# Write $2 (stdin) to file $1 with mode $3, owner $4 — only if content differs.
# Dry-run aware. Returns 0 on write, 1 on no-op (so callers can chain reloads).
write_file() {
  local path="$1" mode="$2" owner="$3" new
  new="$(cat)"
  if [ -f "$path" ] && [ "$(cat "$path")" = "$new" ]; then
    skip "  $path"
    return 1
  fi
  change "  write $path (mode $mode, owner $owner)"
  if [ "$DRY_RUN" = 1 ]; then return 0; fi
  install -d -m 0755 "$(dirname "$path")"
  printf '%s\n' "$new" >"$path"
  chmod "$mode" "$path"
  chown "$owner" "$path"
  return 0
}

need_root() {
  if [ "$(id -u)" -ne 0 ]; then
    warn "must run as root (sudo). Re-run: sudo $0 $*"
    exit 1
  fi
}

ensure_pkg() {
  local pkg="$1"
  if dpkg -s "$pkg" >/dev/null 2>&1; then
    skip "  package $pkg"
  else
    change "  apt-get install $pkg"
    run apt-get install -y "$pkg"
  fi
}

enable_now() {
  local unit="$1"
  if systemctl is-enabled --quiet "$unit" 2>/dev/null && systemctl is-active --quiet "$unit" 2>/dev/null; then
    skip "  systemctl enable --now $unit"
  else
    change "  systemctl enable --now $unit"
    run systemctl enable --now "$unit"
  fi
}

# --- preflight ---------------------------------------------------------------
need_root "$@"
CADDYFILE_SRC="$REPO_DIR/infra/cloud-init/Caddyfile"
[ -f "$CADDYFILE_SRC" ] || { warn "Caddyfile not found at $CADDYFILE_SRC (wrong --repo-dir?)"; exit 1; }

if [ "$DRY_RUN" = 1 ]; then
  info "DRY-RUN — no changes will be made."
fi
info "repo-dir=$REPO_DIR  caddyfile=$CADDYFILE_SRC"
info "firewall is NOT touched — box stays tailnet-only after this run."

# =============================================================================
# 1. SSH hardening — key-only auth (prod.user-data sshd_config.d/99-hardening)
# =============================================================================
info "1) SSH hardening drop-in"
if write_file /etc/ssh/sshd_config.d/99-hardening.conf 0644 root:root <<'EOF'
PermitRootLogin prohibit-password
PasswordAuthentication no
KbdInteractiveAuthentication no
EOF
then
  # Never reload a config that fails validation (prod.user-data precedent).
  if [ "$DRY_RUN" = 1 ]; then
    echo "  would run: sshd -t && systemctl reload ssh"
  elif sshd -t; then
    run systemctl reload ssh
    ok "  sshd validated + reloaded"
  else
    warn "  sshd -t FAILED — not reloading ssh (drop-in left in place for review)"
  fi
fi

# =============================================================================
# 2. fail2ban — sshd jail + Caddy-access jail (T-11)
# =============================================================================
info "2) fail2ban jails"
ensure_pkg fail2ban

write_file /etc/fail2ban/jail.d/sshd.local 0644 root:root <<'EOF' || true
[sshd]
enabled = true
backend = systemd
maxretry = 5
findtime = 10m
bantime = 1h
EOF

# Keys on client_ip (ADR-118), NOT remote_ip — see prod.user-data rationale.
write_file /etc/fail2ban/filter.d/caddy-access.conf 0644 root:root <<'EOF' || true
[Definition]
failregex = "client_ip":"<HOST>".*"status":4\d\d
ignoreregex =
EOF

write_file /etc/fail2ban/jail.d/caddy.local 0644 root:root <<'EOF' || true
[caddy-access]
enabled = true
filter = caddy-access
logpath = /var/log/caddy/access.log
maxretry = 30
findtime = 2m
bantime = 1h
EOF

enable_now fail2ban

# =============================================================================
# 3. Metadata-egress SSRF guard (T-07 / review H8 — Phase 3.3 folded in here)
#    Blocks container egress to 169.254.169.254 (cloud creds + tailscale key).
# =============================================================================
info "3) metadata-egress SSRF guard"
write_file /usr/local/sbin/block-metadata-egress.sh 0755 root:root <<'EOF' || true
#!/usr/bin/env bash
set -euo pipefail
if ! iptables -C DOCKER-USER -d 169.254.169.254 -j DROP 2>/dev/null; then
  iptables -I DOCKER-USER -d 169.254.169.254 -j DROP
fi
EOF

unit_written=0
write_file /etc/systemd/system/block-metadata-egress.service 0644 root:root <<'EOF' && unit_written=1 || true
[Unit]
Description=Block container egress to cloud metadata IP (SSRF guard)
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/block-metadata-egress.sh

[Install]
WantedBy=multi-user.target
EOF
[ "$unit_written" = 1 ] && run systemctl daemon-reload
enable_now block-metadata-egress.service
# Re-assert the rule now (the unit is oneshot; a re-run of the script should
# converge the live iptables state too, not just the unit).
if [ "$DRY_RUN" = 1 ]; then
  echo "  would run: /usr/local/sbin/block-metadata-egress.sh"
else
  run /usr/local/sbin/block-metadata-egress.sh
  if iptables -C DOCKER-USER -d 169.254.169.254 -j DROP 2>/dev/null; then
    ok "  DOCKER-USER DROP 169.254.169.254 active"
  else
    warn "  DOCKER-USER rule NOT present (is docker up? DOCKER-USER chain exists only after dockerd starts)"
  fi
fi

# =============================================================================
# 4. Caddy shared edge engine (ADR-114) + base Caddyfile (incl. ADR-118 CF real-IP)
# =============================================================================
info "4) Caddy edge engine"
if command -v caddy >/dev/null 2>&1; then
  skip "  caddy installed"
else
  change "  install caddy (Cloudsmith stable apt repo)"
  run apt-get install -y debian-keyring debian-archive-keyring apt-transport-https
  if [ "$DRY_RUN" != 1 ]; then
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
    apt-get update
  fi
  run apt-get install -y caddy
fi

# Per-tenant vhost dir (deploy-owned) + access-log dir (caddy-owned).
run install -d -o deploy -g deploy -m 0755 /etc/caddy/sites
run install -d -o caddy -g caddy -m 0755 /var/log/caddy

# Shared reload grant (any tenant deploy reloads the engine after dropping its vhost).
write_file /etc/sudoers.d/99-caddy-reload 0440 root:root <<'EOF' || true
deploy ALL=(root) NOPASSWD: /usr/bin/systemctl reload caddy
EOF

# Drop the base Caddyfile from the repo (single source of truth).
caddy_changed=0
if [ -f /etc/caddy/Caddyfile ] && cmp -s "$CADDYFILE_SRC" /etc/caddy/Caddyfile; then
  skip "  /etc/caddy/Caddyfile"
else
  change "  cp $CADDYFILE_SRC -> /etc/caddy/Caddyfile"
  run cp "$CADDYFILE_SRC" /etc/caddy/Caddyfile
  caddy_changed=1
fi

# Validate BEFORE (re)starting — never run the engine on a bad config.
if [ "$DRY_RUN" = 1 ]; then
  echo "  would run: caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile"
elif caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile; then
  ok "  Caddyfile valid"
  if systemctl is-active --quiet caddy; then
    [ "$caddy_changed" = 1 ] && { change "  reload caddy"; run systemctl reload caddy; } || skip "  caddy running"
  else
    enable_now caddy
  fi
else
  warn "  Caddyfile INVALID — engine NOT (re)started; live config unchanged"
fi

# T-11: attach the caddy-access jail now that /var/log/caddy exists.
info "5) re-arm fail2ban (attach caddy jail)"
run systemctl reload-or-restart fail2ban

# =============================================================================
# 6. Grafana Alloy — host metrics + security logs -> Grafana Cloud (ADR-117)
#    Config is staged unconditionally; the service is only enabled when the
#    Grafana Cloud creds are present (normally set in Phase 2.1).
# =============================================================================
info "6) Grafana Alloy (o11y)"
if command -v alloy >/dev/null 2>&1; then
  skip "  alloy installed"
else
  change "  install alloy (apt.grafana.com)"
  if [ "$DRY_RUN" != 1 ]; then
    curl -fsSL https://apt.grafana.com/gpg.key | gpg --dearmor -o /etc/apt/keyrings/grafana.gpg
    chmod a+r /etc/apt/keyrings/grafana.gpg
    echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" > /etc/apt/sources.list.d/grafana.list
    apt-get update
  fi
  run apt-get install -y alloy
fi

alloy_changed=0
write_file /etc/alloy/config.alloy 0640 root:alloy <<'EOF' && alloy_changed=1 || true
prometheus.exporter.unix "host" {}

prometheus.scrape "host_metrics" {
  targets    = prometheus.exporter.unix.host.targets
  forward_to = [prometheus.remote_write.grafana_cloud.receiver]
}

prometheus.remote_write "grafana_cloud" {
  endpoint {
    url = sys.env("GRAFANA_CLOUD_REMOTE_WRITE_URL")
    basic_auth {
      username = sys.env("GRAFANA_CLOUD_PROM_USER")
      password = sys.env("GRAFANA_CLOUD_API_KEY")
    }
  }
}

// Security logs -> Loki (T-11 / ADR-117, tenant=common): sshd + fail2ban from the
// journal, the Caddy access log from file. Feeds the common security alert rules.
loki.relabel "journal" {
  forward_to = []
  rule {
    source_labels = ["__journal__systemd_unit"]
    target_label  = "unit"
  }
}

loki.source.journal "security" {
  forward_to    = [loki.process.tenant_common.receiver]
  relabel_rules = loki.relabel.journal.rules
  labels        = { job = "systemd-journal" }
}

loki.source.file "caddy" {
  targets    = [{ __path__ = "/var/log/caddy/access.log", job = "caddy" }]
  forward_to = [loki.process.tenant_common.receiver]
}

loki.process "tenant_common" {
  forward_to = [loki.write.grafana_cloud.receiver]
  stage.static_labels {
    values = { tenant = "common" }
  }
}

loki.write "grafana_cloud" {
  endpoint {
    url = sys.env("GRAFANA_CLOUD_LOKI_URL")
    basic_auth {
      username = sys.env("GRAFANA_CLOUD_LOKI_USER")
      password = sys.env("GRAFANA_CLOUD_API_KEY")
    }
  }
}
EOF

write_file /etc/systemd/system/alloy.service.d/grafana-cloud-env.conf 0644 root:root <<'EOF' && alloy_changed=1 || true
[Service]
EnvironmentFile=/etc/alloy/grafana-cloud.env
EOF

# Let Alloy read the journal (sshd/fail2ban) + the Caddy access log.
run usermod -aG systemd-journal alloy || true
run usermod -aG caddy alloy 2>/dev/null || true
[ "$alloy_changed" = 1 ] && run systemctl daemon-reload

if [ -s /etc/alloy/grafana-cloud.env ] || [ "$WITH_ALLOY" = 1 ]; then
  enable_now alloy
else
  warn "  /etc/alloy/grafana-cloud.env absent/empty — Alloy config staged but NOT started."
  warn "  Set the 5 GRAFANA_CLOUD_* values there (Phase 2.1) then: systemctl enable --now alloy"
fi

# =============================================================================
# 7. Verify + summary
# =============================================================================
info "7) verification"
if [ "$DRY_RUN" = 1 ]; then
  ok "dry-run complete — re-run without --dry-run to converge."
  exit 0
fi

fail=0
for unit in fail2ban block-metadata-egress.service caddy; do
  if systemctl is-active --quiet "$unit"; then ok "  active: $unit"; else warn "  NOT active: $unit"; fail=1; fi
done
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null 2>&1 \
  && ok "  caddy config valid" || { warn "  caddy config INVALID"; fail=1; }
fail2ban-client status >/dev/null 2>&1 && ok "  fail2ban responding" || warn "  fail2ban-client not responding"
iptables -C DOCKER-USER -d 169.254.169.254 -j DROP 2>/dev/null \
  && ok "  metadata-egress DROP active" || warn "  metadata-egress DROP NOT active (docker up?)"

echo
if [ "$fail" = 0 ]; then
  ok "edge converged. Firewall still CLOSED — box remains tailnet-only (Phase 4 opens :443)."
else
  warn "edge converged with warnings above — review before proceeding to Phase 4."
  exit 1
fi
