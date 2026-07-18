#!/usr/bin/env bash
# verify-edge.sh — READ-ONLY verification that a box is converged to the ADR-114
# edge + #1160 hardening + ADR-117 o11y state (GOAL-1 go-live plan Phase 1.3).
#
# Companion to apply-edge.sh: apply-edge *converges*, verify-edge *checks*. This
# script NEVER changes anything — it asserts the live state and exits non-zero if
# any critical edge/hardening component is missing, so it is safe to run any time
# and usable as a go/no-go gate before Phase 2 (o11y) / Phase 4 (open firewall).
#
# Run it over the tailnet on the box (as root — several checks need iptables /
# fail2ban-client / sshd -T). Non-root degrades those to SKIP with a note.
#
# What it verifies (mirrors prod.user-data edge/hardening/o11y sections):
#   1. SSH hardening effective (key-only)
#   2. fail2ban engine + sshd jail + caddy-access jail (T-11)
#   3. metadata-egress SSRF DROP present + unit enabled (T-07 / review H8)
#   4. Caddy config valid, engine active, listening on :443
#   5. /etc/caddy/Caddyfile matches the repo source (drift)
#   6. Grafana Alloy active (or staged, pre-Phase-2)
#
# It does NOT (and cannot from the box) prove external reachability — the public
# firewall is Hetzner-cloud-side (Phase 4 / TF), not on the box. It reports the
# LOCAL listen state and reminds you to confirm :443 from outside AFTER Phase 4.2.
#
# Usage:  sudo ./verify-edge.sh [--repo-dir DIR]
set -euo pipefail

REPO_DIR="${REPO_DIR:-/srv/podcast-scraper}"
while [ $# -gt 0 ]; do
  case "$1" in
    --repo-dir) REPO_DIR="${2:?--repo-dir needs a path}"; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

_c() { printf '\033[%sm%s\033[0m' "$1" "$2"; }
PASS=0 FAILC=0 WARNC=0
pass() { echo "$(_c '1;32' ' PASS') $*"; PASS=$((PASS+1)); }
fail() { echo "$(_c '1;31' ' FAIL') $*"; FAILC=$((FAILC+1)); }
warn() { echo "$(_c '1;33' ' WARN') $*"; WARNC=$((WARNC+1)); }
skip() { echo "$(_c '2'    ' SKIP') $*"; }
section() { echo; echo "$(_c '1;34' "== $* ==")"; }

IS_ROOT=0; [ "$(id -u)" -eq 0 ] && IS_ROOT=1
[ "$IS_ROOT" = 1 ] || warn "not root — privileged checks (iptables/fail2ban/sshd -T) will SKIP; re-run with sudo for a full gate"

# --- 1. SSH hardening --------------------------------------------------------
section "1) SSH hardening (key-only)"
if [ "$IS_ROOT" = 1 ] && command -v sshd >/dev/null 2>&1; then
  eff="$(sshd -T 2>/dev/null || true)"
  echo "$eff" | grep -qi '^passwordauthentication no'        && pass "PasswordAuthentication no" || fail "PasswordAuthentication not disabled"
  echo "$eff" | grep -qi '^kbdinteractiveauthentication no'  && pass "KbdInteractiveAuthentication no" || warn "KbdInteractiveAuthentication not disabled"
  # sshd -T normalizes ``prohibit-password`` to its older alias ``without-password``
  # — both mean key-only root (no password). Accept either, else we false-WARN.
  echo "$eff" | grep -qiE '^permitrootlogin (prohibit-password|without-password)' && pass "PermitRootLogin key-only (prohibit/without-password)" || warn "PermitRootLogin allows password login"
else
  skip "sshd -T (needs root)"
fi

# --- 2. fail2ban -------------------------------------------------------------
section "2) fail2ban jails (T-11)"
if systemctl is-active --quiet fail2ban; then
  pass "fail2ban service active"
  if [ "$IS_ROOT" = 1 ]; then
    jails="$(fail2ban-client status 2>/dev/null | sed -n 's/.*Jail list:\s*//p')"
    echo "$jails" | grep -q 'sshd'         && pass "sshd jail present"         || fail "sshd jail missing"
    echo "$jails" | grep -q 'caddy-access' && pass "caddy-access jail present" || fail "caddy-access jail missing (Caddy log dir must exist + fail2ban reloaded)"
  else
    skip "fail2ban-client status (needs root)"
  fi
else
  fail "fail2ban service NOT active"
fi

# --- 3. metadata-egress SSRF guard ------------------------------------------
section "3) metadata-egress SSRF guard (T-07 / H8)"
systemctl is-enabled --quiet block-metadata-egress.service 2>/dev/null \
  && pass "block-metadata-egress.service enabled" || fail "block-metadata-egress.service not enabled"
if [ "$IS_ROOT" = 1 ]; then
  if iptables -C DOCKER-USER -d 169.254.169.254 -j DROP 2>/dev/null; then
    pass "DOCKER-USER DROP 169.254.169.254 active"
  else
    fail "DOCKER-USER DROP missing (docker up? unit run? — container SSRF-to-metadata is OPEN)"
  fi
else
  skip "iptables -C DOCKER-USER (needs root)"
fi

# --- 4. Caddy edge -----------------------------------------------------------
section "4) Caddy edge engine (ADR-114)"
if command -v caddy >/dev/null 2>&1; then
  caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null 2>&1 \
    && pass "Caddyfile valid" || fail "Caddyfile INVALID"
  systemctl is-active --quiet caddy && pass "caddy service active" || fail "caddy service NOT active"
  if command -v ss >/dev/null 2>&1; then
    ss -tlnH 'sport = :443' 2>/dev/null | grep -q . \
      && pass "listening on :443 (tailnet-only until Phase 4 opens the firewall)" \
      || warn ":443 not listening (no tenant vhost yet? base engine still binds :443 catch-all)"
  else
    skip "ss :443 check (ss not found)"
  fi
else
  fail "caddy not installed"
fi

# --- 5. Caddyfile drift vs repo ---------------------------------------------
section "5) Caddyfile drift"
SRC="$REPO_DIR/infra/cloud-init/Caddyfile"
if [ -f "$SRC" ] && [ -f /etc/caddy/Caddyfile ]; then
  cmp -s "$SRC" /etc/caddy/Caddyfile && pass "/etc/caddy/Caddyfile matches repo" \
    || warn "/etc/caddy/Caddyfile DIFFERS from $SRC (re-run apply-edge.sh to converge)"
else
  skip "drift check (missing $SRC or /etc/caddy/Caddyfile)"
fi

# --- 6. Grafana Alloy --------------------------------------------------------
section "6) Grafana Alloy (ADR-117)"
if command -v alloy >/dev/null 2>&1; then
  [ -f /etc/alloy/config.alloy ] && pass "config.alloy staged" || fail "config.alloy missing"
  if systemctl is-active --quiet alloy; then
    pass "alloy service active (shipping to Grafana Cloud)"
  elif [ -s /etc/alloy/grafana-cloud.env ]; then
    warn "grafana-cloud.env present but alloy not active — start it: systemctl enable --now alloy"
  else
    skip "alloy staged, not started — expected until Phase 2.1 sets GRAFANA_CLOUD_* creds"
  fi
else
  warn "alloy not installed (run apply-edge.sh, or defer o11y to Phase 2)"
fi

# --- summary -----------------------------------------------------------------
section "summary"
echo "  $(_c '1;32' "$PASS pass")  $(_c '1;33' "$WARNC warn")  $(_c '1;31' "$FAILC fail")"
echo "  Reminder: external :443 reachability is governed by the Hetzner firewall"
echo "  (Phase 4). Confirm from OUTSIDE only AFTER 4.2 — the box cannot prove it."
if [ "$FAILC" -gt 0 ]; then
  echo "$(_c '1;31' 'edge NOT fully converged') — address FAILs before Phase 2/4."
  exit 1
fi
echo "$(_c '1;32' 'edge converged') — tailnet-only, ready for the Phase 1.3 sign-off."
