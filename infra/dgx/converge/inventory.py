"""pyinfra inventory for the DGX Spark (RFC-089).

Env vars (see infra/.env.dgx.local.example):

    DGX_TAILNET_FQDN     required — runtime HTTP target for assertions
                         (e.g. dgx-llm-1.tail6d0ed4.ts.net)
    DGX_SSH_HOST         optional — overrides DGX_TAILNET_FQDN for the SSH path
                         only (e.g. spark-2c14.local on LAN)
    DGX_SSH_USER         default: root  (non-root auto-escalates via sudo)
    DGX_SSH_KEY          optional — leave unset to use ssh-agent / ~/.ssh/config
    DGX_SSH_PORT         default: 22

The two-host-name split exists because runtime HTTP (:11434 / :8001) and SSH
(:22) may have different reachability — e.g. Tailscale SSH disabled while the
tailnet HTTP path is fine.
"""

from __future__ import annotations

import os

_fqdn = os.environ.get("DGX_TAILNET_FQDN")
if not _fqdn:
    raise SystemExit("DGX_TAILNET_FQDN unset. Source infra/.env.dgx.local or export it manually.")

_ssh_host = os.environ.get("DGX_SSH_HOST") or _fqdn
_user = os.environ.get("DGX_SSH_USER", "root")
_key = os.environ.get("DGX_SSH_KEY") or None
_port = int(os.environ.get("DGX_SSH_PORT", "22"))

_data: dict[str, object] = {
    "ssh_user": _user,
    "ssh_port": _port,
    # Re-surfaced for deploy.py to assert what the runtime tailnet host should be.
    "dgx_fqdn": _fqdn,
}
if _key:
    _data["ssh_key"] = _key

dgx = [(_ssh_host, _data)]
