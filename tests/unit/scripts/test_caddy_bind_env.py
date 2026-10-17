"""The Caddy public-bind line: written quoted, read back in both forms.

Caddy binds ``default_bind {$CADDY_BIND_ADDRS}``, the box's public IPv4 and IPv6. Written
unquoted (``Environment=CADDY_BIND_ADDRS=<v4> <v6>``), systemd keeps only the first address, and
on 2026-10-17 prod's Caddy listened on IPv4 alone. The two writers (cloud-init, apply-edge.sh) now
quote the value; the two readers (deploy_config_caddy.sh, verify-edge.sh) parse it back for
``caddy adapt``/``validate``, and must keep reading a box that still has the old line.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
OPS = REPO_ROOT / "scripts" / "ops"
USER_DATA = REPO_ROOT / "infra" / "cloud-init" / "prod.user-data"
DROP_IN = "/etc/systemd/system/caddy.service.d/10-public-bind.conf"
ADDRS = "178.105.182.17 2a01:4f8:1c1b:8cd4::1"

pytestmark = pytest.mark.unit


def _bash(script: str) -> str:
    return subprocess.run(
        ["/usr/bin/env", "bash", "-c", script], capture_output=True, text=True, check=True
    ).stdout


def _cloud_init_drop_in() -> str:
    line = next(
        ln.strip() for ln in USER_DATA.read_text().splitlines() if "10-public-bind.conf" in ln
    )
    return _bash(f'BIND="{ADDRS}"; {line.replace(DROP_IN, "/dev/stdout")}')


def _apply_edge_drop_in() -> str:
    text = (OPS / "apply-edge.sh").read_text()
    body = re.search(r"10-public-bind\.conf 0644 root:root <<EOF\n(.*?)\nEOF", text, re.S)
    assert body, "apply-edge.sh's bind heredoc"
    return _bash(f'BIND_ADDRS="{ADDRS}"; cat <<EOF\n{body.group(1)}\nEOF')


def _reader(script: str, var: str) -> str:
    line = next(
        ln.strip()
        for ln in (OPS / script).read_text().splitlines()
        if ln.strip().startswith(f"{var}=") and "CADDY_BIND_ADDRS" in ln
    )
    return line


@pytest.mark.parametrize("writer", [_cloud_init_drop_in, _apply_edge_drop_in])
def test_both_writers_quote_the_two_addresses(writer) -> None:  # type: ignore[no-untyped-def]
    assert writer().splitlines() == ["[Service]", f'Environment="CADDY_BIND_ADDRS={ADDRS}"']


@pytest.mark.parametrize(
    "script,var", [("deploy_config_caddy.sh", "BIND"), ("verify-edge.sh", "BIND_ENV")]
)
@pytest.mark.parametrize(
    "drop_in",
    [
        f'[Service]\nEnvironment="CADDY_BIND_ADDRS={ADDRS}"\n',
        f"[Service]\nEnvironment=CADDY_BIND_ADDRS={ADDRS}\n",
    ],
    ids=["quoted", "unquoted-old-box"],
)
def test_both_readers_get_both_addresses_from_either_form(
    tmp_path: Path, script: str, var: str, drop_in: str
) -> None:
    conf = tmp_path / "10-public-bind.conf"
    conf.write_text(drop_in)
    line = _reader(script, var).replace(DROP_IN, str(conf))
    assert _bash(f'{line}\nprintf "%s" "${var}"') == ADDRS
