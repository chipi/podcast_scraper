"""Tests for scripts/ops/resolve_tailnet_domain.sh.

The script exists so ops scripts can build tailnet URLs without hardcoding the operator's tailnet
name (`identifier-denylist`, CONTRIBUTING.md § "No operator identifiers in the repo"). A
`<TAILNET>` placeholder would satisfy the gate and break the scripts, so the domain is derived
instead — and the derivation is what these tests pin.

The rung order is the contract, not an implementation detail: an explicit override must beat the
prod variable, and the prod variable must beat whatever the local machine happens to think, or a
developer running the prod health check from their laptop would silently probe their own tailnet.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "ops" / "resolve_tailnet_domain.sh"

pytestmark = pytest.mark.unit


def _run(env: dict[str, str], *, clear: bool = True) -> subprocess.CompletedProcess[str]:
    assert SCRIPT.is_file(), f"missing {SCRIPT}"
    base = {k: v for k, v in os.environ.items() if k not in {"TAILNET_DOMAIN", "PROD_TAILNET_FQDN"}}
    full_env = {**base, **env} if clear else {**os.environ, **env}
    return subprocess.run(
        ["/usr/bin/env", "bash", str(SCRIPT)],
        cwd=str(REPO_ROOT),
        env=full_env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_explicit_override_wins() -> None:
    r = _run({"TAILNET_DOMAIN": "example.ts.net"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "example.ts.net"


def test_override_beats_the_prod_fqdn() -> None:
    """Rung order matters: the escape hatch must be reachable even on prod."""
    r = _run(
        {"TAILNET_DOMAIN": "override.ts.net", "PROD_TAILNET_FQDN": "prod-podcast.other.ts.net"}
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "override.ts.net"


def test_prod_fqdn_supplies_the_domain() -> None:
    """The prod path adds no new configuration — prod-ops-health.yml already injects this var."""
    r = _run({"PROD_TAILNET_FQDN": "prod-podcast.example.ts.net"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "example.ts.net"


def test_prod_fqdn_keeps_multi_label_domains_intact() -> None:
    """Only the FIRST label is the host; everything after it is the domain."""
    r = _run({"PROD_TAILNET_FQDN": "prod-podcast.sub.example.ts.net"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "sub.example.ts.net"


def test_trailing_and_leading_dots_are_normalised() -> None:
    """MagicDNS reports names with a trailing dot; a URL built from one 404s confusingly."""
    r = _run({"TAILNET_DOMAIN": "example.ts.net."})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "example.ts.net"


def test_a_bare_hostname_in_the_prod_var_is_not_treated_as_a_domain() -> None:
    """`prod-podcast` with no dot has no domain part; it must not resolve to itself."""
    r = _run({"PROD_TAILNET_FQDN": "prod-podcast"})
    # Falls through to local derivation. On a tailnet-connected dev box that succeeds; in CI it
    # does not. Either is fine — what must NOT happen is echoing the bare hostname back.
    assert r.stdout.strip() != "prod-podcast"


def test_emits_a_single_clean_line() -> None:
    """Callers do `d=$(resolve_tailnet_domain)` and interpolate — stray output corrupts the URL."""
    r = _run({"TAILNET_DOMAIN": "example.ts.net"})
    assert r.stdout == "example.ts.net\n"


def test_failure_is_loud_and_actionable(tmp_path: Path) -> None:
    """No silent empty string: an empty domain builds `https://vlogs./…`, which fails obscurely.

    Both lower rungs are pointed at empty fixtures. An earlier version of this test blanked PATH
    instead, which also broke `/usr/bin/env bash` — the script never ran and the assertion passed
    judgement on exit 127. That is why the seams exist.
    """
    empty_status = tmp_path / "status.json"
    empty_status.write_text("{}", encoding="utf-8")
    empty_resolv = tmp_path / "resolv.conf"
    empty_resolv.write_text("nameserver 1.1.1.1\nsearch localdomain\n", encoding="utf-8")

    r = _run(
        {
            "TAILSCALE_STATUS_JSON_PATH": str(empty_status),
            "RESOLV_CONF_PATH": str(empty_resolv),
        }
    )
    assert r.returncode == 1, f"expected a hard failure, got {r.returncode}: {r.stdout!r}"
    assert r.stdout.strip() == "", "a failed resolution must print nothing to stdout"
    assert "TAILNET_DOMAIN" in r.stderr, r.stderr


def test_a_status_fixture_supplies_the_domain(tmp_path: Path) -> None:
    """Rung 3 reads MagicDNSSuffix — the authoritative answer when a real CLI is present."""
    status = tmp_path / "status.json"
    status.write_text('{"MagicDNSSuffix": "fixture.ts.net"}', encoding="utf-8")
    r = _run({"TAILSCALE_STATUS_JSON_PATH": str(status)})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "fixture.ts.net"


def test_resolv_conf_is_the_zero_config_dev_rung(tmp_path: Path) -> None:
    """The rung that makes dev work on EITHER machine with no setup and no tailscale binary."""
    status = tmp_path / "status.json"
    status.write_text("{}", encoding="utf-8")
    resolv = tmp_path / "resolv.conf"
    resolv.write_text("search fixture.ts.net localdomain\nnameserver 1.1.1.1\n", encoding="utf-8")
    r = _run({"TAILSCALE_STATUS_JSON_PATH": str(status), "RESOLV_CONF_PATH": str(resolv)})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "fixture.ts.net"


def test_the_script_itself_carries_no_tailnet_literal() -> None:
    """The whole point: the fix must not reintroduce what it removed."""
    import re

    text = SCRIPT.read_text(encoding="utf-8")
    leaked = re.findall(r"\b[a-z0-9-]+\.tail[a-z0-9]{5,}\.ts\.net\b", text)
    assert not leaked, f"a real tailnet FQDN is back in the resolver: {leaked}"
