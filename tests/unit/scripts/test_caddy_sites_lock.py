"""Every podcast script that edits the shared Caddy edge holds the tenants' one lock.

``/etc/caddy/sites`` is shared by podcast (player, operator), orrery and the websites tenant. The
websites tenant deploys on every push to main and takes ``/tmp/caddy-sites.lock`` around its
install -> ``caddy adapt`` -> reload. A podcast deploy that edits the same dir without the lock can
interleave with it: one side's ``adapt`` sees the other's half-written vhost, or one side's
rollback reloads over the other's install. These tests pin, for each podcast writer, that the lock
is taken before the first edit and released only after the reload/restart.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK = "CADDY_SITES_LOCK=/tmp/caddy-sites.lock"
TAKE = "flock -w 300 9"

pytestmark = pytest.mark.unit

# script -> (first edge write, last caddy reload/restart of the edge section, release)
WRITERS = {
    "infra/deploy/deploy-player.sh": (
        '"infra/caddy/${v}.caddy" >"/etc/caddy/sites/${v}.caddy"',
        "_rollback_player_vhosts\n  sudo -n /usr/bin/systemctl restart caddy || true",
        "exec 9>&-",
    ),
    "infra/deploy/deploy-operator.sh": (
        '"infra/caddy/operator.caddy" >"/etc/caddy/sites/operator.caddy"',
        "_rollback_operator_vhost\n  sudo -n /usr/bin/systemctl restart caddy || true",
        "exec 9>&-",
    ),
    "scripts/ops/deploy_config_caddy.sh": (
        'cp "$new" "$cur"',
        "sudo -n /usr/bin/systemctl restart caddy || true",
        None,  # held until exit
    ),
    "scripts/ops/apply-edge.sh": (
        'run cp "$CADDYFILE_SRC" /etc/caddy/Caddyfile',
        "run systemctl reload caddy || run systemctl restart caddy",
        "exec 9<&-",
    ),
}


@pytest.mark.parametrize("rel", sorted(WRITERS))
def test_lock_taken_before_first_edge_write(rel: str) -> None:
    text = (REPO_ROOT / rel).read_text()
    first_write, _, _ = WRITERS[rel]
    assert text.count(LOCK) == 1, f"{rel}: expected exactly one {LOCK!r}"
    assert first_write in text, f"{rel}: edge write {first_write!r} moved; update this test"
    assert text.index(LOCK) < text.index(TAKE) < text.index(first_write)


@pytest.mark.parametrize("rel", sorted(WRITERS))
def test_lock_released_only_after_the_reload(rel: str) -> None:
    text = (REPO_ROOT / rel).read_text()
    _, last_reload, release = WRITERS[rel]
    assert last_reload in text, f"{rel}: reload {last_reload!r} moved; update this test"
    if release is None:
        assert "exec 9>&-" not in text and "exec 9<&-" not in text
        return
    assert text.count(release) == 1
    assert text.index(last_reload) < text.index(release)


def test_tenants_open_an_existing_lock_read_only() -> None:
    """fs.protected_regular=2 refuses an O_CREAT open of another user's file in sticky /tmp."""
    for rel in (
        "infra/deploy/deploy-player.sh",
        "infra/deploy/deploy-operator.sh",
        "scripts/ops/deploy_config_caddy.sh",
    ):
        text = (REPO_ROOT / rel).read_text()
        assert 'if [ -e "$CADDY_SITES_LOCK" ]; then exec 9<"$CADDY_SITES_LOCK"' in text, rel


def test_apply_edge_creates_the_lock_deploy_owned() -> None:
    """Root creates it for the tenants, which run as deploy and open it with ``>>`` when missing."""
    text = (REPO_ROOT / "scripts/ops/apply-edge.sh").read_text()
    assert 'install -o deploy -g deploy -m 0664 /dev/null "$CADDY_SITES_LOCK"' in text
    assert 'exec 9<"$CADDY_SITES_LOCK"' in text
