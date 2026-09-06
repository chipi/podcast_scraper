"""The rendered cloud-init ``user_data`` must fit Hetzner's 32,768-byte field (#2002).

Hetzner rejects an over-size ``user_data`` at ``tofu apply`` with::

    invalid input in field 'user_data' (invalid_input):
        [user_data => [Length must be between 0 and 32768.]]

That is the FIRST real step of the DR drill, so every downstream job — deploy, corpus
restore, e2e, playwright — reports ``skipped``. The drill failed that way on 2026-08-05,
08-12, 08-19, 08-26 and 09-02 before anyone noticed, because its only alert was gated on an
unset secret (#1999). Disaster recovery was unverified for five weeks.

The template is not the payload: ``main.tf`` renders it with ``templatefile()`` and injects
four shell/config bodies via ``file()``. Measuring ``prod.user-data`` on disk is therefore
misleading — it was 25 KB while the rendered payload was 34.5 KB. This test measures what
Hetzner would actually receive.

It asserts a MARGIN rather than the raw cap. Landing at 32,760 would pass a bare
``< 32768`` check and break on the next line anyone adds.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# Hetzner Cloud API, servers.user_data: "Length must be between 0 and 32768".
HETZNER_USER_DATA_MAX_BYTES = 32_768

# Fail while there is still room to fix it calmly, not at the moment apply breaks.
SAFE_FRACTION = 0.90

_CLOUD_INIT = Path(__file__).resolve().parents[3] / "infra" / "cloud-init"

# The four bodies main.tf injects with `indent(6, chomp(file(...)))`.
_INJECTED = {
    "podcast_tailscale_serve_body": "podcast-tailscale-serve.sh",
    "orrery_tailscale_serve_body": "orrery-tailscale-serve.sh",
    "decrypt_secrets_body": "decrypt-secrets.sh",
    "caddy_base_body": "Caddyfile",
}

# Realistic worst-case scalars. A tailscale key and two ed25519 keys are the only
# variable-length ones, and they are near-constant in size.
_SCALARS = {
    "tailscale_auth_key": "tskey-auth-" + "x" * 50,
    "tailnet_hostname": "prod-podcast",
    "ssh_public_key": "ssh-ed25519 " + "A" * 68 + " operator@ci",
    "additional_authorized_keys": "ssh-ed25519 " + "B" * 68 + " deploy@ci",
    "tailscale_advertise_tags_cli": "--advertise-tags=tag:prod-podcast",
}


def _indent6(body: str) -> str:
    """Terraform's ``indent(6, ...)`` — every line BUT the first gets the prefix."""
    parts = body.rstrip("\n").split("\n")
    return "\n".join([parts[0]] + [f"      {p}" if p.strip() else p for p in parts[1:]])


def _render() -> str:
    text = (_CLOUD_INIT / "prod.user-data").read_text(encoding="utf-8")
    for var, filename in _INJECTED.items():
        body = _indent6((_CLOUD_INIT / filename).read_text(encoding="utf-8"))
        text = text.replace("${" + var + "}", body)
    for var, value in _SCALARS.items():
        text = text.replace("${" + var + "}", value)
    # `%{ for k in ... }` expands per authorized key; one iteration is the realistic case.
    text = re.sub(r"%\{~?\s*for\s+\w+\s+in\s+[^}]+~?\}", "", text)
    text = re.sub(r"%\{~?\s*endfor\s*~?\}", "", text)
    return text.replace("${k}", _SCALARS["additional_authorized_keys"])


def test_rendered_user_data_fits_the_hetzner_field() -> None:
    size = len(_render().encode("utf-8"))
    assert size < HETZNER_USER_DATA_MAX_BYTES, (
        f"rendered user_data is {size:,} bytes, over Hetzner's "
        f"{HETZNER_USER_DATA_MAX_BYTES:,} cap by {size - HETZNER_USER_DATA_MAX_BYTES:,}. "
        "`tofu apply` will fail and the whole DR drill will skip. Move prose to "
        "prod.user-data.NOTES.md, or fetch a large body at boot instead of embedding it."
    )


def test_a_working_margin_is_kept() -> None:
    """Passing at 99% of the cap means the next added line breaks prod rebuilds."""
    size = len(_render().encode("utf-8"))
    budget = int(HETZNER_USER_DATA_MAX_BYTES * SAFE_FRACTION)
    assert size < budget, (
        f"rendered user_data is {size:,} bytes — under the hard cap but past the "
        f"{SAFE_FRACTION:.0%} working margin ({budget:,}). Trim it now, while the drill "
        "still passes, rather than after it breaks."
    )


def test_every_template_variable_is_accounted_for() -> None:
    """If main.tf gains an injected body, this test must learn about it or it under-measures."""
    leftovers = set(re.findall(r"\$\{([a-z_][a-z0-9_]*)\}", _render()))
    # `${dest}`/`${tenant}` are SHELL expansions inside embedded scripts, not template vars.
    shell_locals = {"dest", "tenant", "key", "k"}
    unknown = leftovers - shell_locals
    assert not unknown, (
        f"unsubstituted template variables {sorted(unknown)} — main.tf injects something this "
        "test does not, so the measured size is lower than what Hetzner receives. Add them "
        "to _INJECTED or _SCALARS."
    )
