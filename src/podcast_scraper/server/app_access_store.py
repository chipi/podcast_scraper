"""Runtime persistence for the sign-in access policy (#2190).

Who may sign in used to be decided entirely by ``APP_ALLOWED_EMAILS`` / ``APP_ALLOWED_DOMAINS`` /
``APP_SIGNUP_MODE``, read once at startup. Those come from GitHub repo variables baked into
``.env.player`` by ``deploy-player.yml``, so **admitting one beta tester required a production
redeploy** — while every other property of a user (role, disabled, deletion, MCP access) was
already editable at runtime through the admin API. The thing that actually decides whether someone
can get in was the only thing that needed a deploy.

This is the same shape as its sibling :mod:`app_ranking_config_store`: one
JSON file in the data dir, written by an admin endpoint, **read per request**, with the env policy
as the fallback when the file is absent or unreadable. An instance that never touches the endpoint
behaves exactly as it did before, so the env var keeps working as the bootstrap seed.

The file wins over env when present. That is deliberate: a merge would make it impossible to ever
*remove* an address that came from env, and "revoke access" has to be achievable without a deploy
for the same reason "grant access" does. The lockout risk that creates is handled at the endpoint
by a self-lockout guard, mirroring the one in ``routes/app_admin.py``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.atomic_write import atomic_write_text

logger = logging.getLogger(__name__)

_FILE_NAME = "access_policy.json"


def _policy_path(data_dir: Path) -> Path:
    return data_dir / _FILE_NAME


def policy_to_dict(policy: AccessPolicy) -> dict[str, object]:
    """Serialisable form. Sets are sorted so the file is stable across writes."""
    return {
        "mode": policy.mode,
        "allowed_emails": sorted(policy.allowed_emails),
        "allowed_domains": sorted(policy.allowed_domains),
    }


def policy_from_dict(data: dict[str, object]) -> AccessPolicy:
    """Total parse: anything unrecognised degrades to the default-deny shape rather than raising.

    A malformed persisted policy must not be able to take the platform down, and it must not fail
    OPEN either — an unreadable mode becomes ``allowlist``, which denies rather than admits.
    """
    raw_mode = data.get("mode")
    mode = raw_mode.strip().lower() if isinstance(raw_mode, str) else "allowlist"
    if mode not in ("allowlist", "open"):
        mode = "allowlist"
    return AccessPolicy(
        mode=mode,
        allowed_emails=_clean(data.get("allowed_emails")),
        allowed_domains=_clean(data.get("allowed_domains")),
    )


def _clean(raw: object) -> frozenset[str]:
    """Lowercased, stripped, de-duplicated; non-strings and blanks dropped."""
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(item.strip().lower() for item in raw if isinstance(item, str) and item.strip())


def load_policy(data_dir: Path | None) -> AccessPolicy | None:
    """The persisted policy, or ``None`` when there isn't a usable one.

    ``None`` means "fall back to the env policy" — it is NOT a policy that denies everyone. An
    absent file is the normal state for an instance that has never used the admin endpoint.
    """
    if data_dir is None:
        return None
    path = _policy_path(Path(data_dir))
    if not path.is_file():
        return None  # the normal state for an instance that never used the endpoint — silent
    # A PRESENT but unreadable file is different, and it is loud.
    #
    # Falling back to env here re-admits anyone the operator revoked through the endpoint, because
    # env still lists them. That is fail-open relative to their last expressed intent, so it must
    # never happen quietly. It is still the right fallback rather than denying everyone: writes go
    # through `atomic_write_text`, so this file cannot be torn by us — a corrupt one means external
    # tampering or a hand-edit, and answering that by locking every admin out of the surface that
    # repairs it trades a stale allowlist for a self-inflicted outage with no way back in.
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.error(
            "access policy at %s is unreadable (%s) — FALLING BACK TO THE ENV POLICY, which may "
            "re-admit addresses revoked through the admin endpoint. Fix or delete the file.",
            path,
            exc,
        )
        return None
    if not isinstance(data, dict):
        logger.error(
            "access policy at %s is %s, expected an object — FALLING BACK TO THE ENV POLICY, "
            "which may re-admit revoked addresses. Fix or delete the file.",
            path,
            type(data).__name__,
        )
        return None
    return policy_from_dict(data)


def save_policy(data_dir: Path, policy: AccessPolicy) -> AccessPolicy:
    """Persist *policy* atomically; returns it for convenience."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        _policy_path(data_dir),
        json.dumps(policy_to_dict(policy), indent=2, ensure_ascii=False) + "\n",
    )
    return policy


def effective_policy(data_dir: Path | None, env_policy: AccessPolicy | None) -> AccessPolicy | None:
    """The policy to enforce: the persisted one when present, else the startup env one."""
    persisted = load_policy(data_dir)
    return persisted if persisted is not None else env_policy


__all__ = [
    "effective_policy",
    "load_policy",
    "policy_from_dict",
    "policy_to_dict",
    "save_policy",
]
