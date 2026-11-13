"""Runtime persistence for the sign-in access policy (#2190).

Who may sign in used to be decided entirely by ``APP_ALLOWED_EMAILS`` / ``APP_ALLOWED_DOMAINS`` /
``APP_SIGNUP_MODE``, read once at startup. Those come from GitHub repo variables baked into
``.env.player`` by ``deploy-player.yml``, so **admitting one beta tester required a production
redeploy** — while every other property of a user (role, disabled, deletion, MCP access) was
already editable at runtime through the admin API. The thing that actually decides whether someone
can get in was the only thing that needed a deploy.

One ``access_policy`` row in the plane schema (ADR-169), written by an admin endpoint, **read per
request**, with the env policy as the fallback when there is no row. An instance that never
touches the endpoint behaves exactly as it did before, so the env var keeps working as the
bootstrap seed. A process with no database (the control plane) has no persisted policy, exactly
as it had no file.

The row wins over env when present. That is deliberate: a merge would make it impossible to ever
*remove* an address that came from env, and "revoke access" has to be achievable without a deploy
for the same reason "grant access" does. The lockout risk that creates is handled at the endpoint
by a self-lockout guard, mirroring the one in ``routes/app_admin.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from podcast_scraper.db import database_available, read_connection, read_with_retry, rw_engine
from podcast_scraper.db.schema import access_policy
from podcast_scraper.server.app_access import AccessPolicy

logger = logging.getLogger(__name__)

#: The file the policy was kept in before the database; the importer reads it.
_FILE_NAME = "access_policy.json"


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

    ``None`` means "fall back to the env policy" — it is NOT a policy that denies everyone. No
    row is the normal state for an instance that has never used the admin endpoint, and the only
    state of a process with no database. A database error is raised, never read as "no row": that
    would fall back to env and re-admit anyone revoked through the endpoint.
    """
    if data_dir is None or not database_available():
        return None

    def read(engine: Any) -> AccessPolicy | None:
        with read_connection(engine) as conn:
            row = conn.execute(select(access_policy)).mappings().first()
        if row is None:
            return None
        return policy_from_dict(
            {
                "mode": row["mode"],
                "allowed_emails": list(row["allowed_emails"]),
                "allowed_domains": list(row["allowed_domains"]),
            }
        )

    return read_with_retry(read)


def save_policy(data_dir: Path, policy: AccessPolicy) -> AccessPolicy:
    """Persist *policy* (the one row); returns it for convenience."""
    values = policy_to_dict(policy)
    with rw_engine().begin() as conn:
        conn.execute(
            pg_insert(access_policy)
            .values(id=1, **values)
            .on_conflict_do_update(
                index_elements=[access_policy.c.id], set_={**values, "updated_at": func.now()}
            )
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
