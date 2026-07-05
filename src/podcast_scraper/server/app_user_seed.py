"""Seed a fixed roster of platform users at startup (#1128; dev/local).

So a freshly-started local platform always has known users — to sign in as and to see in the admin
user-management surface — this hooks a small fixed roster into the **mock** OAuth identity space.
Each seed is keyed by a ``hint``; the derived identity matches a ``?as=<hint>`` mock login exactly
(``subject=e2e-<hint>``, ``email=<hint>@e2e.local``), so the seeded user *is* the user you get by
signing in with that hint.

Idempotent and non-destructive: a user that already exists (e.g. whose role an admin changed at
runtime) is left untouched, so seeding never clobbers live state on restart. Opt-in via
``APP_SEED_USERS_FILE`` — unset in prod, so this never runs there.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from podcast_scraper.server import app_roles
from podcast_scraper.server.app_oauth import _safe_hint
from podcast_scraper.server.app_user_store import create_user, get_user, user_id_for

logger = logging.getLogger(__name__)


def _mock_identity(hint: str) -> tuple[str, str, str]:
    """``(provider, subject, email)`` matching a ``?as=<hint>`` mock login (same ``_safe_hint``)."""
    safe = _safe_hint(hint)
    return "mock", f"e2e-{safe}", f"{safe}@e2e.local"


def seed_users(data_dir: Path, seeds: list[dict]) -> int:
    """Ensure each seed user exists (create-if-absent). Returns the number created."""
    created = 0
    for seed in seeds:
        if not isinstance(seed, dict):
            continue
        hint = _safe_hint(str(seed.get("hint", "")))
        if not hint:
            continue
        provider, subject, email = _mock_identity(hint)
        if get_user(data_dir, user_id_for(provider, subject)) is not None:
            continue
        create_user(
            data_dir,
            provider=provider,
            subject=subject,
            email=email,
            name=str(seed.get("name") or hint),
            role=app_roles.normalize_role(seed.get("role")),
        )
        created += 1
    return created


def seeds_from_env() -> list[dict]:
    """Load the seed roster (a JSON list) from ``APP_SEED_USERS_FILE``; ``[]`` if unset/invalid."""
    raw = os.environ.get("APP_SEED_USERS_FILE", "").strip()
    if not raw:
        return []
    path = Path(raw).expanduser()
    if not path.is_file():
        logger.warning("APP_SEED_USERS_FILE is set but not a file: %s", path)
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read APP_SEED_USERS_FILE %s: %s", path, exc)
        return []
    return [s for s in doc if isinstance(s, dict)] if isinstance(doc, list) else []


def seed_from_env(data_dir: Path | None) -> int:
    """Seed from ``APP_SEED_USERS_FILE`` into ``data_dir`` (no-op when unconfigured)."""
    if data_dir is None:
        return 0
    seeds = seeds_from_env()
    if not seeds:
        return 0
    n = seed_users(data_dir, seeds)
    if n:
        logger.info("Seeded %d platform user(s) from APP_SEED_USERS_FILE", n)
    return n
