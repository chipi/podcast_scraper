#!/usr/bin/env python3
"""Empty the e2e database's ``users`` table so a local Playwright run starts as clean as CI.

The player e2e used to wipe ``e2e/.app-state`` (accounts were files there). Accounts are rows now
(ADR-169), so the same reset deletes them from the database ``PODCAST_DATABASE_URL`` names. Only
the plane's ``users`` table is touched.
"""

from __future__ import annotations

from sqlalchemy import create_engine, delete

from podcast_scraper.db.schema import users
from podcast_scraper.db.settings import DatabaseSettings


def main() -> int:
    settings = DatabaseSettings.from_env()
    engine = create_engine(
        settings.rw_url, execution_options={"schema_translate_map": {None: settings.plane}}
    )
    try:
        with engine.begin() as conn:
            removed = conn.execute(delete(users)).rowcount
    finally:
        engine.dispose()
    print(f"[e2e] emptied users on plane {settings.plane}: {removed} row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
