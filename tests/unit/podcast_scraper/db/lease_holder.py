"""A second process for the lease tests: takes leases, then obeys lines on stdin.

``python -m tests.unit.podcast_scraper.db.lease_holder URL HOLDER TTL_S RESOURCE=MODE...`` prints
``HELD n`` or ``REFUSED ...``. Then each stdin line is a command: ``chunk KEY`` runs one chunk
transaction (heartbeat, then a ``corpus_meta`` row KEY, then commit) and prints ``COMMITTED`` or
``LOST``; ``release`` drops the leases, prints ``RELEASED`` and exits.
"""

from __future__ import annotations

import sys
from datetime import timedelta

from sqlalchemy import create_engine, insert

from podcast_scraper.db.leases import acquire, heartbeat, LeaseLost, LeaseRefused, LeaseRequest
from podcast_scraper.db.schema import corpus_meta
from podcast_scraper.db.settings import normalize_url


def main(argv: list[str]) -> int:
    """Run the holder; see the module docstring."""
    url, holder, ttl_s, *specs = argv
    engine = create_engine(normalize_url(url))
    ttl = timedelta(seconds=float(ttl_s))
    requests = [LeaseRequest(*spec.split("=", 1)) for spec in specs]
    try:
        held = acquire(engine, requests, holder, ttl=ttl)
    except LeaseRefused as exc:
        print(f"REFUSED {exc}", flush=True)
        return 1
    print(f"HELD {held}", flush=True)
    for line in sys.stdin:
        command, *args = line.split()
        if command == "chunk":
            try:
                with engine.begin() as conn:
                    heartbeat(conn, holder, held, ttl=ttl)
                    conn.execute(insert(corpus_meta).values(key=args[0], value={}))
                print("COMMITTED", flush=True)
            except LeaseLost:
                print("LOST", flush=True)
        elif command == "release":
            from podcast_scraper.db.leases import release

            release(engine, holder)
            print("RELEASED", flush=True)
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
