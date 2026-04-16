#!/usr/bin/env python3
"""Run the pytest E2E HTTP mock (RSS + API stubs) on a fixed port for manual YAML configs.

Default port **18765** matches ``config/examples/manual_e2e_mock_five_podcasts.yaml``.

Usage::
    make serve-e2e-mock
    E2E_MOCK_PORT=19000 make serve-e2e-mock
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Repo root (…/scripts/tools -> parents[2])
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("E2E_MOCK_PORT", "18765")),
        help="Listen port (default: 18765 or E2E_MOCK_PORT)",
    )
    parser.add_argument(
        "--podcasts",
        type=str,
        default="podcast1,podcast2,podcast3,podcast4,podcast5",
        help="Comma-separated feed names (under /feeds/<name>/feed.xml)",
    )
    parser.add_argument(
        "--fast-fixtures",
        action="store_true",
        help="Use fast single-minute episodes (E2E fast mode); default is full fixtures",
    )
    args = parser.parse_args()

    from tests.e2e.fixtures.e2e_http_server import (  # noqa: E402 (sys.path first)
        E2EHTTPRequestHandler,
        E2EHTTPServer,
    )

    names = {x.strip() for x in args.podcasts.split(",") if x.strip()}
    E2EHTTPRequestHandler.set_allowed_podcasts(names)
    E2EHTTPRequestHandler.set_use_fast_fixtures(bool(args.fast_fixtures))

    server = E2EHTTPServer(port=args.port)
    server.start()
    print(f"E2E mock server listening: {server.base_url}", flush=True)
    print("Feeds (example):", flush=True)
    for n in sorted(names):
        print(f"  {server.urls.feed(n)}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Stopping…", flush=True)
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
