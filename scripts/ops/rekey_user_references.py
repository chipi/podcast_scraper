#!/usr/bin/env python3
"""Re-key or flag every listener's stored references, once (#2345). Dry-run unless ``--apply``.

The command lives in the package (``podcast_scraper.server.rekey_references``) so it runs in any
image (``python -m podcast_scraper.server.rekey_references``); this is the same command from a
checkout. See that module for what it does and how to undo it.
"""

from __future__ import annotations

import sys

from podcast_scraper.server.rekey_references import main, RECEIPTS_FILE  # noqa: F401

if __name__ == "__main__":
    sys.exit(main())
