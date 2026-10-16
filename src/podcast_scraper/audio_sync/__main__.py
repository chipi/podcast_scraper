"""``python -m podcast_scraper.audio_sync <command>`` — RFC-129 operations (#2359).

    drift-check        which episodes' served audio no longer matches what we transcribed
    build-maps         time maps for those episodes, from the archived original
    build-references   fingerprint checkpoints of the original, for listener-side sync

Each command's ``--help`` says what it reads and writes. Runbook:
``docs/recipes/audio-timing-drift.md``.
"""

from __future__ import annotations

import sys


def main(argv: list) -> int:
    """Run the named command with the rest of ``argv``; with none, print the list and return 2."""
    commands = {
        "drift-check": "podcast_scraper.audio_sync.drift",
        "build-maps": "podcast_scraper.audio_sync.maps",
        "build-references": "podcast_scraper.audio_sync.references",
    }
    if not argv or argv[0] not in commands:
        print(__doc__)
        return 2
    import importlib

    module = importlib.import_module(commands[argv[0]])
    result: int = module.main(argv[1:])
    return result


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
