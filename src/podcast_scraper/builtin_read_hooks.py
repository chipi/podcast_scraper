"""Public's own read-side owners on the read hooks (E04 S3): none today.

The insight and identity read side moved to the Close Listening stages extension with the E04
move (``Extension.read_hooks``, and its GI and person routes through ``Extension.routers``).
Without it each read hook answers its plain default (``podcast_scraper.read_hooks``).
"""

from __future__ import annotations

from typing import Any, Sequence

from .read_hooks import ReadHooks


def ROUTERS() -> Sequence[Any]:
    """Public's own stage routers: none (the GI explore, person and CIL routes are the
    extension's)."""
    return ()


BUILTIN = ReadHooks(name="builtin")
