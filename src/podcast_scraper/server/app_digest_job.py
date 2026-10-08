"""The player's scheduled ``digest`` job (#1415): enqueue every due per-user delivery envelope.

Registered by the player extension as job kind ``digest`` (ADR-158). Every enqueuer lives in
``app_digest_dispatch.ENQUEUERS``, which the production sidecar
(``infra/deploy/digest_scheduler.py``) drives from the same list: adding one there wires both
paths. Maintaining two lists is what
shipped ``daily_recap`` and the monthly recommendations digest dead to prod (#2119). Each enqueuer
applies its own cadence-slot gate and per-period envelope id, so an hourly fire is idempotent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_digest_job(name: str, corpus_root: Path, app: Any) -> None:
    """Enqueue the due digests for every user; log what went out and what failed."""
    from podcast_scraper.server import app_digest_dispatch

    data_dir = getattr(app.state, "app_data_dir", None)
    if data_dir is None:
        logger.warning("scheduler: digest %r fired but app_data_dir unset; skipping", name)
        return
    result = app_digest_dispatch.enqueue_all_due(corpus_root, Path(data_dir))
    logger.info(
        "scheduler: digest %r enqueued %d envelope(s) [%s]", name, result.total, result.summary()
    )
    for label, err in result.errors.items():
        logger.error("scheduler: digest %r enqueuer %s failed: %s", name, label, err)
