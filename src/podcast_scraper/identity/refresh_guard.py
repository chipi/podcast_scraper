"""The ledger refresh of ``upgrade run`` and ``enrich``, holding back the respelling until 0041.

RFC-131 0041: the first respelling of the corpus is migration 0041's own run, with its receipts and
the outcome hook's measurement (record ``0041_joined_names_respelled``, "the deploy order").
:func:`~podcast_scraper.identity.ledger.refresh_ledger` respells on every build, so a refresh after
an earlier migration's run, or after an enrich, would do 0041's work unmeasured. While 0041 is not
applied, the ledger is built and written and the respelling is deferred, logged and recorded in the
change log; from 0041 on, ``refresh_ledger`` runs as it is.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

RESPELL_MIGRATION = "0041_joined_names_respelled"


def refresh_ledger_after(
    root: Path | str, *, after: str, log: Optional[logging.Logger] = None
) -> bool:
    """:func:`refresh_ledger`, without the respelling while 0041 is pending. True when written."""
    from podcast_scraper.identity.ledger import build_ledger, refresh_ledger, write_ledger
    from podcast_scraper.upgrade.state import FilesystemStateStore

    log = log or logger
    if RESPELL_MIGRATION in FilesystemStateStore(Path(root)).applied_migration_ids():
        return refresh_ledger(root, after=after, log=log)
    try:
        built = build_ledger(root)
        change = write_ledger(
            root,
            built,
            extra={"respell": {"deferred": True, "reason": f"{RESPELL_MIGRATION} pending"}},
        )
    except Exception:  # noqa: BLE001 - any build or write failure leaves the previous ledger
        log.exception("identity ledger refresh after %s failed; the previous ledger stays", after)
        return False
    log.info(
        "identity ledger refreshed after %s: %d joins; respelling deferred until %s is applied",
        after,
        change["joins"],
        RESPELL_MIGRATION,
    )
    return True


def refresh_ledger_after_batch(
    corpus_parent: Path | str, template_cfg: object, log: Optional[logging.Logger] = None
) -> bool:
    """The identity ledger refresh a multi-feed batch owns, called after its finalize.

    A batch's feeds walk their run dirs only, so their edge passes leave the ledger to the batch
    (``is_multi_feed_batch_child``). When the batch queued an enrichment, that run refreshes the
    ledger when it finishes, and a refresh here would invalidate every API cache once more for
    the same joins; so it refreshes only when no enrichment was queued. True when nothing was
    needed or the ledger was written.

    Called from the batch loops (``cli.py``, ``service.py``), not from
    ``finalize_multi_feed_batch``: that module is inside the D2 migrations' measured code, and
    this one must stay outside it.
    """
    block = getattr(template_cfg, "enrichment", None)
    if isinstance(block, dict) and block.get("enabled"):
        return True
    return refresh_ledger_after(corpus_parent, after="multi-feed batch", log=log)
