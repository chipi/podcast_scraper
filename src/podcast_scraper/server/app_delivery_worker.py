"""Drain the outbox and deliver pending emails (RFC-122 #2039 / #1412).

The renderer + sender for the app↔infra delivery seam, co-located in-repo so the daily recap (and
the other digest emails) are deployable end-to-end. It consumes ONLY the committed seam
(``app_outbox_store.list_pending`` / ``record_status``), so it could move to a standalone infra
service unchanged.

Safe-by-default: with no ``RESEND_API_KEY`` it **dry-runs** — renders + logs, sends nothing, leaves
the envelope pending — so dev/CI never send and no secret is needed to build or test. In the deploy
env, set ``RESEND_API_KEY`` (+ optional ``EMAIL_FROM`` / ``APP_ORIGIN``) and it delivers.

Retry stance: a send failure LEAVES the envelope pending (logged, not dead-lettered), so the next
drain retries; the envelope's ``expires_at`` TTL is the backstop. Only a confirmed send records the
terminal ``delivered`` status.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

from podcast_scraper.server import app_email_render, app_email_send, app_outbox_store
from podcast_scraper.server.app_email_send import EmailTransport

logger = logging.getLogger(__name__)

DEFAULT_ORIGIN = "https://closelistening.app"
DEFAULT_FROM = "closelistening <recap@mail.closelistening.app>"


@dataclass
class DeliverySummary:
    """Per-drain counts (for logs + the scheduler)."""

    delivered: int = 0
    failed: int = 0
    dry_run: int = 0
    skipped: int = 0


def _unsubscribe_urls(origin: str, envelope: dict) -> tuple[str, str]:
    """The RFC-8058 one-click unsubscribe URL (type-aware) + the in-app settings URL."""
    snap = envelope.get("consent_snapshot") or {}
    ref = quote_plus(str(snap.get("unsubscribe_ref") or ""))
    ntype = quote_plus(str(envelope.get("type") or "digest"))
    base = origin.rstrip("/")
    return f"{base}/api/app/comms/unsubscribe?ref={ref}&type={ntype}", f"{base}/profile"


def _list_unsubscribe_headers(unsubscribe_url: str) -> dict[str, str]:
    return {
        "List-Unsubscribe": f"<{unsubscribe_url}>",
        "List-Unsubscribe-Post": "List-Unsubscribe=One-Click",
    }


def deliver_pending_emails(
    data_dir: Path,
    *,
    origin: str | None = None,
    api_key: str | None = None,
    from_addr: str | None = None,
    transport: EmailTransport | None = None,
    now: int | None = None,
    limit: int = 50,
) -> DeliverySummary:
    """Render + send every pending email envelope; record ``delivered`` on success.

    ``transport`` is injected in tests; in prod it is built from ``api_key`` (env ``RESEND_API_KEY``
    by default). With neither a transport nor a key, this dry-runs.
    """
    origin = origin or os.environ.get("APP_ORIGIN") or DEFAULT_ORIGIN
    api_key = api_key if api_key is not None else os.environ.get("RESEND_API_KEY")
    from_addr = from_addr or os.environ.get("EMAIL_FROM") or DEFAULT_FROM
    if transport is None and api_key:
        transport = app_email_send.ResendTransport(api_key)

    summary = DeliverySummary()
    for env in app_outbox_store.list_pending(data_dir, channel="email", limit=limit, now=now):
        env_id = str(env.get("id") or "")
        to = str((env.get("recipient") or {}).get("email") or "")
        if not to or not env_id:
            summary.skipped += 1
            continue
        unsub_url, settings_url = _unsubscribe_urls(origin, env)
        try:
            rendered = app_email_render.render_email(
                env, origin=origin, unsubscribe_url=unsub_url, settings_url=settings_url
            )
        except app_email_render.UnknownTemplateError:
            logger.warning(
                "delivery: no renderer for template %r (%s); skipping", env.get("template"), env_id
            )
            summary.skipped += 1
            continue
        if transport is None:
            logger.info("delivery[dry-run]: would send %s to %s — %r", env_id, to, rendered.subject)
            summary.dry_run += 1
            continue
        result = transport.send(
            to=to, email=rendered, from_addr=from_addr, headers=_list_unsubscribe_headers(unsub_url)
        )
        if result.ok:
            app_outbox_store.record_status(data_dir, env_id, "delivered", result.detail)
            summary.delivered += 1
        else:
            # Ambiguous transient/permanent — leave PENDING so the next drain retries; the
            # envelope's expires_at TTL is the backstop against forever-retrying a bad one.
            logger.warning("delivery: send failed for %s: %s (left pending)", env_id, result.detail)
            summary.failed += 1
    return summary


def main() -> int:
    """CLI: drain ``APP_DATA_DIR``'s outbox. Dry-runs without ``RESEND_API_KEY``."""
    logging.basicConfig(level=logging.INFO)
    data_dir = os.environ.get("APP_DATA_DIR")
    if not data_dir:
        logger.error("delivery: APP_DATA_DIR is unset")
        return 2
    summary = deliver_pending_emails(Path(data_dir))
    logger.info(
        "delivery: delivered=%d failed=%d dry_run=%d skipped=%d",
        summary.delivered,
        summary.failed,
        summary.dry_run,
        summary.skipped,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
