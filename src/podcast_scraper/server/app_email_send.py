"""Send a rendered email via Resend's REST API (RFC-122 #2039 / #1412).

No SDK — a plain HTTPS POST with ``httpx`` (already a core dependency). The API key is PASSED IN
(never read at import), and the transport is a small injectable object so tests use a fake and
dev/CI never touch the network. The delivery worker constructs a :class:`ResendTransport` only when
a key is configured; otherwise it dry-runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import httpx

from podcast_scraper.server.app_email_render import RenderedEmail

logger = logging.getLogger(__name__)

_RESEND_URL = "https://api.resend.com/emails"


@dataclass(frozen=True)
class SendResult:
    """Outcome of one send. ``ok`` False leaves the envelope pending for a later retry."""

    ok: bool
    detail: str


class EmailTransport(Protocol):
    """The send seam. A real Resend client in prod; a fake in tests."""

    def send(
        self, *, to: str, email: RenderedEmail, from_addr: str, headers: dict[str, str]
    ) -> SendResult: ...


class ResendTransport:
    """Send via Resend's REST API. Constructed only when an API key is present."""

    def __init__(self, api_key: str, *, timeout: float = 10.0) -> None:
        self._api_key = api_key
        self._timeout = timeout

    def send(
        self, *, to: str, email: RenderedEmail, from_addr: str, headers: dict[str, str]
    ) -> SendResult:
        try:
            resp = httpx.post(
                _RESEND_URL,
                timeout=self._timeout,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "from": from_addr,
                    "to": [to],
                    "subject": email.subject,
                    "html": email.html,
                    "text": email.text,
                    "headers": headers,
                },
            )
        except httpx.HTTPError as exc:
            return SendResult(False, f"transport error: {exc}")
        if resp.status_code // 100 == 2:
            return SendResult(True, f"resend {resp.status_code}")
        return SendResult(False, f"resend {resp.status_code}: {resp.text[:200]}")
