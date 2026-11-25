"""The re-check's page fetch (CRFC-006 Part B, B.2 steps 1 and 3; public's fetch half).

One check of a written article: fetch its page as the ingest does (robots.txt, the source's user
agent, the shared downloader's retries, SSRF guard and per-host throttle), conditionally when the
last answer carried validators, and tell apart what the schedule must treat differently: a page to
compare, "not modified", gone (404 or 410: two in a row withdraw the item) and failed (a timeout,
a 5xx, a refusal: retried on the next day, never counted as gone).

The re-check job (Common ingestion) calls :func:`check` and passes its ``status`` to
``recheck.advance``. Not here: the Guardian's re-read by id (its transport's), re-extracting the
stored raw page when the extractor changed (B.2 step 2; needs the extractor's version on the row),
and the rows themselves (the database session's).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..rss import downloader
from . import extract, recheck
from .fetch import robots_allow, source_spec, TIMEOUT_S, user_agent

logger = logging.getLogger(__name__)

PAGE = "page"
NOT_MODIFIED = "not_modified"

_GONE_STATUSES = (404, 410)


@dataclass(frozen=True)
class Validators:
    """What a conditional GET sends back: the page's ``ETag`` and ``Last-Modified``."""

    etag: Optional[str] = None
    last_modified: Optional[str] = None

    def request_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.etag:
            headers["If-None-Match"] = self.etag
        if self.last_modified:
            headers["If-Modified-Since"] = self.last_modified
        return headers


@dataclass(frozen=True)
class Refetch:
    """One fetch's answer: ``PAGE`` (with its HTML), ``NOT_MODIFIED``, ``recheck.GONE`` or
    ``recheck.FAILED``; ``http_status`` is 0 when no response came back."""

    kind: str
    http_status: int
    validators: Validators
    html: Optional[str] = None


def refetch(url: str, source: Any, validators: Validators = Validators()) -> Refetch:
    """Fetch ``url`` for a re-check. A robots.txt refusal, no response or any status other than
    2xx, 304, 404 and 410 is ``FAILED``: only a page that says it is gone counts towards a
    withdrawal."""
    agent = user_agent(source)
    if not robots_allow(url, agent):
        logger.info("news re-check: robots.txt disallows %s; failed", url)
        return Refetch(recheck.FAILED, 0, validators)
    resp = downloader.http_get_response(
        url, agent, TIMEOUT_S, extra_headers=validators.request_headers()
    )
    if resp is None:
        return Refetch(recheck.FAILED, 0, validators)
    status = resp.status_code
    if status == 304:
        return Refetch(NOT_MODIFIED, status, _merged(validators, resp.headers))
    if status in _GONE_STATUSES:
        return Refetch(recheck.GONE, status, validators)
    if 200 <= status < 300:
        return Refetch(PAGE, status, _merged(Validators(), resp.headers), html=resp.text)
    logger.info("news re-check: %s answered %d; failed", url, status)
    return Refetch(recheck.FAILED, status, validators)


def _merged(old: Validators, headers: Any) -> Validators:
    """The response's validators, keeping an old one the response did not repeat."""
    return Validators(
        etag=headers.get("ETag") or old.etag,
        last_modified=headers.get("Last-Modified") or old.last_modified,
    )


@dataclass(frozen=True)
class Check:
    """One check's result: ``status`` for ``recheck.advance``, the comparison when a page was
    compared, the validators to store for the next check, and the page (the raw cache)."""

    status: str
    validators: Validators
    http_status: int
    comparison: Optional[recheck.Comparison] = None
    html: Optional[str] = None


def check(
    served_text: str,
    url: str,
    source: Any,
    validators: Validators = Validators(),
    *,
    hold_share: float = recheck.DEFAULT_HOLD_SHARE,
) -> Check:
    """Re-check one served article. "Not modified" is ``SAME`` without extracting. A page is
    extracted as the ingest extracts it (the source's ``text_from``) and compared; a page with no
    text compares as empty, so it is held, never written."""
    got = refetch(url, source, validators)
    if got.kind == NOT_MODIFIED:
        return Check(recheck.SAME, got.validators, got.http_status)
    if got.kind != PAGE:
        return Check(got.kind, got.validators, got.http_status)
    found = extract.main_text(got.html or "", url, text_from=source_spec(source).text_from)
    comparison = recheck.compare(
        served_text, found.text if found else "", url=url, hold_share=hold_share
    )
    return Check(comparison.outcome, got.validators, got.http_status, comparison, got.html)
