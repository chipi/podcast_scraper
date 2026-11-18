"""The canonical form of a URL, for ids minted from URLs (ontology v3; RFC-134 §2).

An item or source identified by URL gets the same id however the URL arrives: one article over
two transports, or linked with and without tracking parameters, is one item. The rules, each
tested:

- **scheme:** lower case; ``http`` becomes ``https``;
- **host:** lower case; the default port dropped; an ``amp.`` host prefix dropped;
- **path:** a trailing slash dropped (an empty path is ``/``); an AMP suffix (``/amp``) dropped;
- **query:** tracking parameters dropped (``utm_*``, the click ids and feed tags below) and AMP
  switches
  (``amp``, ``outputType=amp``); the rest sorted, so their order does not matter;
- **fragment:** dropped.

Anything else is kept as given: path case, percent-encoding, ``www.``. A rule that is not sure
two URLs are one page leaves them two items, never one.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

#: Query parameters that only track a click, never select content.
TRACKING_PARAMS = frozenset(
    {
        "fbclid",
        "gclid",
        "dclid",
        "msclkid",
        "yclid",
        "igshid",
        "mc_cid",
        "mc_eid",
        "_ga",
        "ref_src",
        # News feeds' own tracking (the operator, 2026-11-18; seen on the first real news corpus):
        # the BBC's analytics tags and Al Jazeera's feed tag. The BBC family is listed name by
        # name, not as an ``at_`` prefix, so a content parameter that happens to start ``at_``
        # is never dropped.
        "at_medium",
        "at_campaign",
        "at_link_id",
        "at_link_origin",
        "at_link_type",
        "at_ptr_name",
        "at_format",
        "at_bbc_team",
        "traffic_source",
    }
)
_AMP_SWITCHES = {("amp", ""), ("amp", "1"), ("amp", "true"), ("outputtype", "amp")}
_DEFAULT_PORTS = {"http": 80, "https": 443}


def _is_tracking(key: str) -> bool:
    k = key.lower()
    return k.startswith("utm_") or k in TRACKING_PARAMS


def canonical_url(url: str) -> str:
    """``url`` in the canonical form described in this module's docstring.

    Raises ``ValueError`` for a URL with no scheme or host: an id cannot be minted from it.
    """
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if not scheme or not host:
        raise ValueError(f"not an absolute URL: {url!r}")
    if scheme == "http":
        scheme = "https"
    if host.startswith("amp."):
        host = host[len("amp.") :]
    port = parts.port
    netloc = host if port is None or port == _DEFAULT_PORTS.get(scheme) else f"{host}:{port}"

    path = parts.path or "/"
    if path.rstrip("/").endswith("/amp"):
        path = path.rstrip("/")[: -len("/amp")] or "/"
    if len(path) > 1:
        path = path.rstrip("/") or "/"

    query = sorted(
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not _is_tracking(k) and (k.lower(), v.lower()) not in _AMP_SWITCHES
    )
    return urlunsplit((scheme, netloc, path, urlencode(query), ""))
