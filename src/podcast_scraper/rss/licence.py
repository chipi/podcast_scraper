"""Licence and copyright declarations in a podcast feed, recorded as evidence (#2325).

What a publisher DECLARES about its show, kept verbatim, by level (channel or episode) and by the
asset it applies to. Nothing here decides what the declaration permits: a recognised licence is
evidence, not clearance for any use of ours.

Mechanisms, matched by namespace URI and local name, never by prefix:

- RSS 2.0 ``<copyright>`` (no namespace): a channel-level free-text notice. Kept verbatim; it is
  not a licence identifier.
- Podcasting 2.0 ``<podcast:license>``: channel or item, an identifier (an SPDX-style slug such
  as ``cc-by-4.0``) or free text, with an optional ``url``; a custom licence must carry one. It
  describes the AUDIO/VIDEO licence only, so it is not extended to artwork, metadata or
  transcripts.
- Creative Commons RSS module ``<creativeCommons:license>``: channel or item, a licence URL.
- Notices ("All rights reserved", "licensed under …") and licence/terms links in a channel or
  item description, kept for a reader as notices and documents to review.

Recognised licences (the Creative Commons family) carry their conditions; anything else is kept
as it was written, without guessing. Conflicting declarations are flagged, never resolved. A feed
with none says "no explicit licence found" -- not "unrestricted", not "prohibited".
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from typing import Any, Iterable, List, Literal, Optional, Tuple
from urllib.parse import urljoin
from xml.etree import ElementTree as ET

from pydantic import BaseModel, Field

PODCAST_NS = "https://podcastindex.org/namespace/1.0"
CREATIVE_COMMONS_NS = "http://backend.userland.com/creativeCommonsRssModule"
_CONTENT_NS = "http://purl.org/rss/1.0/modules/content/"
_ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"

Level = Literal["channel", "episode"]
Mechanism = Literal["rss_copyright", "podcast_license", "creative_commons_license", "notice"]
#: The asset a declaration speaks for. ``unspecified``: the mechanism does not say.
AssetScope = Literal["audio_video", "unspecified"]
#: ``not_checked`` is the reader's state for a corpus written before this existed.
Status = Literal["declared", "no_explicit_licence_found", "not_checked"]


class KnownLicence(BaseModel):
    """A licence recognised from its identifier or URL, with what it says about four uses."""

    id: str = Field(description="Canonical id, e.g. 'CC-BY-NC-SA-4.0'.")
    name: str = Field(description="Display name, e.g. 'CC BY-NC-SA 4.0'.")
    url: str
    commercial_use: bool
    adaptations: bool
    attribution: bool
    share_alike: bool


class LicenceDeclaration(BaseModel):
    """One declaration as found in the feed."""

    level: Level
    mechanism: Mechanism
    #: Exactly as written (element text, or the notice sentence), whitespace collapsed.
    raw: str = ""
    url: Optional[str] = None
    asset_scope: AssetScope = "unspecified"
    known: Optional[KnownLicence] = None
    notes: List[str] = Field(default_factory=list)


class LicenceRecord(BaseModel):
    """A channel's or an episode's declarations, where they were read and when."""

    status: Status
    declarations: List[LicenceDeclaration] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    #: Licence or terms documents linked from the feed, for a reader to open.
    documents_for_review: List[str] = Field(default_factory=list)
    source_feed_url: Optional[str] = None
    retrieved_at: Optional[str] = None


# --- recognised licences ---------------------------------------------------------------------

_CC_CONDITIONS = {
    # slug -> (commercial_use, adaptations, attribution, share_alike)
    "by": (True, True, True, False),
    "by-sa": (True, True, True, True),
    "by-nd": (True, False, True, False),
    "by-nc": (False, True, True, False),
    "by-nc-sa": (False, True, True, True),
    "by-nc-nd": (False, False, True, False),
}
_CC_URL = re.compile(
    r"creativecommons\.org/licenses/(?P<type>by(?:-nc)?(?:-sa|-nd)?)/(?P<version>\d\.\d)", re.I
)
_CC0_URL = re.compile(r"creativecommons\.org/publicdomain/zero/(?P<version>\d\.\d)", re.I)
_CC_SLUG = re.compile(
    r"^cc[-_ ](?P<type>by(?:[-_ ]nc)?(?:[-_ ]sa|[-_ ]nd)?)[-_ ](?P<version>\d\.\d)$", re.I
)
_CC0_SLUG = re.compile(r"^cc0[-_ ](?P<version>\d\.\d)$", re.I)


def _cc(kind: str, version: str) -> KnownLicence:
    kind = kind.lower().replace("_", "-").replace(" ", "-")
    commercial, adaptations, attribution, share_alike = _CC_CONDITIONS[kind]
    return KnownLicence(
        id=f"CC-{kind.upper()}-{version}",
        name=f"CC {kind.upper()} {version}",
        url=f"https://creativecommons.org/licenses/{kind}/{version}/",
        commercial_use=commercial,
        adaptations=adaptations,
        attribution=attribution,
        share_alike=share_alike,
    )


def _cc0(version: str) -> KnownLicence:
    return KnownLicence(
        id=f"CC0-{version}",
        name=f"CC0 {version}",
        url=f"https://creativecommons.org/publicdomain/zero/{version}/",
        commercial_use=True,
        adaptations=True,
        attribution=False,
        share_alike=False,
    )


def recognise(text: Optional[str], url: Optional[str] = None) -> Optional[KnownLicence]:
    """A known licence from an identifier (``cc-by-4.0``) or a licence URL, else None."""
    for candidate in (url, text):
        if not candidate:
            continue
        candidate = candidate.strip()
        m = _CC_URL.search(candidate)
        if m:
            return _cc(m.group("type"), m.group("version"))
        m = _CC0_URL.search(candidate)
        if m:
            return _cc0(m.group("version"))
        m = _CC_SLUG.match(candidate)
        if m:
            return _cc(m.group("type"), m.group("version"))
        m = _CC0_SLUG.match(candidate)
        if m:
            return _cc0(m.group("version"))
    return None


# --- notices and links in descriptions -------------------------------------------------------

_NOTICE = re.compile(
    r"all rights reserved|creative commons|\bcc[- ]by\b|\bcc0\b|licen[cs]ed under|"
    r"under (?:a|the) [\w .-]{0,40}licen[cs]e",
    re.I,
)
_TAG = re.compile(r"<[^>]+>")
_SENTENCE = re.compile(r"[^.!?\n]*[.!?\n]?")
_HREF = re.compile(r"""href\s*=\s*["']([^"']+)["']""", re.I)
_BARE_URL = re.compile(r"https?://[^\s<>\"')]+")
_LICENCE_LINK = re.compile(
    r"creativecommons\.org|/licen[cs]e|/copyright|/terms|/legal|/rights|/usage", re.I
)


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _description_parts(el: ET.Element) -> List[str]:
    out = []
    for child in el:
        if not isinstance(child.tag, str):
            continue
        uri, name = _split(child.tag)
        if (uri, name) in (("", "description"), (_CONTENT_NS, "encoded"), (_ITUNES_NS, "summary")):
            if child.text:
                out.append(child.text)
    return out


def _notices_and_links(
    parts: Iterable[str], base_url: str, level: Level
) -> Tuple[List[LicenceDeclaration], List[str]]:
    notices: List[LicenceDeclaration] = []
    links: List[str] = []
    seen = set()
    for raw_html in parts:
        for href in _HREF.findall(raw_html) + _BARE_URL.findall(raw_html):
            url = urljoin(base_url, html.unescape(href.strip()))
            if _LICENCE_LINK.search(url) and url not in links:
                links.append(url)
        text = _collapse(html.unescape(_TAG.sub(" ", raw_html)))
        for sentence in _SENTENCE.findall(text):
            sentence = sentence.strip()
            if sentence and _NOTICE.search(sentence) and sentence.lower() not in seen:
                seen.add(sentence.lower())
                notices.append(
                    LicenceDeclaration(
                        level=level,
                        mechanism="notice",
                        raw=sentence[:400],
                        known=recognise(sentence),
                    )
                )
    return notices, links


# --- extraction --------------------------------------------------------------------------------


def _split(tag: str) -> Tuple[str, str]:
    if tag.startswith("{"):
        uri, _, name = tag[1:].partition("}")
        return uri, name
    return "", tag


def _declarations(el: ET.Element, level: Level, base_url: str) -> List[LicenceDeclaration]:
    out: List[LicenceDeclaration] = []
    for child in el:
        if not isinstance(child.tag, str):
            continue
        uri, name = _split(child.tag)
        text = _collapse(child.text or "")
        if (uri, name) == ("", "copyright") and level == "channel":
            out.append(
                LicenceDeclaration(
                    level=level, mechanism="rss_copyright", raw=text, known=recognise(text)
                )
            )
        elif (uri, name) == (PODCAST_NS, "license"):
            url = (child.get("url") or "").strip() or None
            url = urljoin(base_url, url) if url else None
            known = recognise(text, url)
            notes = []
            if not known and not url:
                notes.append("custom licence without a url (the Podcasting 2.0 spec requires one)")
            out.append(
                LicenceDeclaration(
                    level=level,
                    mechanism="podcast_license",
                    raw=text,
                    url=url,
                    asset_scope="audio_video",
                    known=known,
                    notes=notes,
                )
            )
        elif (uri, name) == (CREATIVE_COMMONS_NS, "license"):
            out.append(
                LicenceDeclaration(
                    level=level,
                    mechanism="creative_commons_license",
                    raw=text,
                    url=text or None,
                    known=recognise(None, text),
                )
            )
    return out


def _conflicts(declarations: List[LicenceDeclaration]) -> List[str]:
    out: List[str] = []
    by_mechanism: dict[str, int] = {}
    for d in declarations:
        if d.mechanism != "notice":
            by_mechanism[d.mechanism] = by_mechanism.get(d.mechanism, 0) + 1
    for mechanism, n in sorted(by_mechanism.items()):
        if n > 1:
            out.append(f"{n} {mechanism} declarations at one level")
    known_ids = sorted({d.known.id for d in declarations if d.known})
    if len(known_ids) > 1:
        out.append("different licences declared: " + ", ".join(known_ids))
    reserved = any(
        "all rights reserved" in d.raw.lower()
        for d in declarations
        if d.mechanism in ("rss_copyright", "notice")
    )
    if reserved and known_ids:
        out.append("'All rights reserved' alongside a licence: " + ", ".join(known_ids))
    return out


def _record(
    declarations: List[LicenceDeclaration],
    links: List[str],
    source_feed_url: Optional[str],
    retrieved_at: Optional[str],
) -> LicenceRecord:
    for d in declarations:
        if d.url and d.url not in links:
            links.append(d.url)
    return LicenceRecord(
        status="declared" if declarations else "no_explicit_licence_found",
        declarations=declarations,
        conflicts=_conflicts(declarations),
        documents_for_review=links,
        source_feed_url=source_feed_url,
        retrieved_at=retrieved_at,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def channel_licence(
    channel: ET.Element,
    base_url: str,
    source_feed_url: Optional[str] = None,
    retrieved_at: Optional[str] = None,
) -> LicenceRecord:
    """The channel's declarations: ``<copyright>``, ``<podcast:license>``,
    ``<creativeCommons:license>``, and notices / licence links in its description."""
    declarations = _declarations(channel, "channel", base_url)
    notices, links = _notices_and_links(_description_parts(channel), base_url, "channel")
    return _record(
        declarations + notices, links, source_feed_url or base_url, retrieved_at or _now()
    )


def episode_licence(
    item: ET.Element,
    base_url: str,
    channel: Optional[LicenceRecord] = None,
    source_feed_url: Optional[str] = None,
    retrieved_at: Optional[str] = None,
) -> LicenceRecord:
    """An item's own declarations (``<podcast:license>``, ``<creativeCommons:license>``, notices
    and licence links in its show notes). The channel's are not copied in: an episode that
    declares nothing records "no explicit licence found" at the episode level, and the channel
    record still applies. An episode licence different from the channel's is noted, not
    resolved -- Podcasting 2.0 lets an item override its channel."""
    declarations = _declarations(item, "episode", base_url)
    notices, links = _notices_and_links(_description_parts(item), base_url, "episode")
    record = _record(
        declarations + notices, links, source_feed_url or base_url, retrieved_at or _now()
    )
    if channel is not None:
        mine = {d.known.id for d in record.declarations if d.known}
        theirs = {d.known.id for d in channel.declarations if d.known}
        if mine and theirs and mine != theirs:
            record.conflicts.append(
                "episode licence differs from the channel's: "
                + ", ".join(sorted(mine))
                + " vs "
                + ", ".join(sorted(theirs))
            )
    return record


def channel_licence_from_bytes(
    xml_bytes: bytes, base_url: str, source_feed_url: Optional[str] = None
) -> Optional[LicenceRecord]:
    """``channel_licence`` from raw feed XML; None when the XML has no channel."""
    from podcast_scraper.rss.parser import safe_fromstring

    try:
        root = safe_fromstring(xml_bytes)
    except Exception:  # noqa: BLE001 - a feed the main parser accepted must not fail here
        return None
    if root is None:
        return None
    channel = next(
        (e for e in root.iter() if isinstance(e.tag, str) and _split(e.tag)[1] == "channel"),
        None,
    )
    if channel is None:
        return None
    return channel_licence(channel, base_url, source_feed_url)


# --- the one line a reader sees ------------------------------------------------------------------

NO_LICENCE_LABEL = "No explicit licence found"


def summary_label(
    record: Optional[LicenceRecord | dict[str, Any]], limit: int = 90
) -> Optional[str]:
    """A short line for the show page: a recognised licence's name, else the publisher's own
    notice shortened, else "No explicit licence found". None when the feed was not checked."""
    if record is None:
        return None
    if isinstance(record, dict):
        record = LicenceRecord.model_validate(record)
    if record.status == "not_checked":
        return None
    order = {"podcast_license": 0, "creative_commons_license": 1, "rss_copyright": 2, "notice": 3}
    decls = sorted(record.declarations, key=lambda d: order[d.mechanism])
    known = next((d.known for d in decls if d.known), None)
    if known is not None:
        return known.name
    for d in decls:
        if d.raw:
            return d.raw if len(d.raw) <= limit else d.raw[: limit - 1].rstrip() + "…"
    return NO_LICENCE_LABEL
