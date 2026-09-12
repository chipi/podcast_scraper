"""Render a delivery-envelope payload to an email (subject + HTML + text) — RFC-122 #2039 / #1412.

This is the renderer the delivery worker uses. It lives in-repo (co-located with the outbox seam it
drains) so the daily recap is deployable end-to-end; it still consumes only the committed
``DeliveryEnvelope`` payloads, so it could move to a standalone infra service unchanged.

Email-safe output: table layout + inline styles, a light background (dark-mode is client-
inconsistent), the ``closelistening.`` wordmark + gold accent. All user text is HTML-escaped and
every ``/player/...`` deep link is absolutized against the app origin. The visual reference is
``docs/wip/daily-recap-email.html``.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any

_ACCENT = "#b6791f"
_MUTED = "#8a938d"
_QUOTE_INK = "#26302c"

# Reusable inline-style fragments (kept as constants so the builders below stay within line length).
_S_WORDMARK = "font:700 18px/1 'Segoe UI',Helvetica,Arial,sans-serif;letter-spacing:-.01em"
_S_KICKER = (
    "font:600 12px/1.4 ui-monospace,Menlo,monospace;letter-spacing:.12em;"
    f"text-transform:uppercase;color:{_MUTED}"
)
_S_TITLE_LG = "font:700 22px/1.3 'Segoe UI',Helvetica,Arial,sans-serif"
_S_EP_TITLE = "margin-top:3px;font:700 19px/1.35 'Segoe UI',Helvetica,Arial,sans-serif"
_S_EP_TITLE_SM = "margin-top:2px;font:700 16px/1.35 'Segoe UI',Helvetica,Arial,sans-serif"
_S_QUOTE = f"font:italic 600 16px/1.45 Georgia,serif;color:{_QUOTE_INK}"
_S_CHIP = (
    "display:inline-block;border:1px solid #d8dcd6;border-radius:999px;padding:5px 12px;"
    f"margin:0 6px 6px 0;font-size:13px;color:{_QUOTE_INK}"
)
_S_BUTTON = (
    f"display:inline-block;background:{_ACCENT};color:#fff;text-decoration:none;font-weight:600;"
    "font-size:15px;padding:11px 22px;border-radius:999px"
)
_S_LINK = f"color:{_ACCENT};font-weight:600;font-size:14px;text-decoration:none"


@dataclass(frozen=True)
class RenderedEmail:
    """A rendered email ready for the transport."""

    subject: str
    html: str
    text: str


class UnknownTemplateError(ValueError):
    """The envelope's template has no renderer here."""


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _abs(origin: str, deep_link: Any) -> str:
    """Absolutize an app-relative deep link against the origin; pass through absolute/empty."""
    link = str(deep_link or "")
    if not link:
        return _esc(origin or "")
    if link.startswith("http://") or link.startswith("https://"):
        return _esc(link)
    return _esc(f"{origin.rstrip('/')}{link}" if origin else link)


def _wordmark() -> str:
    return (
        f'<span style="{_S_WORDMARK}">closelistening'
        f'<span style="color:{_ACCENT};">.</span></span>'
    )


def _kicker(text: str) -> str:
    return f'<div style="{_S_KICKER};">{_esc(text)}</div>'


def _shell(masthead_html: str, body_rows: str, footer_html: str) -> str:
    """The 600px email frame: masthead header block + body rows + footer, on a light page."""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>closelistening</title></head>"
        '<body style="margin:0;background:#e9ebe8;'
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;"
        'color:#1a1c1b;">'
        '<div style="max-width:600px;margin:0 auto;padding:24px 12px;">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        'style="background:#fff;border:1px solid #e2e5e1;border-radius:16px;overflow:hidden;">'
        '<tr><td style="padding:22px 28px 14px;border-bottom:1px solid #eef0ed;">'
        f"{masthead_html}</td></tr>"
        f"{body_rows}"
        '<tr><td style="padding:16px 28px 22px;border-top:1px solid #eef0ed;">'
        f"{footer_html}</td></tr>"
        "</table></div></body></html>"
    )


def _masthead(day: str, headline: str, subline: str) -> str:
    day_line = (
        f'<div style="margin-top:12px;{_S_KICKER};letter-spacing:.14em;">'
        f"Your day, recapped · {_esc(day)}</div>"
    )
    head_line = f'<div style="margin-top:4px;{_S_TITLE_LG};">{_esc(headline)}</div>'
    sub_line = f'<div style="margin-top:2px;font-size:14px;color:#6b7770;">{_esc(subline)}</div>'
    return f"{_wordmark()}{day_line}{head_line}{sub_line}"


def _quote_block(quote: dict[str, Any]) -> str:
    speaker = quote.get("speaker")
    by = (
        f'<span style="font-style:normal;font-size:13px;color:#6b7770;"> — {_esc(speaker)}</span>'
        if speaker
        else ""
    )
    return (
        f'<div style="margin-top:12px;border-left:3px solid {_ACCENT};padding-left:14px;">'
        f'<span style="{_S_QUOTE};">“{_esc(quote.get("text"))}”</span>{by}</div>'
    )


def _key_point_rows(points: list[Any]) -> str:
    cells = ""
    for p in points[:3]:
        cells += (
            f'<tr><td style="vertical-align:top;padding:3px 8px 3px 0;color:{_ACCENT};'
            'font-weight:700;">•</td>'
            f'<td style="padding:3px 0;font-size:15px;line-height:1.5;">{_esc(p)}</td></tr>'
        )
    return (
        f'<div style="margin-top:14px;">{_kicker("Key points")}</div>'
        '<table role="presentation" cellpadding="0" cellspacing="0" '
        f'style="margin-top:6px;">{cells}</table>'
    )


def _episode_full(ep: dict[str, Any], origin: str) -> str:
    parts = [
        _kicker(ep.get("podcast_title") or ""),
        f'<div style="{_S_EP_TITLE};">{_esc(ep.get("title"))}</div>',
    ]
    if ep.get("key_points"):
        parts.append(_key_point_rows(list(ep["key_points"])))
    if ep.get("signature_quote"):
        parts.append(_quote_block(ep["signature_quote"]))
    insights = ep.get("insights") or []
    if insights:
        parts.append(f'<div style="margin-top:16px;">{_kicker("Top insights")}</div>')
        for text in insights[:3]:
            parts.append(
                f'<div style="margin-top:6px;font-size:15px;line-height:1.5;">{_esc(text)}</div>'
            )
    chips = "".join(
        f'<span style="{_S_CHIP};">{_esc(t.get("label"))}</span>' for t in (ep.get("topics") or [])
    )
    if chips:
        parts.append(f'<div style="margin-top:16px;">{chips}</div>')
    parts.append(
        f'<div style="margin-top:20px;"><a href="{_abs(origin, ep.get("deep_link"))}" '
        f'style="{_S_BUTTON};">Open in the app</a></div>'
    )
    return "".join(parts)


def _episode_compact(ep: dict[str, Any], origin: str, *, last: bool) -> str:
    border = "" if last else "border-bottom:1px solid #f1f3f0;"
    quote_html = _quote_block(ep["signature_quote"]) if ep.get("signature_quote") else ""
    points = ep.get("key_points") or []
    points_html = ""
    if points:
        joined = "&nbsp;&nbsp;".join(f"• {_esc(p)}" for p in points[:2])
        points_html = (
            '<div style="margin-top:8px;font-size:14px;line-height:1.5;'
            f'color:#3a423d;">{joined}</div>'
        )
    open_link = (
        f'<div style="margin-top:8px;"><a href="{_abs(origin, ep.get("deep_link"))}" '
        f'style="{_S_LINK};">Open in the app →</a></div>'
    )
    return (
        f'<tr><td style="padding:18px 28px;{border}">'
        f"{_kicker(ep.get('podcast_title') or '')}"
        f'<div style="{_S_EP_TITLE_SM};">{_esc(ep.get("title"))}</div>'
        f"{quote_html}{points_html}{open_link}</td></tr>"
    )


def _footer(unsubscribe_url: str, settings_url: str) -> str:
    return (
        f'<div style="font-size:13px;color:{_MUTED};">{_wordmark()}'
        " — a learning player for podcasts.</div>"
        '<div style="margin-top:6px;font-size:12px;color:#a7afa9;">'
        "You’re getting the daily recap because you turned it on. "
        f'<a href="{unsubscribe_url}" style="color:{_MUTED};">Unsubscribe</a> · '
        f'<a href="{settings_url}" style="color:{_MUTED};">Notification settings</a></div>'
    )


def _text_version(day: str, headline: str, episodes: list[dict[str, Any]], origin: str) -> str:
    lines = [headline, f"Your day, recapped — {day}", ""]
    for ep in episodes:
        lines.append(f"• {ep.get('title')} ({ep.get('podcast_title') or ''})")
        q = ep.get("signature_quote")
        if q:
            who = f" — {q.get('speaker')}" if q.get("speaker") else ""
            lines.append(f'  “{q.get("text")}”{who}')
        for p in (ep.get("key_points") or [])[:2]:
            lines.append(f"  - {p}")
        lines.append(f"  {origin.rstrip('/')}{ep.get('deep_link') or ''}")
        lines.append("")
    return "\n".join(lines).strip()


def render_daily_recap(
    payload: dict[str, Any], *, origin: str, unsubscribe_url: str, settings_url: str
) -> RenderedEmail:
    """Render the daily recap adaptively — full for one episode, a compact stack for several."""
    episodes = list(payload.get("episodes") or [])
    day = str(payload.get("day") or "")
    count = int(payload.get("count") or len(episodes))
    headline = "We took notes for you"
    if count == 1:
        title = episodes[0].get("title") if episodes else ""
        subject = f"Your recap: {title}".strip()
        subline = "You finished 1 episode today."
        body = f'<tr><td style="padding:22px 28px;">{_episode_full(episodes[0], origin)}</td></tr>'
    else:
        subject = f"Your day, recapped — {count} episodes"
        subline = f"You finished {count} episodes today."
        body = "".join(
            _episode_compact(ep, origin, last=(i == len(episodes) - 1))
            for i, ep in enumerate(episodes)
        )
    doc = _shell(_masthead(day, headline, subline), body, _footer(unsubscribe_url, settings_url))
    return RenderedEmail(subject, doc, _text_version(day, headline, episodes, origin))


def _render_sections_digest(
    payload: dict[str, Any],
    *,
    origin: str,
    unsubscribe_url: str,
    settings_url: str,
    subject: str,
) -> RenderedEmail:
    """Minimal renderer for the existing sections-shaped digests (your-week / recommendations).

    Keeps those email types deliverable through the same worker; the daily recap is the designed
    surface. Each item links to its deep link.
    """
    rows: list[str] = []
    text_lines: list[str] = [subject, ""]
    for section in payload.get("sections") or []:
        rows.append(
            f'<tr><td style="padding:14px 28px 4px;">{_kicker(str(section.get("kind") or ""))}'
            "</td></tr>"
        )
        for item in section.get("items") or []:
            href = _abs(origin, item.get("deep_link"))
            label = item.get("quote") or item.get("episode_slug") or "Open"
            rows.append(
                '<tr><td style="padding:2px 28px 10px;font-size:15px;line-height:1.5;">'
                f'<a href="{href}" style="color:{_ACCENT};text-decoration:none;">{_esc(label)}</a>'
                "</td></tr>"
            )
            text_lines.append(f"- {label}: {origin.rstrip('/')}{item.get('deep_link') or ''}")
    doc = _shell(_masthead("", subject, ""), "".join(rows), _footer(unsubscribe_url, settings_url))
    return RenderedEmail(subject, doc, "\n".join(text_lines).strip())


def render_email(
    envelope: dict[str, Any], *, origin: str, unsubscribe_url: str, settings_url: str
) -> RenderedEmail:
    """Dispatch an email envelope to its template renderer. Raises on an unknown email template."""
    template = str(envelope.get("template") or "")
    payload = envelope.get("payload") or {}
    if template == "daily-recap.v1":
        return render_daily_recap(
            payload, origin=origin, unsubscribe_url=unsubscribe_url, settings_url=settings_url
        )
    if template == "your-week-digest.v1":
        return _render_sections_digest(
            payload,
            origin=origin,
            unsubscribe_url=unsubscribe_url,
            settings_url=settings_url,
            subject="Your Week",
        )
    if template == "recommendations-digest.v1":
        return _render_sections_digest(
            payload,
            origin=origin,
            unsubscribe_url=unsubscribe_url,
            settings_url=settings_url,
            subject="New for you",
        )
    raise UnknownTemplateError(template)
