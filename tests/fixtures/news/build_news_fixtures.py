"""Write the news writer's invented fixtures: a source's RSS feed, its pages and its robots.txt.

Run from the repo root: ``python tests/fixtures/news/build_news_fixtures.py``. The source and its
items are the ones the app validation corpus holds (``app-validation-corpus/build_news_items.py``:
same URLs, titles, times, bylines; each RSS guid is the item's stem), so the writer over these
fixtures produces those items. Names and domains are invented (RFC-134 §12). Every body holds
``NEWS-BODY-MARKER-<n>``, which no served response may contain.
"""

from __future__ import annotations

from datetime import datetime
from email.utils import format_datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
OUT = HERE / "daily-example"
FEED_URL = "https://www.dailyexample.test/feed.xml"
BODY_MARKER = "NEWS-BODY-MARKER"

#: Paragraphs long enough to be a full article (the writer's teaser floor is 600 characters).
FILLER_EN = [
    "Desk heads described the same pattern: the losses that hurt most came from trades that were"
    " right on paper and wrong in timing, closed in a hurry by people who could not sit still.",
    "Several firms now rehearse quiet days the way pilots rehearse failures, with a checklist for"
    " what not to do when the screens go red and the phones start ringing at once.",
    "The models changed little over the decade, one risk officer said; what changed was how long"
    " traders were willing to wait before acting on what the models already told them.",
    "None of the firms would say what the rehearsals cost, and none would say what they saved.",
]
FILLER_ES = [
    "Los responsables de las mesas describen el mismo patrón: las pérdidas más dolorosas vienen de"
    " operaciones correctas en el papel y equivocadas en el momento, cerradas con prisa.",
    "Varias firmas ensayan ahora los días tranquilos como los pilotos ensayan las averías, con una"
    " lista de lo que no hay que hacer cuando las pantallas se ponen en rojo.",
    "Los modelos cambiaron poco en una década; lo que cambió fue cuánto tiempo estaban dispuestos a"
    " esperar los operadores antes de actuar sobre lo que los modelos ya decían.",
    "Ninguna de las firmas quiso decir cuánto cuestan los ensayos, ni cuánto ahorran.",
]

ITEMS: List[Dict[str, Any]] = [
    {
        "n": 1,
        "stem": "risk-desks",
        "url": "https://www.dailyexample.test/2026/11/risk-desks-learn-to-sit-still",
        "title": "Risk desks learn to sit still",
        "published_at": "2026-11-17T07:30:00Z",
        "lang": "en",
        "author": "Ana Example",
        "paragraphs": FILLER_EN,
    },
    {
        "n": 2,
        "stem": "soil-report",
        "url": "https://www.dailyexample.test/2026/11/soil-report",
        "title": "Soil report finds erosion slowing",
        "published_at": "2026-11-16T12:00:00Z",
        "lang": "en",
        "author": "Ana Example and Cy Example",
        # A paywalled page: only the opening shows, so the writer marks it a teaser.
        "paragraphs": [],
    },
    {
        "n": 3,
        "stem": "withheld",
        "url": "https://www.dailyexample.test/2026/11/withheld",
        "title": "WITHHELD-ITEM-TITLE",
        "published_at": "2026-11-17T08:00:00Z",
        "lang": "en",
        "author": None,
        "paragraphs": FILLER_EN,
    },
    {
        "n": 4,
        "stem": "mercados",
        "url": "https://www.dailyexample.test/es/2026/11/mercados",
        "title": "Los mercados y la gestión del riesgo",
        "published_at": "2026-11-15T09:00:00Z",
        "lang": "es",
        "author": "Dora Ejemplo",
        "paragraphs": FILLER_ES,
    },
]

ROBOTS = "User-agent: *\nDisallow: /private/\n"


def _rfc2822(iso: str) -> str:
    return format_datetime(datetime.fromisoformat(iso.replace("Z", "+00:00")))


def feed() -> str:
    """The source's RSS 2.0 feed, newest item first."""
    entries = []
    for item in sorted(ITEMS, key=lambda i: i["published_at"], reverse=True):
        author = (
            f"\n      <dc:creator>{escape(item['author'])}</dc:creator>" if item["author"] else ""
        )
        entries.append(f"""    <item>
      <title>{escape(item['title'])}</title>
      <link>{item['url']}</link>
      <guid isPermaLink="false">{item['stem']}</guid>
      <pubDate>{_rfc2822(item['published_at'])}</pubDate>{author}
      <description>{escape(item['title'])}.</description>
    </item>""")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        "  <channel>\n"
        "    <title>Daily Example</title>\n"
        "    <link>https://www.dailyexample.test/</link>\n"
        "    <description>Invented news for tests.</description>\n"
        "    <language>en-gb</language>\n" + "\n".join(entries) + "\n  </channel>\n</rss>\n"
    )


def page(item: Dict[str, Any]) -> str:
    """An article page: site navigation and footer around the article."""
    opening = f"{BODY_MARKER}-{item['n']} The opening paragraph of {item['title'].lower()}."
    paragraphs = [opening, *item["paragraphs"]]
    body = "\n".join(f"      <p>{escape(p)}</p>" for p in paragraphs)
    paywall = (
        '\n      <div class="paywall"><p>Subscribe to read the rest.</p></div>'
        if not item["paragraphs"]
        else ""
    )
    byline = f'<p class="byline">By {escape(item["author"])}</p>' if item["author"] else ""
    return f"""<!doctype html>
<html lang="{item['lang']}">
  <head>
    <meta charset="utf-8">
    <title>{escape(item['title'])} | Daily Example</title>
    <link rel="canonical" href="{item['url']}">
  </head>
  <body>
    <nav><a href="/">Home</a> <a href="/business">Business</a> <a href="/science">Science</a></nav>
    <article>
      <h1>{escape(item['title'])}</h1>
      {byline}
{body}{paywall}
    </article>
    <footer><p>Daily Example is an invented publication. All rights reserved.</p></footer>
  </body>
</html>
"""


def build() -> List[Path]:
    """Write the feed, one page per item and robots.txt under ``daily-example/``."""
    (OUT / "pages").mkdir(parents=True, exist_ok=True)
    files = {OUT / "feed.xml": feed(), OUT / "robots.txt": ROBOTS}
    for item in ITEMS:
        files[OUT / "pages" / f"{item['stem']}.html"] = page(item)
    for path, text in files.items():
        path.write_text(text, encoding="utf-8")
    return list(files)


def routes() -> Dict[str, Path]:
    """URL -> fixture file, for a test's fake network."""
    out = {
        FEED_URL: OUT / "feed.xml",
        "https://www.dailyexample.test/robots.txt": OUT / "robots.txt",
    }
    for item in ITEMS:
        out[item["url"]] = OUT / "pages" / f"{item['stem']}.html"
    return out


if __name__ == "__main__":
    written = build()
    print(f"{len(written)} files under {OUT.relative_to(HERE.parents[2])}")
