"""Write the invented news items into the app validation corpus (News 1.0).

Run from the repo root: ``python tests/fixtures/app-validation-corpus/build_news_items.py``. It
rewrites ``v3/sources/news_article/`` from the table below, so the fixture is reproducible and
reviewed as code. Names and domains are invented (RFC-134 §12: fixtures never use real
publishers). Each item follows ``docs/architecture/corpus/news-item-1.0.md``; each KG file is KG
3.0 and is validated strictly here. Every body holds ``NEWS-BODY-MARKER``, which no served
response may contain (RFC-134 §10). One topic and one person are shared with the podcasts so the
views that list both types have something to show.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from podcast_scraper.content_types import url_identity  # noqa: E402
from podcast_scraper.kg.schema import validate_artifact  # noqa: E402

CORPUS = Path(__file__).resolve().parent / "v3"
OUT = CORPUS / "sources" / "news_article"
RUN = "run_20261117-090000"
BODY_MARKER = "NEWS-BODY-MARKER"
INGESTED_AT = "2026-11-17T09:00:00Z"

SOURCES = {
    "daily-example": {
        "url": "https://www.dailyexample.test/feed.xml",
        "title": "Daily Example",
        "publisher": "org:daily-example",
    },
    "link-board": {"url": "https://linkboard.test/api/front", "title": "Link Board"},
}

ORGS = {"org:daily-example": "Daily Example", "org:other-wire": "Other Wire"}

ITEMS: List[Dict[str, Any]] = [
    {
        "source": "daily-example",
        "stem": "risk-desks",
        "url": "https://www.dailyexample.test/2026/11/risk-desks-learn-to-sit-still",
        "title": "Risk desks learn to sit still",
        "published_at": "2026-11-17T07:30:00Z",
        "language": "en",
        "serveable": "summary",
        "summary": "Trading desks say the hard part of risk management is behavioural, not maths.",
        "byline": ["Ana Example"],
        "editorial_type": "analysis",
        "text_status": "full",
        "paywalled": False,
        "topics": ["topic:risk-management", "topic:trading-desks"],
        "mentions": ["person:skanda-amarnath"],
    },
    {
        "source": "daily-example",
        "stem": "soil-report",
        "url": "https://www.dailyexample.test/2026/11/soil-report",
        "title": "Soil report finds erosion slowing",
        "published_at": "2026-11-16T12:00:00Z",
        "language": "en",
        "serveable": "metadata",
        "summary": "Erosion slowed on three of five test plots.",
        "byline": ["Ana Example", "Cy Example"],
        "editorial_type": "reporting",
        "text_status": "teaser",
        "paywalled": True,
        "topics": ["topic:soil-erosion"],
        "mentions": [],
    },
    {
        "source": "daily-example",
        "stem": "withheld",
        "url": "https://www.dailyexample.test/2026/11/withheld",
        "title": "WITHHELD-ITEM-TITLE",
        "published_at": "2026-11-17T08:00:00Z",
        "language": "en",
        "serveable": "none",
        "summary": "Never shown.",
        "byline": [],
        "editorial_type": None,
        "text_status": "full",
        "paywalled": False,
        "topics": ["topic:risk-management"],
        "mentions": [],
    },
    {
        "source": "daily-example",
        "stem": "mercados",
        "url": "https://www.dailyexample.test/es/2026/11/mercados",
        "title": "Los mercados y la gestión del riesgo",
        "published_at": "2026-11-15T09:00:00Z",
        "language": "es",
        "serveable": "summary",
        "summary": "Una columna sobre la gestión del riesgo en los mercados.",
        "byline": ["Dora Ejemplo"],
        "editorial_type": "opinion",
        "text_status": "full",
        "paywalled": False,
        "topics": ["topic:risk-management"],
        "mentions": [],
    },
    {
        "source": "link-board",
        "stem": "wire-reliability",
        "url": "https://m.otherwire.test/reliability-at-scale",
        "title": "Reliability at scale, a field report",
        "published_at": "2026-11-17T06:00:00Z",
        "language": "en",
        "serveable": "summary",
        "summary": "Operators describe how they keep systems reliable as they grow.",
        "byline": ["Bo Example"],
        "editorial_type": None,
        "text_status": "full",
        "paywalled": None,
        "topics": ["topic:reliability"],
        "mentions": ["person:skanda-amarnath"],
    },
    {
        "source": "link-board",
        "stem": "trail-video",
        "url": "https://nowhere.test/videos/trail-building",
        "title": "Trail building, in pictures",
        "published_at": "2026-11-14T18:00:00Z",
        "language": "en",
        "serveable": "metadata",
        "summary": "A video about trail building.",
        "byline": [],
        "editorial_type": None,
        "text_status": "video",
        "paywalled": None,
        "topics": ["topic:trail-building"],
        "mentions": [],
    },
]


def _id20(url: str) -> str:
    return hashlib.sha256(url_identity(url).encode("utf-8")).hexdigest()[:20]


def source_id(name: str) -> str:
    return f"source:{_id20(SOURCES[name]['url'])}"


def item_id(item: Dict[str, Any]) -> str:
    return f"item:news_article:{_id20(item['url'])}"


def _body(item: Dict[str, Any], n: int) -> str:
    return (
        f"{item['title']}\n\n{BODY_MARKER}-{n} The first paragraph of the article body.\n\n"
        "A second paragraph that no surface may show.\n"
    )


def _metadata(item: Dict[str, Any], body: str) -> Dict[str, Any]:
    src = SOURCES[item["source"]]
    doc: Dict[str, Any] = {
        "content_type": "news_article",
        "item_id": item_id(item),
        "url": url_identity(item["url"]),
        "source_id": source_id(item["source"]),
        "title": item["title"],
        "published_at": item["published_at"],
        "ingested_at": INGESTED_AT,
        "language": item["language"],
        "licence": None,
        "paywalled": item["paywalled"],
        "serveable": item["serveable"],
        "mineable": True,
        "summary": item["summary"],
        "fetched_at": INGESTED_AT,
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "news_article": {
            "transport_ids": {"rss" if item["source"] == "daily-example" else "api": item["stem"]},
            "byline": item["byline"],
            "byline_raw": " and ".join(item["byline"]) or None,
            "editorial_type": item["editorial_type"],
            "text_status": item["text_status"],
        },
    }
    if "publisher" in src:
        doc["publisher_id"] = src["publisher"]
    elif "otherwire.test" in item["url"]:
        doc["publisher_id"] = "org:other-wire"
    return doc


def _slug_label(node_id: str) -> str:
    return node_id.split(":", 1)[1].replace("-", " ").title()


def _kg(item: Dict[str, Any], doc: Dict[str, Any]) -> Dict[str, Any]:
    src = SOURCES[item["source"]]
    iid, sid = doc["item_id"], doc["source_id"]
    nodes: List[Dict[str, Any]] = [
        {
            "id": sid,
            "type": "Source",
            "properties": {
                "title": src["title"],
                "source_kind": "news",
                "transport": "rss" if item["source"] == "daily-example" else "json_api",
                "url": src["url"],
            },
        },
        {
            "id": iid,
            "type": "ContentItem",
            "properties": {
                "title": item["title"],
                "content_type": "news_article",
                "source_id": sid,
                "url": doc["url"],
                "published_at": item["published_at"],
            },
        },
    ]
    edges: List[Dict[str, Any]] = [{"type": "HAS_ITEM", "from": sid, "to": iid}]
    if "publisher" in src:
        org = src["publisher"]
        nodes.append({"id": org, "type": "Organization", "properties": {"name": ORGS[org]}})
        edges.append({"type": "PUBLISHED_BY", "from": sid, "to": org})
    for tid in item["topics"]:
        slug = tid.split(":", 1)[1]
        nodes.append(
            {"id": tid, "type": "Topic", "properties": {"label": _slug_label(tid), "slug": slug}}
        )
        edges.append({"type": "MENTIONS", "from": tid, "to": iid})
    for name in item["byline"]:
        pid = "person:" + name.lower().replace(" ", "-")
        nodes.append({"id": pid, "type": "Person", "properties": {"name": name}})
        edges.append(
            {"type": "PARTICIPATES_IN", "from": pid, "to": iid, "properties": {"role": "author"}}
        )
    for pid in item["mentions"]:
        nodes.append({"id": pid, "type": "Person", "properties": {"name": _slug_label(pid)}})
        edges.append({"type": "MENTIONS", "from": pid, "to": iid})
    art = {
        "schema_version": "3.0",
        "item_id": iid,
        "content_type": "news_article",
        "extraction": {
            "model_version": "metadata_only",
            "extracted_at": INGESTED_AT,
            "document_ref": f"transcripts/{item['stem']}.txt",
            "variant": "raw",
        },
        "nodes": nodes,
        "edges": edges,
    }
    validate_artifact(art, strict=True)
    return art


def build() -> List[Path]:
    shutil.rmtree(OUT, ignore_errors=True)
    written: List[Path] = []
    for n, item in enumerate(ITEMS, start=1):
        run = OUT / item["source"] / RUN
        (run / "metadata").mkdir(parents=True, exist_ok=True)
        (run / "transcripts").mkdir(parents=True, exist_ok=True)
        body = _body(item, n)
        doc = _metadata(item, body)
        files = {
            run / "transcripts" / f"{item['stem']}.txt": body,
            run / "metadata" / f"{item['stem']}.metadata.json": json.dumps(doc, indent=2) + "\n",
            run / "metadata" / f"{item['stem']}.kg.json": json.dumps(_kg(item, doc), indent=2)
            + "\n",
        }
        for path, text in files.items():
            path.write_text(text, encoding="utf-8")
            written.append(path)
    return written


#: The fixture's publishers, for the registry the e2e server is given.
PUBLISHERS = {
    "publishers": [
        {
            "id": "org:daily-example",
            "name": "Daily Example",
            "domains": ["dailyexample.test"],
            "sources": [source_id("daily-example")],
        },
        {"id": "org:other-wire", "name": "Other Wire", "domains": ["otherwire.test"]},
    ]
}

if __name__ == "__main__":
    import yaml

    paths = build()
    reg = Path(__file__).resolve().parent / "publishers.fixture.yaml"
    reg.write_text(
        "# The app validation corpus's publishers (invented); written by build_news_items.py.\n"
        + yaml.safe_dump(PUBLISHERS, sort_keys=False),
        encoding="utf-8",
    )
    print(f"{len(paths)} files under {OUT.relative_to(ROOT)}; registry {reg.relative_to(ROOT)}")
