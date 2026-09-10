"""Full-cycle e2e for the person_web enricher (wave-G), against the mock server.

Mirrors the mock-feeds approach: the external source (Wikipedia) is served locally by the same
E2E HTTP mock, so the WHOLE cycle runs offline —

    mock Wikipedia  →  enricher FETCH (real HTTP)  →  raw cache  →  DERIVE  →  person_web.json
                                                                              →  person card bio

No live network; deterministic. This is the "does the person_web arc actually work end-to-end"
guard the unit tests (which inject a fake provider) cannot give.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from podcast_scraper.enrichment.enrichers.person_web import (
    _read_raw_cache,
    person_image_path,
    PersonWebEnricher,
    WikipediaProvider,
)
from podcast_scraper.enrichment.protocol import EpisodeArtifactBundle, RunContext
from podcast_scraper.server.app_relational_view import build_person_card

pytestmark = pytest.mark.e2e

_STEM = "0001 - ep"
_PID = "person:jane-doe"
_NAME = "Jane Doe"


def _write_corpus(root: Path) -> EpisodeArtifactBundle:
    """A one-episode corpus with a Person in both GI (enricher input) and KG (card input)."""
    meta_dir = root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    (root / "transcripts" / f"{_STEM}.txt").write_text("hello", encoding="utf-8")

    (meta_dir / f"{_STEM}.metadata.json").write_text(
        json.dumps(
            {
                "feed": {"feed_id": "f1", "title": "Show", "url": "https://pod.example/f.xml"},
                "episode": {
                    "episode_id": "e1",
                    "title": "Episode e1",
                    "published_date": "2024-03-10T00:00:00",
                    "duration_seconds": 1000,
                },
                "summary": {"title": "Sum", "bullets": ["a"]},
                "content": {"transcript_file_path": f"transcripts/{_STEM}.txt"},
            }
        ),
        encoding="utf-8",
    )
    person_node = {"id": _PID, "type": "Person", "properties": {"name": _NAME}}
    (meta_dir / f"{_STEM}.gi.json").write_text(
        json.dumps({"nodes": [person_node]}), encoding="utf-8"
    )
    (meta_dir / f"{_STEM}.kg.json").write_text(
        json.dumps(
            {
                "episode_id": "e1",
                "nodes": [
                    person_node,
                    {"id": "topic:ai", "type": "Topic", "properties": {"label": "AI"}},
                ],
            }
        ),
        encoding="utf-8",
    )
    gi = meta_dir / f"{_STEM}.gi.json"
    return EpisodeArtifactBundle(
        metadata_path=meta_dir / f"{_STEM}.metadata.json",
        gi_path=gi,
        kg_path=meta_dir / f"{_STEM}.kg.json",
        bridge_path=None,
        episode_id="e1",
        stem=_STEM,
    )


def _ctx() -> RunContext:
    return RunContext(
        run_id="r",
        parent_run_id=None,
        enricher_id="person_web",
        enricher_version="0.1.0",
        tier="web",
        attempt=1,
        job_id="r",
        cancel_event=asyncio.Event(),
    )


def test_person_web_full_cycle_against_mock(e2e_server, tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    bundle = _write_corpus(root)

    # FETCH + DERIVE + HOST PHOTO — the provider points at the mock Wikipedia (summary) + action
    # API (imageinfo) + Wikimedia image, all on the E2E server (real HTTP).
    provider = WikipediaProvider(
        summary_base=e2e_server.urls.wikipedia_summary_base(),
        api_base=e2e_server.urls.wikipedia_api_base(),
    )
    result = asyncio.run(
        PersonWebEnricher(provider=provider).enrich(
            bundle=None, corpus_root=root, all_bundles=[bundle], config={}, ctx=_ctx()
        )
    )
    assert result.status == "ok", f"{result.error_class}: {result.error}"
    assert result.data is not None
    rows = result.data["persons"]
    assert [r["person_id"] for r in rows] == [_PID]
    assert _NAME in rows[0]["bio"] and "Close Listening corpus" in rows[0]["bio"]
    assert rows[0]["source"] == "wikipedia"
    assert rows[0]["source_url"].endswith("Jane_Doe")
    # PHOTO hosted like the avatar: downloaded, validated, stored, with its OWN license.
    assert rows[0]["image_hosted"] is True
    assert rows[0]["image_ext"] == "png"
    assert rows[0]["image_license"] == "CC BY-SA 4.0"
    stored = person_image_path(root, _PID)
    assert stored is not None and stored[0].is_file() and stored[1] == "image/png"

    # RAW persisted for later re-derivation / mining.
    raw = _read_raw_cache(root, _PID)
    assert raw is not None and raw["extract"].startswith(_NAME)

    # SURFACE — write the derived output as the executor would, then the person card carries it.
    (root / "enrichments").mkdir(parents=True, exist_ok=True)
    (root / "enrichments" / "person_web.json").write_text(json.dumps(result.data), encoding="utf-8")
    card = build_person_card(root, _PID)
    assert card is not None and card.web is not None
    assert card.web.bio == rows[0]["bio"]
    assert card.web.source == "wikipedia"
    # The card exposes OUR served photo route (never the raw external URL) + the photo's license.
    assert card.web.image_url == f"/api/app/persons/{_PID}/photo"
    assert card.web.image_license == "CC BY-SA 4.0"
