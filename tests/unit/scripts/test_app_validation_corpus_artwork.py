"""The shipped corpus must carry cover art AND reference it.

Covers on disk that nothing points at is the silent failure mode: the corpus looks complete, every
artwork surface in the app falls back to a blank tile, and the apps look unfinished for a reason
that has nothing to do with the apps.

It has happened. `build_app_validation_corpus.py` rewrites every `*.metadata.json` from scratch, so
regenerating the corpus dropped the `image_url` / `image_local_relpath` that
`build_corpus_artwork.py` had patched in. The covers stayed on disk; nothing referenced them; it
was caught in visual review rather than by anything automated. The builder now applies the artwork
step itself, and this test is the guard that keeps the two in step.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

CORPUS = Path(__file__).resolve().parents[2] / "fixtures" / "app-validation-corpus" / "v3"
ART_DIR = CORPUS / ".podcast_scraper" / "corpus-art"


def _metadata_files() -> list[Path]:
    return sorted(CORPUS.glob("feeds/*/*/metadata/*.metadata.json"))


@pytest.mark.skipif(not CORPUS.is_dir(), reason="validation corpus not present")
class TestShippedCorpusArtwork:
    def test_every_show_has_a_cover_on_disk(self) -> None:
        covers = sorted(p.name for p in ART_DIR.glob("*.svg"))
        assert covers, f"no cover art in {ART_DIR}"
        feed_ids = {
            str((json.loads(m.read_text(encoding="utf-8")).get("feed") or {}).get("feed_id") or "")
            for m in _metadata_files()
        }
        feed_ids.discard("")
        missing = sorted(f"{fid}.svg" for fid in feed_ids if f"{fid}.svg" not in covers)
        assert not missing, f"shows without a cover: {missing}"

    def test_every_episode_points_at_its_show_cover(self) -> None:
        """The half that regeneration silently drops."""
        unreferenced: list[str] = []
        for m in _metadata_files():
            feed = json.loads(m.read_text(encoding="utf-8")).get("feed") or {}
            rel = str(feed.get("image_local_relpath") or "")
            if "corpus-art" not in rel:
                unreferenced.append(m.name)
        assert not unreferenced, (
            f"{len(unreferenced)} episode(s) do not reference cover art — regenerating the corpus "
            f"without the artwork step drops it: {unreferenced[:5]}"
        )

    def test_referenced_covers_actually_exist(self) -> None:
        """A dangling reference renders the same blank tile as a missing one."""
        dangling: list[str] = []
        for m in _metadata_files():
            feed = json.loads(m.read_text(encoding="utf-8")).get("feed") or {}
            rel = str(feed.get("image_local_relpath") or "")
            if rel and not (CORPUS / rel).is_file():
                dangling.append(f"{m.name} -> {rel}")
        assert not dangling, f"metadata points at covers that do not exist: {dangling[:5]}"

    def test_covers_are_non_trivial_svg(self) -> None:
        for svg in sorted(ART_DIR.glob("*.svg")):
            text = svg.read_text(encoding="utf-8")
            assert text.lstrip().startswith("<svg"), f"{svg.name} is not an SVG"
            assert len(text) > 300, f"{svg.name} looks empty ({len(text)} bytes)"
