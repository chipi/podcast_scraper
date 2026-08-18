"""Guards for feed-served show artwork.

The corpus lost every image once already: artwork existed only as a post-hoc patch applied to a
BUILT corpus, so a rebuild that regenerated metadata dropped it and nothing noticed. The repair
(2026-08-16) makes the FEED the source of truth — the pipeline downloads art over HTTP during the
run, like production — and these tests keep it that way.

What each test would have caught:
  * ``test_every_advertised_cover_exists``   — the original defect: 8 feeds advertised
    ``/images/pNN_cover.jpg`` and no such file was ever produced, so every real pipeline run
    404'd on artwork.
  * ``test_generator_is_deterministic``      — art regenerating differently each run, which turns
    every corpus rebuild into a spurious diff.
  * ``test_committed_art_is_current``        — a feed title changing without its cover being
    regenerated, so the picture and the show disagree.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
from defusedxml.ElementTree import parse as safe_parse

pytestmark = [pytest.mark.unit]

ROOT = Path(__file__).resolve().parents[3]
RSS_DIR = ROOT / "tests" / "fixtures" / "rss"
SCRIPT = ROOT / "scripts" / "build_fixture_cover_art.py"
_ITUNES_NS = "{http://www.itunes.com/dtds/podcast-1.0.dtd}"
_HREF = re.compile(r"^/images/(?P<stem>[A-Za-z0-9_\-]+)\.(?P<ext>[A-Za-z0-9]+)$")


def _version() -> str:
    return (ROOT / "tests" / "fixtures" / "FIXTURES_VERSION").read_text(encoding="utf-8").strip()


def _images_dir() -> Path:
    return ROOT / "tests" / "fixtures" / "images" / _version()


def _advertised() -> list[tuple[str, str, str]]:
    """(feed filename, cover stem, extension) for every feed that advertises artwork."""
    out: list[tuple[str, str, str]] = []
    for xml_path in sorted(RSS_DIR.glob("*.xml")):
        channel = safe_parse(xml_path).getroot().find("channel")
        if channel is None:
            continue
        image = channel.find(f"{_ITUNES_NS}image")
        if image is None:
            continue
        match = _HREF.match(str(image.get("href") or "").strip())
        if match is None:
            continue
        out.append((xml_path.name, match.group("stem"), match.group("ext")))
    return out


def _load_script():
    spec = importlib.util.spec_from_file_location("_fixture_cover_art", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestAdvertisedArtworkExists:
    def test_some_feed_advertises_artwork(self) -> None:
        # Guards the guard: if the href pattern ever changes, the tests below would
        # vacuously pass over an empty list.
        assert _advertised(), "no feed advertises <itunes:image href='/images/...'>"

    def test_every_advertised_cover_exists(self) -> None:
        missing = [
            f"{feed} -> /images/{stem}.{ext}"
            for feed, stem, ext in _advertised()
            if not (_images_dir() / f"{stem}.{ext}").is_file()
        ]
        assert not missing, (
            "feeds advertise cover art that does not exist, so a real pipeline run 404s on "
            f"artwork: {missing}. Run: python scripts/build_fixture_cover_art.py"
        )

    def test_advertised_extension_is_servable(self) -> None:
        """The mock server's /images/ route only serves .svg (see _SUBDIR_EXTENSIONS)."""
        bad = [f"{feed} -> .{ext}" for feed, _, ext in _advertised() if ext != "svg"]
        assert not bad, f"mock server serves only .svg from /images/; these would 403/404: {bad}"


class TestGenerator:
    def test_committed_art_is_current(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert (
            result.returncode == 0
        ), f"committed cover art is stale or missing:\n{result.stdout}\n{result.stderr}"

    def test_generator_is_deterministic(self) -> None:
        mod = _load_script()
        targets = mod._feed_art_targets(RSS_DIR)  # noqa: SLF001
        assert targets, "generator found no feeds to render"

        art = mod._load_renderer(ROOT)  # noqa: SLF001
        feed_ids = sorted({fid for fid, _, _ in targets.values()})
        first = art._assign_palettes(feed_ids)  # noqa: SLF001
        second = art._assign_palettes(feed_ids)  # noqa: SLF001
        assert first == second, "palette assignment is not deterministic"

        stem, (feed_id, title, desc) = sorted(targets.items())[0]
        a = art.render_cover(feed_id, title, desc, first[feed_id])
        b = art.render_cover(feed_id, title, desc, first[feed_id])
        assert a == b, f"render_cover is not deterministic for {stem}"
