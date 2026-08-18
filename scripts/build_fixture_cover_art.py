#!/usr/bin/env python3
"""Synthesize the show cover art that the RSS fixtures advertise, so the PIPELINE can fetch it.

Why this exists (2026-08-16)
----------------------------
Every feed under ``tests/fixtures/rss/`` carries ``<itunes:image href="/images/pNN_cover.svg">``,
but nothing ever produced those files and the E2E mock server had no ``/images/`` route. So a real
pipeline run against the fixtures 404'd on artwork for every show:

    WARNING podcast_scraper.rss.downloader: Failed to fetch
    http://127.0.0.1:18765/images/p01_cover.jpg: Client error '404 File not found'

The corpus got pictures anyway, but only because ``build_corpus_artwork.py`` writes them into a
built corpus and patches the metadata afterwards. That is a post-hoc repair, not the production
path: in production the artwork arrives *from the feed*, over HTTP, during the run. The difference
is exactly why the corpus lost every image once — a rebuild regenerated metadata and there was no
feed-side source of truth to restore it from.

This script closes that gap. It renders art from the FEEDS (which exist before any corpus does),
into ``tests/fixtures/images/<version>/``, where the mock server's ``/images/`` route serves it.

Rendering is delegated to ``build_corpus_artwork.render_cover`` — one renderer, two callers, so the
feed-served art and any corpus-side art cannot drift apart in style.

SVG rather than JPEG/PNG: Pillow is not a dependency here, and a committed fixture that is text
reviews as a readable diff instead of an opaque binary blob. See build_corpus_artwork.py's own
rationale, which this mirrors deliberately.

Determinism: colours derive from a SHA-256 of the feed id, so re-running never produces a spurious
diff.

Usage::

    python scripts/build_fixture_cover_art.py
    python scripts/build_fixture_cover_art.py --check   # CI: non-zero if stale/missing
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import xml.etree.ElementTree as ET  # nosec B405 — Element/ParseError typing only
from pathlib import Path
from typing import Optional

from defusedxml.ElementTree import parse as safe_parse, ParseError as DefusedParseError

_ITUNES_NS = "{http://www.itunes.com/dtds/podcast-1.0.dtd}"
# href="/images/p01_cover.svg" -> stem "p01_cover", feed id "p01"
_IMAGE_HREF = re.compile(r"^/images/(?P<stem>[A-Za-z0-9_\-]+)\.(?:svg|jpg|jpeg|png)$")


def _load_renderer(repo_root: Path):
    """Import build_corpus_artwork by path (both files are scripts, not a package)."""
    script = repo_root / "scripts" / "build_corpus_artwork.py"
    if not script.is_file():
        raise SystemExit(f"renderer not found: {script}")
    spec = importlib.util.spec_from_file_location("_corpus_artwork", script)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _feed_art_targets(rss_dir: Path) -> dict[str, tuple[str, str, str]]:
    """Map cover stem -> (feed_id, channel title, channel description).

    Keyed by the stem the FEED asks for, so what gets written is what gets requested. Several
    fixture feeds describe the same show (p01_mtb / p01_fast / p01_multi); they agree on the image
    href, so the first one that names a given stem wins and the rest are consistent duplicates.
    """
    out: dict[str, tuple[str, str, str]] = {}
    for xml_path in sorted(rss_dir.glob("*.xml")):
        try:
            channel = safe_parse(xml_path).getroot().find("channel")
        except (ET.ParseError, DefusedParseError) as exc:
            print(f"  skip {xml_path.name}: unparseable ({exc})", file=sys.stderr)
            continue
        if channel is None:
            continue
        image = channel.find(f"{_ITUNES_NS}image")
        if image is None:
            continue
        match = _IMAGE_HREF.match(str(image.get("href") or "").strip())
        if match is None:
            continue
        stem = match.group("stem")
        if stem in out:
            continue
        feed_id = stem.split("_")[0]
        title = (channel.findtext("title") or "").strip()
        desc = (channel.findtext("description") or "").strip()
        if title:
            out[stem] = (feed_id, title, desc)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rss-dir", type=Path, default=repo_root / "tests" / "fixtures" / "rss")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: tests/fixtures/images/<FIXTURES_VERSION>",
    )
    ap.add_argument(
        "--check", action="store_true", help="verify without writing; non-zero if stale"
    )
    args = ap.parse_args(argv)

    if not args.rss_dir.is_dir():
        print(f"--rss-dir does not exist: {args.rss_dir}", file=sys.stderr)
        return 2

    out_dir = args.out_dir
    if out_dir is None:
        version = (
            (repo_root / "tests" / "fixtures" / "FIXTURES_VERSION")
            .read_text(encoding="utf-8")
            .strip()
        )
        out_dir = repo_root / "tests" / "fixtures" / "images" / version

    targets = _feed_art_targets(args.rss_dir)
    if not targets:
        print(f"no <itunes:image href='/images/...'> found under {args.rss_dir}", file=sys.stderr)
        return 2

    art = _load_renderer(repo_root)
    palettes = art._assign_palettes(sorted({fid for fid, _, _ in targets.values()}))  # noqa: SLF001

    if not args.check:
        out_dir.mkdir(parents=True, exist_ok=True)

    stale: list[str] = []
    for stem, (feed_id, title, desc) in sorted(targets.items()):
        svg = art.render_cover(feed_id, title, desc, palettes[feed_id])
        dst = out_dir / f"{stem}.svg"
        if args.check:
            if not dst.is_file() or dst.read_text(encoding="utf-8") != svg:
                stale.append(dst.name)
        else:
            dst.write_text(svg, encoding="utf-8")

    if args.check:
        if stale:
            print(f"stale or missing cover art ({len(stale)}): {', '.join(stale)}", file=sys.stderr)
            print("run: python scripts/build_fixture_cover_art.py", file=sys.stderr)
            return 1
        print(f"cover art up to date ({len(targets)} shows) in {out_dir}")
        return 0

    print(f"wrote {len(targets)} covers to {out_dir}")
    for stem in sorted(targets):
        print(f"  {stem}.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
