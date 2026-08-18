#!/usr/bin/env python3
"""Synthesize deterministic cover art for the app-validation corpus's shows (#1619 follow-up).

Why this exists
---------------
Every feed in ``app-validation-corpus`` shipped with ``image_url: null`` and
``image_local_relpath: null``, so every consumer surface that renders artwork — Catalog cards,
Home rails, the show header, Your Week — fell back to text. The apps looked unfinished in review
for a reason that had nothing to do with the apps: the fixture simply had no pictures.

The plumbing already existed on both ends. The API resolves
``artwork_url(image_local_relpath, size)`` and serves the bytes from
``<corpus>/.podcast_scraper/corpus-art/``; the mock feed server already exposes a ``/images/``
location. Only the images and the two metadata fields were missing.

Why SVG rather than JPEG/PNG
---------------------------
* **No new dependency.** Pillow is not installed here, and ``app_artwork.ensure_thumbnail``
  explicitly falls back to serving the original file with a guessed mimetype when Pillow is
  missing or cannot decode the image — and ``mimetypes`` maps ``.svg`` to ``image/svg+xml``. So
  this works whether or not Pillow is present.
* **A committed fixture should be reviewable.** These files live in git forever. As text, a change
  to the artwork shows up as a readable diff instead of an opaque binary blob.
* **Resolution independence.** One file serves the 48px catalog thumb and the full-bleed player
  header without a derived-thumbnail cache.

If byte-realistic raster art is ever wanted (to exercise the Pillow thumbnail path itself), this
script is the place to add it — it would need Pillow as a dev dependency.

Determinism
-----------
Colours derive from a SHA-256 of the feed id, so a given show always gets the same cover and
re-running this never produces a spurious diff. No randomness.

Usage
-----
    python scripts/build_corpus_artwork.py [--corpus tests/fixtures/app-validation-corpus/v3]
    python scripts/build_corpus_artwork.py --check     # verify only, non-zero exit if stale
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ART_REL_PREFIX = ".podcast_scraper/corpus-art"
#: The mock feed server already serves this location; using it keeps the fixture honest about
#: where a real feed's artwork would come from.
MOCK_IMAGE_BASE = "http://localhost/images"


#: Curated gradient pairs. A hash-driven hue produces muddy colours as often as good ones, so the
#: palette is hand-picked and only the *choice* is deterministic. Each entry is (from, to).
PALETTES: list[tuple[str, str]] = [
    ("#F0603A", "#7A1E52"),  # ember → plum
    ("#2E9E8F", "#123A54"),  # teal → deep sea
    ("#3B6FE0", "#101C4A"),  # azure → midnight
    ("#C2417A", "#3A1250"),  # magenta → violet
    ("#E0A32E", "#6E2312"),  # amber → rust
    ("#5B8C3A", "#12331F"),  # moss → forest
    ("#8A5BD6", "#1B1145"),  # iris → indigo
    ("#D9524C", "#2A1030"),  # coral → aubergine
    ("#20A4C9", "#0D2B3E"),  # cyan → petrol
    ("#B8823A", "#2B1A0E"),  # bronze → coffee
]


def _assign_palettes(feed_ids: list[str]) -> dict[str, tuple[str, str, str]]:
    """Give every show a DIFFERENT colourway, deterministically.

    Hashing each id independently collides: with 10 palettes and 9 shows the birthday problem makes
    repeats near-certain, and the first pass shipped four amber covers that were hard to tell apart
    in a catalog scroll — which defeats the point of colour-coding shows at all.

    So the palette is chosen by position in the sorted feed list. Stable for a fixed corpus, and
    collision-free while there are no more shows than palettes. The hash still picks the starting
    offset, so the set of colours is not always "the first N in the list".
    """
    ids = sorted(feed_ids)
    seed = hashlib.sha256("|".join(ids).encode("utf-8")).digest()[0]
    out: dict[str, tuple[str, str, str]] = {}
    for i, fid in enumerate(ids):
        c_from, c_to = PALETTES[(seed + i) % len(PALETTES)]
        out[fid] = (c_from, c_to, "#FFFFFF")
    return out


#: Subject icons, keyed by feed id, then by keyword against the show description. Line art at a
#: 0..100 viewBox so the caller can scale/translate it freely. An icon does the recognition work at
#: thumbnail size, where a title — however large — is still only a few pixels tall.
_ICON_PATHS: dict[str, str] = {
    "bike": (
        '<circle cx="24" cy="70" r="19"/><circle cx="76" cy="70" r="19"/>'
        '<path d="M24 70 L44 34 L64 70 M44 34 L38 22 M32 22 H50 M64 70 L54 34 H70"/>'
    ),
    "systems": (
        '<rect x="16" y="16" width="68" height="20" rx="5"/>'
        '<rect x="16" y="42" width="68" height="20" rx="5"/>'
        '<rect x="16" y="68" width="68" height="20" rx="5"/>'
        '<circle cx="28" cy="26" r="3.2" fill="currentColor" stroke="none"/>'
        '<circle cx="28" cy="52" r="3.2" fill="currentColor" stroke="none"/>'
        '<circle cx="28" cy="78" r="3.2" fill="currentColor" stroke="none"/>'
    ),
    "scuba": (
        '<path d="M18 42 a32 26 0 0 1 64 0 v10 a14 14 0 0 1 -14 14 h-8 l-8 10 l-8 -10 h-8 '
        'a14 14 0 0 1 -14 -14 z"/><path d="M50 20 v22"/>'
        '<circle cx="78" cy="20" r="6"/><circle cx="90" cy="34" r="3.5"/>'
    ),
    "camera": (
        '<path d="M12 32 h18 l8 -10 h24 l8 10 h18 a6 6 0 0 1 6 6 v40 a6 6 0 0 1 -6 6 '
        'H12 a6 6 0 0 1 -6 -6 V38 a6 6 0 0 1 6 -6 z"/>'
        '<circle cx="50" cy="58" r="18"/><circle cx="50" cy="58" r="8"/>'
    ),
    "chart": (
        '<path d="M12 88 V26 M12 88 H92"/>'
        '<path d="M26 72 V54 M44 78 V38 M62 66 V28 M80 74 V44"/>'
        '<path d="M22 62 L44 30 L62 46 L88 18"/>'
    ),
    "waves": (
        '<path d="M8 34 q14 -14 28 0 t28 0 t28 0"/>'
        '<path d="M8 56 q14 -14 28 0 t28 0 t28 0"/>'
        '<path d="M8 78 q14 -14 28 0 t28 0 t28 0"/>'
    ),
    "horizon": (
        '<circle cx="70" cy="30" r="13"/>' '<path d="M6 82 L34 40 L54 68 L68 52 L94 82 Z"/>'
    ),
    "mic": (
        '<rect x="38" y="10" width="24" height="44" rx="12"/>'
        '<path d="M26 46 a24 24 0 0 0 48 0"/><path d="M50 70 V88 M34 88 H66"/>'
    ),
    "crossshow": (
        '<path d="M22 36 a30 30 0 0 1 52 -6"/><path d="M78 34 h-14 v-14"/>'
        '<path d="M78 64 a30 30 0 0 1 -52 6"/><path d="M22 66 h14 v14"/>'
        '<circle cx="50" cy="50" r="7"/>'
    ),
    "default": ('<path d="M14 50 h10 l8 -22 l10 44 l10 -32 l8 20 h26"/>'),
}

#: description keyword → icon key. First match wins, so order matters.
_ICON_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("mountain bik", "cycling", "trail"), "bike"),
    (("scuba", "diving", "underwater", "marine"), "scuba"),
    (("photograph", "camera", "image"), "camera"),
    (("investing", "risk", "portfolio", "market"), "chart"),
    (("software", "reliability", "architecture", "engineering"), "systems"),
    (("sustainab", "future", "systems thinking"), "horizon"),
    (("public-radio", "public radio", "npr", "radio"), "mic"),
    (("recurring guests", "cross-show", "revisit"), "crossshow"),
    (("meandering", "long-form", "dialogue"), "waves"),
]


def _icon_for(title: str, description: str) -> str:
    """Pick a subject icon from the show's own description — the thing that survives at 120px."""
    hay = f"{title} {description}".lower()
    for keywords, key in _ICON_KEYWORDS:
        if any(k in hay for k in keywords):
            return _ICON_PATHS[key]
    return _ICON_PATHS["default"]


#: Skipped when abbreviating a show name — "Below the Surface" should read BS, not BT.
_STOPWORDS = {"the", "a", "an", "of", "and", "for", "on", "in", "&"}


def _initials(title: str) -> str:
    """Up to two initials from a show title, ignoring articles."""
    words = [w for w in title.replace("&", " ").split() if w and w[0].isalnum()]
    significant = [w for w in words if w.lower().strip(".,") not in _STOPWORDS] or words
    if not significant:
        return "?"
    if len(significant) == 1:
        return significant[0][:2].upper()
    return (significant[0][0] + significant[1][0]).upper()


def _wrap(title: str, width: int) -> list[str]:
    """Greedy wrap so long show names stay inside the cover instead of overflowing it."""
    out: list[str] = []
    line = ""
    for word in title.split():
        candidate = f"{line} {word}".strip()
        if len(candidate) <= width:
            line = candidate
        else:
            if line:
                out.append(line)
            line = word
    if line:
        out.append(line)
    return out[:3]


def _fit_title(title: str) -> tuple[list[str], int]:
    """Pick the largest type size at which the title still fits — big names, readable thumbs.

    Sized against a 600px canvas so the cover survives being drawn at ~120px in a catalog row:
    at that scale a 96px cap-height renders around 19px on screen, which is legible; the 40px it
    used to be rendered at 8px, which is not. That was the actual complaint.
    """
    for size, per_line in ((96, 9), (78, 11), (64, 14), (52, 17)):
        lines = _wrap(title, per_line)
        if len(lines) <= 3 and all(len(line) <= per_line for line in lines):
            return lines, size
    return _wrap(title, 17), 52


def render_cover(
    feed_id: str,
    title: str,
    description: str = "",
    palette: tuple[str, str, str] | None = None,
) -> str:
    """A 600×600 SVG cover: rich gradient, subject icon, and the show name set large."""
    c_from, c_to, ink = palette or _assign_palettes([feed_id])[feed_id]
    icon = _icon_for(title, description)
    lines, size = _fit_title(title)
    leading = int(size * 1.06)

    # Title block sits on the baseline grid from the bottom up, above the accent rule.
    last_baseline = 508
    start_y = last_baseline - (len(lines) - 1) * leading
    tspans = "".join(
        f'<tspan x="52" y="{start_y + i * leading}">{_esc(line)}</tspan>'
        for i, line in enumerate(lines)
    )

    # Source-wrapped with adjacent literals, NOT reflowed: the emitted bytes are compared
    # byte-for-byte by `--check` against the committed covers, so a newline inside this tag would
    # mark every one of them stale.
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 600" width="600" '
        f'height="600" role="img" aria-label="{_esc(title)}">\n'
        f"""\
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{c_from}"/>
      <stop offset="55%" stop-color="{c_from}" stop-opacity="0.55"/>
      <stop offset="100%" stop-color="{c_to}"/>
    </linearGradient>
    <radialGradient id="glow" cx="0.72" cy="0.22" r="0.75">
      <stop offset="0%" stop-color="{ink}" stop-opacity="0.30"/>
      <stop offset="100%" stop-color="{ink}" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="shade" x1="0" y1="0" x2="0" y2="1">
      <stop offset="45%" stop-color="#000" stop-opacity="0"/>
      <stop offset="100%" stop-color="#000" stop-opacity="0.55"/>
    </linearGradient>
  </defs>

  <rect width="600" height="600" fill="{c_to}"/>
  <rect width="600" height="600" fill="url(#bg)"/>
  <rect width="600" height="600" fill="url(#glow)"/>

  <!-- Subject mark: the part that still reads at 120px. -->
  <g transform="translate(300 34) scale(2.32)" fill="none" stroke="{ink}" stroke-opacity="0.92"
     stroke-width="4.2" stroke-linecap="round" stroke-linejoin="round" color="{ink}">
    <g transform="translate(-50 0)">{icon}</g>
  </g>

  <!-- Bottom scrim keeps the title legible over any part of the gradient. -->
  <rect width="600" height="600" fill="url(#shade)"/>

  <text fill="{ink}" font-family="Inter, Helvetica, Arial, system-ui, sans-serif" font-size="{size}"
        font-weight="800" letter-spacing="-1.5">{tspans}</text>
  <rect x="52" y="{last_baseline + 30}" width="88" height="7" rx="3.5" fill="{ink}" opacity="0.9"/>
  <text x="548" y="{last_baseline + 36}" fill="{ink}" opacity="0.55" text-anchor="end"
        font-family="Inter, Helvetica, Arial, system-ui, sans-serif" font-size="26"
        font-weight="700" letter-spacing="2">{_esc(_initials(title))}</text>
</svg>
"""
    )


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _feed_titles(corpus: Path) -> dict[str, tuple[str, str]]:
    """Map feed_id → (display title, description), read from the corpus's own episode metadata.

    The description drives icon selection, so a show that talks about scuba gets a mask rather than
    a generic waveform.
    """
    titles: dict[str, tuple[str, str]] = {}
    for meta in sorted(corpus.glob("feeds/*/*/metadata/*.metadata.json")):
        try:
            doc = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a malformed fixture file should not abort the run
            continue
        feed = doc.get("feed")
        if not isinstance(feed, dict):
            continue
        fid = str(feed.get("feed_id") or doc.get("feed_id") or "").strip()
        title = str(feed.get("title") or feed.get("display_title") or "").strip()
        desc = str(feed.get("description") or "").strip()
        if fid and title and fid not in titles:
            titles[fid] = (title, desc)
    return titles


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="tests/fixtures/app-validation-corpus/v3")
    ap.add_argument(
        "--check", action="store_true", help="verify without writing; non-zero if stale"
    )
    args = ap.parse_args()

    corpus = Path(args.corpus).resolve()
    if not corpus.is_dir():
        print(f"corpus not found: {corpus}", file=sys.stderr)
        return 2

    titles = _feed_titles(corpus)
    if not titles:
        print("no feed titles found in corpus metadata", file=sys.stderr)
        return 2

    art_dir = corpus / ART_REL_PREFIX
    stale: list[str] = []
    if not args.check:
        art_dir.mkdir(parents=True, exist_ok=True)

    palettes = _assign_palettes(list(titles))
    for fid, (title, desc) in sorted(titles.items()):
        svg = render_cover(fid, title, desc, palettes[fid])
        dst = art_dir / f"{fid}.svg"
        if args.check:
            if not dst.is_file() or dst.read_text(encoding="utf-8") != svg:
                stale.append(str(dst.relative_to(corpus)))
        else:
            dst.write_text(svg, encoding="utf-8")

    # Point every episode's feed block at the cover.
    patched = 0
    for meta in sorted(corpus.glob("feeds/*/*/metadata/*.metadata.json")):
        try:
            doc = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        feed = doc.get("feed")
        if not isinstance(feed, dict):
            continue
        fid = str(feed.get("feed_id") or doc.get("feed_id") or "").strip()
        if fid not in titles:
            continue
        want_local = f"{ART_REL_PREFIX}/{fid}.svg"
        want_remote = f"{MOCK_IMAGE_BASE}/{fid}.svg"
        if feed.get("image_local_relpath") == want_local and feed.get("image_url") == want_remote:
            continue
        if args.check:
            stale.append(str(meta.relative_to(corpus)))
            continue
        feed["image_local_relpath"] = want_local
        feed["image_url"] = want_remote
        meta.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        patched += 1

    if args.check:
        if stale:
            print(f"artwork is stale ({len(stale)} item(s)); run scripts/build_corpus_artwork.py")
            for s in stale[:10]:
                print(f"  {s}")
            return 1
        print(f"artwork up to date for {len(titles)} shows")
        return 0

    print(
        f"wrote {len(titles)} covers to {art_dir.relative_to(corpus)}; "
        f"patched {patched} metadata files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
