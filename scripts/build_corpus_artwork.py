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
* **No new dependency.** Pillow is not installed here, and ``artwork.ensure_thumbnail``
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
    # Ten more, added 2026-09-30, because the corpus reached ten shows and four more languages
    # are queued (it, fr, de, pt) — at 14 shows against 10 palettes, two shows would have had to
    # share a colourway, which is the "four amber covers" failure the docstring below describes.
    #
    # NOT picked by eye. The subject icon is painted in white at 0.92 opacity over the LIGHT end
    # of the gradient, so `c_from` has to stay inside the luminance band the first ten occupy
    # (contrast-vs-white 2.22..4.85) or the icon washes out; `c_to` carries the title under a 55%
    # black scrim and only has to be dark (9.84..17.32). Each of these was solved for a target
    # contrast of 3.6 / 13.5 at a chosen hue, then verified inside both bands.
    #
    # Hues fill the gaps in the first ten (3, 13, 34, 39, 96, 172, 193, 221, 263, 333), and the
    # last four are deliberately DESATURATED: ten distinct hues is about all this scheme has, so
    # past that, separation has to come from saturation instead.
    ("#858D2D", "#2A320A"),  # chartreuse → olive
    ("#399774", "#08352C"),  # jade → pine
    ("#44984F", "#0B3616"),  # fern → deep green
    ("#6961FF", "#281589"),  # indigo → nightfall
    ("#C75CCD", "#4D125C"),  # orchid → damson
    ("#D6607C", "#59142B"),  # rose → wine
    ("#76899F", "#1F2F45"),  # slate → gunmetal
    ("#A3806B", "#44281A"),  # clay → cocoa
    ("#688D94", "#17323D"),  # steel → ink
    ("#997CAC", "#422050"),  # heather → deep plum
]


def _assign_palettes(feed_ids: list[str]) -> dict[str, tuple[str, str, str]]:
    """Give every show a DIFFERENT colourway, deterministically — and STABLY.

    Hashing each id independently collides: with 10 palettes and 9 shows the birthday problem makes
    repeats near-certain, and the first pass shipped four amber covers that were hard to tell apart
    in a catalog scroll — which defeats the point of colour-coding shows at all.

    The fix for that was to choose by position in the sorted feed list, with a hash of the whole
    list picking the starting offset. It solved collisions and introduced a worse problem: the
    seed was a function of the SET, so adding one show re-coloured every existing one. Adding
    `p10` rotated all nine other covers, and a test requires the committed art to match the
    generator, so the churn was mandatory rather than optional. A show's cover is how a person
    finds it in a catalog scroll; it should not change because a sibling was added.

    So: each show's palette is chosen by a hash of its OWN id, and collisions are resolved by
    probing forward to the next free slot, in sorted-id order. A new show takes the first slot
    free at its own hash and never displaces a show that is already placed — so existing covers
    are stable under growth, which is the property that actually matters here. Collision-free
    while shows <= palettes; past that it repeats rather than failing, and the assert says so.

    Changing the scheme re-colours all ten committed covers ONCE. That is the trade: one
    re-colour now to stop re-colouring on every future show.
    """
    ids = sorted(feed_ids)
    taken: dict[int, str] = {}
    out: dict[str, tuple[str, str, str]] = {}
    for fid in ids:
        start = hashlib.sha256(fid.encode("utf-8")).digest()[0] % len(PALETTES)
        slot = start
        for step in range(len(PALETTES)):
            slot = (start + step) % len(PALETTES)
            if slot not in taken:
                break
        else:  # pragma: no cover — every slot taken; more shows than palettes
            slot = start
        taken[slot] = fid
        c_from, c_to = PALETTES[slot]
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
#: Subject keywords, matched against the show's OWN title + description — so they have to be in
#: the show's own language. The list was English-only, and `p10` (the Spanish edition of `p01`,
#: same trail-building show) matched nothing and fell through to the generic waveform: a
#: bicycle-shaped show wearing the default icon because the words were Spanish. The same
#: English-NLP-over-non-English-text failure the rest of this arc is about, surfacing in the
#: artwork.
#:
#: Cover art cannot read the English render to avoid this — it is generated from feed metadata,
#: before anything has been translated — so the terms travel with the languages instead. When a
#: language is enabled in `config/languages.yaml`, add its words for the shows that exist in it.
_ICON_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    # es: sendero (trail), ciclismo, bicicleta · it: sentiero/sentieri · fr: sentier
    # de: weg/wege/wegebau/pfad · pt: trilha/trilhas
    (
        (
            "mountain bik",
            "cycling",
            "trail",
            "sendero",
            "ciclismo",
            "bicicleta",
            "sentier",
            "sentieri",
            "sentiero",
            "wegebau",
            "pfad",
            "radfahren",
            "trilha",
            "trilhas",
            "ciclismo",
        ),
        "bike",
    ),
    (
        (
            "scuba",
            "diving",
            "underwater",
            "marine",
            "buceo",
            "submarin",
            "immersion",
            "plongée",
            "tauchen",
            "mergulho",
        ),
        "scuba",
    ),
    (
        (
            "photograph",
            "camera",
            "image",
            "fotograf",
            "cámara",
            "camara",
            "fotografia",
            "photographie",
            "kamera",
        ),
        "camera",
    ),
    (
        (
            "investing",
            "risk",
            "portfolio",
            "market",
            "inversión",
            "inversion",
            "mercado",
            "investimento",
            "investissement",
            "anlage",
            "mercato",
            "marché",
            "markt",
        ),
        "chart",
    ),
    (
        (
            "software",
            "reliability",
            "architecture",
            "engineering",
            "ingeniería",
            "ingenieria",
            "ingegneria",
            "ingénierie",
            "technik",
        ),
        "systems",
    ),
    (
        (
            "sustainab",
            "future",
            "systems thinking",
            "sostenib",
            "futuro",
            "durabilité",
            "nachhaltig",
            "sustentab",
            "futur",
            "zukunft",
        ),
        "horizon",
    ),
    (("public-radio", "public radio", "npr", "radio"), "mic"),
    (("recurring guests", "cross-show", "revisit", "invitados recurrentes"), "crossshow"),
    (
        (
            "meandering",
            "long-form",
            "dialogue",
            "diálogo",
            "dialogo",
            "dialogo",
            "dialog",
            "diálogo",
        ),
        "waves",
    ),
]


def _icon_for(title: str, description: str) -> str:
    """Pick a subject icon from the show's own description — the thing that survives at 120px.

    Matching is substring and case-folded, over title AND description together, so a show whose
    subject is only named in its description still gets its icon. Falls back to the generic mark
    rather than guessing; see `_ICON_KEYWORDS` for why that fallback is worth watching.
    """
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


#: The icon is authored on a 0..100 grid, hung from this y, and centred on x=300 by the inner
#: `translate(-50 0)`.
_ICON_TOP = 34
_ICON_SPAN = 100
#: What the icon gets when the title leaves room — the size every one- and two-line cover uses.
_ICON_MAX_SCALE = 2.32
#: Clear air between the icon's baseline and the tallest line of the title.
_ICON_TITLE_GAP = 30
#: Cap height as a fraction of font size, for Inter at weight 800. Close enough to place a box.
_CAP_HEIGHT_RATIO = 0.72


def _icon_scale(lines: int, size: int) -> float:
    """How big the subject mark can be before it runs into the title.

    The title block is anchored at the BOTTOM (baseline 508) and grows upward, so a third line
    pushes its top edge up into the icon. At 96px that put the text top at ~237 against an icon
    reaching 266 — `Sesiones de Sendero`, `Long Horizon Notes` and `The Long View: Biohacking`
    all had type crossing the mark. The icon was a fixed 2.32 and had no idea the title had
    grown.

    So the icon takes the space the title does not need. One- and two-line covers have room to
    spare, the scale clamps at `_ICON_MAX_SCALE`, and their bytes are unchanged; only the
    three-line covers shrink. The floor keeps the mark readable at a 120px thumbnail rather than
    letting a very tall title squeeze it to nothing — if it is ever hit, the honest fix is a
    shorter show name, not a smaller icon.
    """
    leading = int(size * 1.06)
    text_top = 508 - (lines - 1) * leading - size * _CAP_HEIGHT_RATIO
    available = (text_top - _ICON_TITLE_GAP) - _ICON_TOP
    return round(min(_ICON_MAX_SCALE, max(1.0, available / _ICON_SPAN)), 2)


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
    # The icon gets whatever vertical space the title left it — see `_icon_scale`. Composed
    # here rather than inline: the tag has to stay on ONE source line (see the note below about
    # byte-for-byte `--check`), and interpolating both values there ran it past the 100-col lint.
    icon_transform = f"translate(300 {_ICON_TOP}) scale({_icon_scale(len(lines), size):g})"
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
  <g transform="{icon_transform}" fill="none" stroke="{ink}" stroke-opacity="0.92"
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
        # A committed REALISTIC cover wins over the synthesised one.
        #
        # The SVG path below stays the fallback — it is what gives any new or unadorned feed a
        # cover at all, and it remains reviewable-as-text. But a flat two-stop gradient is not
        # what the app renders in production, where feeds supply real, dense, art-directed cover
        # images. Designing or judging the artwork zone against a gradient measures the generator,
        # not the product: the INSIGHT NOW overlay is trivially legible over it, and
        # `deriveShowAccent` samples a single hue instead of a busy image. So where a real-looking
        # cover is committed as `<fid>.webp`, it is used as-is and never overwritten.
        real = art_dir / f"{fid}.webp"
        if real.is_file():
            continue
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
        ext = "webp" if (corpus / ART_REL_PREFIX / f"{fid}.webp").is_file() else "svg"
        want_local = f"{ART_REL_PREFIX}/{fid}.{ext}"
        want_remote = f"{MOCK_IMAGE_BASE}/{fid}.{ext}"
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
