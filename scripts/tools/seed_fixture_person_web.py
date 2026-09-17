#!/usr/bin/env python3
"""Seed the fixture corpus with a synthetic ``person_web`` enrichment: bio + hosted photo.

The person card renders a large photo and an external bio whenever the ``person_web`` enricher has
matched (``AppPersonCard.web``). Nothing in the fixture corpus ever populated it, so that whole
block was unreachable locally — it could only be seen against production, which needs a
session.
This makes it testable on the fixture, which is where the UI work happens.

DELIBERATELY SYNTHETIC. It does not copy production's ``person_web.json``: that carries real
people's Wikipedia bios and real hosted photographs, and real data does not belong in fixtures.
Several fixture ids are real names, so the bios here are written as obvious fixture prose tied to
the corpus's own themes, and ``source`` is ``fixture`` — the card renders "via fixture", which can
never be mistaken for sourced Wikipedia content. Photos are generated initials, not photographs.

    python scripts/tools/seed_fixture_person_web.py \\
        [--corpus tests/fixtures/app-validation-corpus/v3]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - environment guard
    sys.exit("FAIL: Pillow is required — use .venv/bin/python")

# Same derivation the enricher uses (person_web._safe_name), so the serve route finds these files.
STEM_RE = re.compile(r"[^a-z0-9._-]")

# Muted, distinguishable backgrounds — enough variety that a grid of avatars reads as different
# people at a glance, without pretending to be photography.
PALETTE = [
    (0x3B, 0x5B, 0x7A),
    (0x6B, 0x4A, 0x5E),
    (0x3F, 0x6B, 0x5A),
    (0x7A, 0x5C, 0x38),
    (0x4A, 0x4A, 0x7A),
    (0x6B, 0x3B, 0x3B),
    (0x35, 0x6B, 0x6B),
    (0x5C, 0x6B, 0x38),
]

ROLES = [
    ("systems thinking", "traces how failures propagate between coupled systems"),
    ("risk management", "argues that correlation, not component failure, is the real exposure"),
    ("lifelong learning", "studies how people retain what they hear"),
    ("safety practices", "works on the gap between written procedure and actual practice"),
    ("expert interviews", "interviews practitioners about decisions made under uncertainty"),
    ("endurance sport", "writes about pacing, recovery and long-horizon training"),
]


def safe_stem(person_id: str) -> str:
    slug = STEM_RE.sub("_", person_id.split(":", 1)[-1].lower())
    return slug or "unknown"


def display_name(person_id: str) -> str:
    raw = person_id.split(":", 1)[-1]
    parts = [p for p in raw.split("-") if p]
    out = []
    for p in parts:
        if p == "dr":
            out.append("Dr.")
        elif len(p) <= 2 and p.isalpha():
            out.append(p.upper())
        else:
            out.append(p.capitalize())
    return " ".join(out) or raw


def person_ids(corpus: Path) -> list[str]:
    """Every ``person:*`` id referenced anywhere in the corpus."""
    found: set[str] = set()
    for path in list(corpus.rglob("*.json")) + list(corpus.rglob("*.jsonl")):
        try:
            found.update(re.findall(r'"(person:[a-z0-9-]+)"', path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            continue
    return sorted(found)


def initials(name: str) -> str:
    parts = [p for p in name.replace(".", " ").split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def draw_avatar(path: Path, name: str, colour: tuple[int, int, int], size: int = 512) -> None:
    img = Image.new("RGB", (size, size), colour)
    draw = ImageDraw.Draw(img)
    text = initials(name)
    font = None
    for candidate in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        try:
            font = ImageFont.truetype(candidate, size // 3)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(
        ((size - (box[2] - box[0])) / 2 - box[0], (size - (box[3] - box[1])) / 2 - box[1]),
        text,
        fill=(0xF0, 0xF2, 0xF6),
        font=font,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "PNG")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("tests/fixtures/app-validation-corpus/v3"))
    args = ap.parse_args()
    corpus: Path = args.corpus
    if not corpus.is_dir():
        sys.exit(f"FAIL: not a corpus directory: {corpus}")

    ids = person_ids(corpus)
    if not ids:
        sys.exit(f"FAIL: no person:* ids found under {corpus}")

    image_dir = corpus / "enrichments" / "person_images"
    rows = []
    for i, pid in enumerate(ids):
        name = display_name(pid)
        topic, doing = ROLES[i % len(ROLES)]
        stem = safe_stem(pid)
        draw_avatar(image_dir / f"{stem}.png", name, PALETTE[i % len(PALETTE)])
        (image_dir / f"{stem}.image.json").write_text(
            json.dumps(
                {
                    "person_id": pid,
                    "source": "fixture",
                    "license": "CC0-1.0",
                    "artist": "generated fixture avatar",
                    "width": 512,
                    "height": 512,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "person_id": pid,
                "name": name,
                "description": f"Fixture persona — {topic}",
                "bio": (
                    f"{name} is a fixture persona used by the local validation corpus. In these "
                    f"episodes {name.split()[-1]} {doing}, and returns to {topic} as the thread "
                    f"connecting the shows they appear on. This text is synthetic: it exists "
                    f"so the person card's bio and photo block can be exercised without "
                    f"production data."
                ),
                "source": "fixture",
                "source_url": None,
                "license": "CC0-1.0",
                "image_hosted": True,
                "image_license": "CC0-1.0",
                "image_artist": "generated fixture avatar",
            }
        )

    # The executor's envelope — the reader unwraps `data` (app_relational_view._person_web_payload).
    artifact = {
        "derived": True,
        "status": "ok",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "enricher_id": "person_web",
        "enricher_version": "0.1.0",
        "schema_version": "1.0",
        "data": {"provider": "fixture", "persons": rows},
    }
    out = corpus / "enrichments" / "person_web.json"
    out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"✓ {out} ({len(rows)} people)")
    print(f"✓ {image_dir} ({len(rows)} avatars + meta)")


if __name__ == "__main__":
    main()
