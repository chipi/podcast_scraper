#!/usr/bin/env python3
"""Generate the INTERNAL-build app icon, and the per-platform files that use it (#2193).

Why this exists
---------------
The operator installs two copies of this app on one phone: the TestFlight/Play build that testers
get, and a dev build pointed at a private API. They are the same app id family, the same name and
the same icon, so on a home screen they are indistinguishable — and tapping the wrong one while
debugging a tester's report wastes the whole investigation.

So the internal build gets a differently-coloured icon. Deliberately the SAME artwork, recoloured,
rather than a badge or a different mark: the operator asked for "slightly different colour and
that's it", and a recoloured disc reads as "the other one" at a glance without looking like a
broken or unofficial build.

What it does
------------
Desaturates and slightly darkens the coloured disc of ``assets/icon.png``, leaving its hue alone:
the vivid #FF6A3D becomes a muted dusty rose. The operator chose this from six rendered candidates
(2026-09-29) — a hue rotation to teal or violet was unmistakable but stopped looking like the
product, and the brief was "slightly different colour and that's it".

Only pixels with saturation > 0.25 are touched, so the near-black ground and the cream bars are
untouched by construction rather than by luck.

Outputs, all committed so a build never depends on this script having been run:

* ``assets/icon-internal.png`` — the 1024² source, for regenerating the rest.
* ``ios/App/App/Assets.xcassets/AppIcon-Internal.appiconset/`` — selected by the fastlane ``device``
  lane via ``ASSETCATALOG_COMPILER_APPICON_NAME``. Both iOS lanes build the *Release* configuration,
  so the Xcode config cannot be the discriminator; the lane has to name the asset explicitly.
* ``android/app/src/debug/res/mipmap-*/`` — Android picks these up automatically for the ``debug``
  build type, which is what ``make android-build`` / ``android-device-install`` produce. The store
  artifact comes from ``bundleRelease`` and is untouched.

Run: ``python scripts/tools/make_internal_icon.py`` (needs Pillow).
"""

from __future__ import annotations

import colorsys
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "web" / "learning-player"
SRC = APP / "assets" / "icon.png"
OUT_SRC = APP / "assets" / "icon-internal.png"

IOS_SET = APP / "ios/App/App/Assets.xcassets/AppIcon-Internal.appiconset"
ANDROID_DEBUG_RES = APP / "android/app/src/debug/res"

# Keep the hue; drain the colour. The disc stays recognisably the same mark and reads as a
# washed-out sibling of the shipped one rather than a different brand.
SATURATION = 0.45
VALUE = 0.85
# Below this saturation a pixel is the near-black ground or a cream bar — leave those exactly as
# they are, so the two icons differ in precisely one way.
SATURATION_FLOOR = 0.25

# The densities Capacitor generated, and the files in each. Sizes are read from the originals
# rather than hardcoded, so this stays correct if the icon set is ever regenerated at new sizes.
ANDROID_DENSITIES = ("mdpi", "hdpi", "xhdpi", "xxhdpi", "xxxhdpi")
ANDROID_FILES = ("ic_launcher.png", "ic_launcher_round.png", "ic_launcher_foreground.png")


def mute(img: Image.Image) -> Image.Image:
    """Desaturate + darken the saturated pixels, preserving hue and alpha."""
    img = img.convert("RGBA")
    px = img.load()
    assert px is not None
    for y in range(img.height):
        for x in range(img.width):
            r, g, b, a = px[x, y]
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            if s > SATURATION_FLOOR:
                s = min(1.0, s * SATURATION)
                v = min(1.0, v * VALUE)
            nr, ng, nb = colorsys.hsv_to_rgb(h, s, v)
            px[x, y] = (round(nr * 255), round(ng * 255), round(nb * 255), a)
    return img


def main() -> int:
    if not SRC.exists():
        print(f"FAIL: no source icon at {SRC}", file=sys.stderr)
        return 1

    base = Image.open(SRC)
    tinted = mute(base)
    OUT_SRC.parent.mkdir(parents=True, exist_ok=True)
    tinted.convert("RGB").save(OUT_SRC)
    print(f"OK: {OUT_SRC.relative_to(REPO)}")

    # --- iOS: one 1024² entry, mirroring the shape of the real AppIcon.appiconset ---
    IOS_SET.mkdir(parents=True, exist_ok=True)
    tinted.convert("RGB").resize((1024, 1024), Image.LANCZOS).save(IOS_SET / "AppIcon-512@2x.png")
    (IOS_SET / "Contents.json").write_text(
        "{\n"
        '  "images" : [\n'
        "    {\n"
        '      "filename" : "AppIcon-512@2x.png",\n'
        '      "idiom" : "universal",\n'
        '      "platform" : "ios",\n'
        '      "size" : "1024x1024"\n'
        "    }\n"
        "  ],\n"
        '  "info" : {\n'
        '    "author" : "xcode",\n'
        '    "version" : 1\n'
        "  }\n"
        "}\n"
    )
    print(f"OK: {IOS_SET.relative_to(REPO)}")

    # --- Android: the debug build type's own mipmaps, at the sizes main/ already uses ---
    main_res = APP / "android/app/src/main/res"
    written = 0
    for density in ANDROID_DENSITIES:
        src_dir = main_res / f"mipmap-{density}"
        dst_dir = ANDROID_DEBUG_RES / f"mipmap-{density}"
        if not src_dir.is_dir():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in ANDROID_FILES:
            origin = src_dir / name
            if not origin.exists():
                continue
            with Image.open(origin) as ref:
                size = ref.size
                has_alpha = ref.mode in ("RGBA", "LA") or "transparency" in ref.info
            out = tinted.resize(size, Image.LANCZOS)
            # `ic_launcher_foreground` is drawn over the adaptive background and must keep its
            # transparency; the legacy square/round icons are opaque.
            out.save(dst_dir / name) if has_alpha else out.convert("RGB").save(dst_dir / name)
            written += 1
    print(f"OK: {ANDROID_DEBUG_RES.relative_to(REPO)} ({written} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
