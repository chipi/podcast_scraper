"""iOS kills an app that touches a privacy-protected resource without its usage description.

Not a prompt and not an error the app can catch: TCC terminates the process. The iOS app died that
way on 2026-09-15 — the profile picture is an ``<input type="file" accept="image/...">``, the iOS
picker for it always offers "Take Photo", and ``NSCameraUsageDescription`` was missing. Nothing in
the web build or the test tiers can see that, so it is held here: every resource the web layer can
reach on iOS must have its description in Info.plist.
"""

from __future__ import annotations

import plistlib
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

_APP = Path(__file__).resolve().parents[3] / "web" / "learning-player"
_PLIST = _APP / "ios" / "App" / "App" / "Info.plist"


def _plist() -> dict:
    with _PLIST.open("rb") as fh:
        return plistlib.load(fh)


def _vue_sources() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in (_APP / "src").rglob("*.vue"))


def test_an_image_file_input_requires_the_camera_description() -> None:
    image_input = re.search(r'type="file"[^>]*accept="[^"]*image/', _vue_sources(), re.S)
    assert image_input, "no image file input found — if the avatar picker moved, update this test"
    description = _plist().get("NSCameraUsageDescription", "")
    assert description.strip(), (
        "an <input type=file accept=image/...> exists, so iOS offers 'Take Photo' — without "
        "NSCameraUsageDescription that is a TCC kill (crash of 2026-09-15)"
    )


def test_note_dictation_keeps_both_of_its_descriptions() -> None:
    plist = _plist()
    for key in ("NSMicrophoneUsageDescription", "NSSpeechRecognitionUsageDescription"):
        assert str(plist.get(key, "")).strip(), f"{key} is required by note dictation"
