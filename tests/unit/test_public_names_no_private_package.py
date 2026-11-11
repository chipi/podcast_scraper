"""This repository names no private package or repository (ADR-170: "This repository does not name
them").

Public code reaches Common, Player and the other private packages only through extension entry
points and hooks (ADR-168, ADR-176), never by name. Nothing enforced the rule before this test.

Allowed: the product's domain and the bare namespace prefix (``scripts/measure/api_memory.py``
attributes memory to any package starting with it). Refused: a private package or repository named
in full, in any tracked text file.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
# Built from parts so this file does not match itself.
_BRAND = "close" + "listening"
PRIVATE_NAME = re.compile(rf"\b{_BRAND}[_-][a-z]", re.IGNORECASE)
BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".mp3",
    ".wav",
    ".m4a",
    ".ogg",
    ".pdf",
    ".gz",
    ".zip",
    ".woff",
    ".woff2",
    ".ttf",
    ".lance",
    ".npy",
    ".pkl",
    ".db",
    ".sqlite",
}


def _tracked_text_files() -> list[Path]:
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout: the tracked files cannot be listed")
    out = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True
    ).stdout.decode("utf-8")
    return [
        ROOT / p for p in out.split("\0") if p and Path(p).suffix.lower() not in BINARY_SUFFIXES
    ]


def test_no_tracked_file_names_a_private_package() -> None:
    offenders = []
    for path in _tracked_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            m = PRIVATE_NAME.search(line)
            if m:
                offenders.append(f"{path.relative_to(ROOT)}:{n}: {m.group(0)}…")
    assert not offenders, "private package or repository named in public:\n" + "\n".join(offenders)


@pytest.mark.parametrize(
    ("text", "refused"),
    [
        (f"from {_BRAND}_common.identity import ledger", True),
        (f"see chipi/{_BRAND}-player#22", True),
        (f"APP_PACKAGES = ('podcast_scraper', '{_BRAND}_')", False),
        (f"https://{_BRAND}.app/mcp", False),
    ],
)
def test_the_pattern_refuses_names_and_allows_the_prefix_and_domain(
    text: str, refused: bool
) -> None:
    assert bool(PRIVATE_NAME.search(text)) is refused
