"""Symlink-escape safety for the artwork / photo / logo serving helpers (#2036 review round 3).

`is_file()` and the string-level `safe_relpath_under_corpus_root` both FOLLOW symlinks, so a link
placed inside the corpus art/photo/logo store pointing OUTSIDE the corpus would otherwise be served.
Each serving helper now also checks `resolves_under_root` (realpath). These tests lock that in and
confirm real files still serve.
"""

from __future__ import annotations

import os
from pathlib import Path

from podcast_scraper.enrichment.enrichers.org_web import _logo_dir, org_logo_path
from podcast_scraper.enrichment.enrichers.person_web import _image_dir, person_image_path
from podcast_scraper.server.app_artwork import safe_artwork_target
from podcast_scraper.utils.corpus_artwork import CORPUS_ART_REL_PREFIX


def _write(path: Path, data: bytes = b"img") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# ── safe_artwork_target ─────────────────────────────────────────────────────────────────────
def test_safe_artwork_target_allows_a_real_file(tmp_path: Path) -> None:
    art = _write(tmp_path / CORPUS_ART_REL_PREFIX / "p01.webp")
    got = safe_artwork_target(tmp_path, f"{CORPUS_ART_REL_PREFIX}/p01.webp")
    assert got == os.path.normpath(str(art))


def test_safe_artwork_target_rejects_outside_the_art_prefix(tmp_path: Path) -> None:
    _write(tmp_path / "enrichments" / "secret.json", b"{}")
    assert safe_artwork_target(tmp_path, "enrichments/secret.json") is None


def test_safe_artwork_target_rejects_dotdot(tmp_path: Path) -> None:
    assert safe_artwork_target(tmp_path, f"{CORPUS_ART_REL_PREFIX}/../../etc/passwd") is None


def test_safe_artwork_target_rejects_a_symlink_escaping_the_corpus(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    (root / CORPUS_ART_REL_PREFIX).mkdir(parents=True)
    secret = _write(tmp_path / "outside" / "secret.webp", b"SECRET")
    (root / CORPUS_ART_REL_PREFIX / "evil.webp").symlink_to(secret)
    assert safe_artwork_target(root, f"{CORPUS_ART_REL_PREFIX}/evil.webp") is None


def test_safe_artwork_target_allows_a_symlink_that_stays_inside(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    store = root / CORPUS_ART_REL_PREFIX
    store.mkdir(parents=True)
    real = _write(store / "real.webp")
    (store / "alias.webp").symlink_to(real)
    assert safe_artwork_target(root, f"{CORPUS_ART_REL_PREFIX}/alias.webp") is not None


# ── person_image_path / org_logo_path ───────────────────────────────────────────────────────
def test_person_image_path_rejects_symlink_escape_but_serves_real(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    d = _image_dir(root)
    d.mkdir(parents=True)
    secret = _write(tmp_path / "outside" / "jane.jpg", b"SECRET")
    (d / "jane.jpg").symlink_to(secret)
    assert person_image_path(root, "person:jane") is None  # symlink escape → rejected
    real = _write(d / "bob.jpg")
    got = person_image_path(root, "person:bob")
    assert got is not None and got[0] == real


def test_org_logo_path_rejects_symlink_escape_but_serves_real(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    d = _logo_dir(root)
    d.mkdir(parents=True)
    secret = _write(tmp_path / "outside" / "acme.png", b"SECRET")
    (d / "acme.png").symlink_to(secret)
    assert org_logo_path(root, "org:acme") is None
    real = _write(d / "globex.png")
    got = org_logo_path(root, "org:globex")
    assert got is not None and got[0] == real
