"""``write_json_atomic`` must not leave artifacts owner-only.

``mkstemp`` creates its temp file 0600 and ``os.replace`` keeps the temp file's mode, so every
gi.json / kg.json written through it ended up 0600. On prod 2026-10-06 one job's finalize left
2,447 gi.json at 0600 while the rest of the corpus was 0644.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from podcast_scraper.utils import atomic_io

pytestmark = pytest.mark.unit


def _mode(p: Path) -> int:
    return stat.S_IMODE(p.stat().st_mode)


def test_a_new_file_gets_what_open_would_give_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(atomic_io, "_UMASK", 0o022)
    target = tmp_path / "ep.gi.json"
    atomic_io.write_json_atomic(target, {"a": 1})
    assert _mode(target) == 0o644


def test_a_rewrite_keeps_the_existing_mode(tmp_path) -> None:
    target = tmp_path / "ep.kg.json"
    target.write_text("{}", encoding="utf-8")
    os.chmod(target, 0o640)
    atomic_io.write_json_atomic(target, {"b": 2})
    assert _mode(target) == 0o640
    assert target.read_text(encoding="utf-8") == '{"b": 2}'


def test_the_umask_is_honoured_for_new_files(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(atomic_io, "_UMASK", 0o077)
    target = tmp_path / "private.json"
    atomic_io.write_json_atomic(target, [])
    assert _mode(target) == 0o600


def test_the_m0007_bridge_writer_keeps_permissions_too(tmp_path) -> None:
    from podcast_scraper.upgrade import rewrite_bridges_m0007

    bridge = tmp_path / "ep.bridge.json"
    bridge.write_text("{}", encoding="utf-8")
    os.chmod(bridge, 0o644)
    rewrite_bridges_m0007._write_atomic(bridge, {"identities": []})
    assert _mode(bridge) == 0o644
    assert bridge.read_text(encoding="utf-8") == '{\n  "identities": []\n}'
