"""Registry ↔ config consistency invariants (#1176, upgrade-framework CI net B).

These tests catch the failure modes the path-based migration guard (workflow A)
cannot: a mismatched producer / reader range, a mis-numbered migration id, a
missing ``to_version`` or ``id``, or a registry that is not sorted. They are
fast (<10 ms), deterministic, and part of ``make ci-fast`` via the standard
unit-test discovery.

Together with the migration-guard workflow and the fixture-pinned end-to-end
test, these tests are the "framework knows about itself" net. See
``docs/guides/CORPUS_UPGRADE.md`` → "When is a migration required?" for the
authoring contract.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from podcast_scraper.upgrade.migration import Migration
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
FORMAT_CFG = REPO_ROOT / "config" / "corpus_snapshot_format.json"
READER_CFG = REPO_ROOT / "config" / "corpus_snapshot_reader_support.json"

# Migration ids are ``mNNNN_<snake_name>`` where NNNN is a zero-padded 4-digit
# integer. Enforce this at test time so a fat-finger like ``m3_foo`` fails fast.
_ID_PATTERN = re.compile(r"^[0-9]{4}_[a-z0-9][a-z0-9_]*[a-z0-9]$")

# Simple SemVer-ish check on Migration.to_version (major.minor.patch). We do not
# use ``packaging.version`` here so this test stays dependency-free.
_TO_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


def _load_json(path: Path) -> dict:
    doc: dict = json.loads(path.read_text(encoding="utf-8"))
    return doc


# ---------- format + reader config invariants ------------------------------


def test_producer_format_is_self_readable() -> None:
    """The producer's own ``corpus_format_version`` must fall within the reader's
    supported range, otherwise a freshly produced corpus would fail its own
    reader-range check on restore.
    """
    producer = _load_json(FORMAT_CFG)
    reader = _load_json(READER_CFG)

    fmt = producer["corpus_format_version"]
    lo = reader["supported_corpus_format_version_min"]
    hi = reader["supported_corpus_format_version_max"]

    assert isinstance(fmt, int) and isinstance(lo, int) and isinstance(hi, int)
    assert lo <= fmt <= hi, (
        f"producer corpus_format_version={fmt} outside reader range [{lo},{hi}]. "
        "A bump on one config file must be matched on the other — this is the "
        "single most common upgrade-framework drift."
    )


def test_reader_range_is_ordered_and_positive() -> None:
    reader = _load_json(READER_CFG)
    lo = reader["supported_corpus_format_version_min"]
    hi = reader["supported_corpus_format_version_max"]
    assert lo >= 1, f"reader min must be ≥ 1, got {lo}"
    assert hi >= lo, f"reader range must be non-empty and ordered: [{lo},{hi}]"


def test_producer_schema_version_is_1() -> None:
    """Manifest schema is v1 (RFC-084). Bumping schema_version is a separate
    concern from bumping corpus_format_version and requires a coordinated migration
    of every reader — assert at 1 until we make that move deliberately.
    """
    producer = _load_json(FORMAT_CFG)
    assert producer["schema_version"] == 1


# ---------- registry invariants -------------------------------------------


def test_registry_is_non_empty() -> None:
    assert get_migrations(), "registry has no migrations — the framework is unusable"


def test_registry_ids_are_unique() -> None:
    ids = [m.id for m in get_migrations()]
    assert len(ids) == len(set(ids)), f"duplicate migration ids: {ids}"


def test_registry_ids_match_naming_convention() -> None:
    for m in get_migrations():
        assert _ID_PATTERN.match(m.id), (
            f"migration id {m.id!r} does not match mNNNN_<snake_name> convention "
            f"(pattern: {_ID_PATTERN.pattern}). Rename the class attribute."
        )


def test_registry_is_sorted_by_id() -> None:
    """``get_migrations`` returns lex-sorted; upstream reordering would silently
    change apply order. Guard against that.
    """
    ids = [m.id for m in get_migrations()]
    assert ids == sorted(ids), (
        f"registry order {ids} != sorted order {sorted(ids)}. The runner "
        "trusts lexicographic order — a mis-sort would apply migrations "
        "out of dependency order."
    )


def test_every_migration_declares_metadata() -> None:
    """Every registered migration must set id + to_version + description so the
    ledger + status output can identify it. A default-empty attribute would render
    a mystery entry.
    """
    for m in get_migrations():
        assert m.id, f"migration {type(m).__name__} has empty id"
        assert m.to_version, f"migration {m.id} has empty to_version"
        assert _TO_VERSION_PATTERN.match(m.to_version), (
            f"migration {m.id} has malformed to_version {m.to_version!r} "
            f"(expected major.minor.patch)"
        )
        assert m.description, f"migration {m.id} has empty description"


def test_every_migration_subclasses_Migration_base() -> None:
    """The runner expects the ``apply`` / ``verify`` signature from ``Migration``.
    Registering something that only duck-types would work at runtime but drift
    silently the first time we extend the base class.
    """
    for m in get_migrations():
        assert isinstance(
            m, Migration
        ), f"registered object {type(m).__name__} is not a Migration subclass"


def test_to_version_is_non_decreasing_across_registry() -> None:
    """``to_version`` should not decrease as we walk migrations in apply order.
    A regression here means someone renumbered a migration but forgot to align
    the target version — future readers would see the ledger jump backwards.
    """

    def _semver_tuple(v: str) -> tuple[int, int, int]:
        parts = v.split(".")
        return int(parts[0]), int(parts[1]), int(parts[2])

    prev: tuple[int, int, int] | None = None
    for m in get_migrations():
        cur = _semver_tuple(m.to_version)
        if prev is not None:
            assert cur >= prev, f"migration {m.id} moves to_version backwards from {prev} to {cur}"
        prev = cur
