"""Every shipped verdicts file covers its whole frozen list, one row per entry.

``reviewer_verdicts.parse`` accepts a file with rows missing: a missing row reads as "not
reviewed", and for 0034 / 0035 that applies nothing, the same as a kept voice. So a deleted kept
row changed no behaviour and survived every committed test (step-8 round 6 of 0034, 2026-10-27).
This pins the shipped files: one row for each list entry, none left out, and every decided row
carries its deciding segment and line hash.
"""

from __future__ import annotations

import pytest

from podcast_scraper.upgrade import reviewer_verdicts as rv
from podcast_scraper.upgrade.migrations import (
    m0034_talk_floor_names_removed as m34,
    m0035_voice_identity_names as m35,
    m0037_named_voices_corrected as m37,
)

pytestmark = [pytest.mark.unit]


def _names_34(entries):
    return {k: str(e[0]) for k, e in entries.items()}


def _names_35(entries):
    return {k: str(e[2] or e[0]) for k, e in entries.items()}


@pytest.mark.parametrize(
    "mod, names_of, kwargs",
    [
        (m34, _names_34, {}),
        (m35, _names_35, {"bases": m35.BASES}),
        (m37, _names_35, {"bases": m35.BASES}),
    ],
    ids=["0034", "0035", "0037"],
)
def test_the_verdicts_file_has_one_row_for_every_list_entry(mod, names_of, kwargs) -> None:
    entries = mod.load_list(mod.LIST_FILE)
    parsed = rv.load(
        mod.MIGRATION_ID, names_of(entries), listed=entries, require_basis=True, **kwargs
    )
    missing = sorted(set(entries) - set(parsed.rows))
    assert not missing, f"{len(missing)} list entries without a verdict row, e.g. {missing[:3]}"
    assert len(parsed.rows) == len(entries)
    for key, row in parsed.rows.items():
        if row.basis != rv.UNDECIDABLE:
            assert row.segment_idx is not None and len(row.line_sha256) == 64, key
