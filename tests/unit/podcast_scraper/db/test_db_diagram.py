"""The committed database diagram matches the table definitions (ADR-169).

When a migration changes a table, this fails until the diagram is regenerated:
``python -m podcast_scraper.db.diagram --out docs/architecture/diagrams/database-kernel.md``.
"""

from __future__ import annotations

from pathlib import Path

from podcast_scraper.db import diagram
from podcast_scraper.db.schema import metadata, shared_metadata

REPO = Path(__file__).resolve().parents[4]
KERNEL_DIAGRAM = REPO / "docs" / "architecture" / "diagrams" / "database-kernel.md"


def test_the_kernel_diagram_is_current() -> None:
    assert diagram.main(["--check", str(KERNEL_DIAGRAM)]) == 0, (
        "the kernel diagram is stale: run "
        "`python -m podcast_scraper.db.diagram --out docs/architecture/diagrams/database-kernel.md`"
    )


def test_every_table_and_foreign_key_is_in_it() -> None:
    text = KERNEL_DIAGRAM.read_text(encoding="utf-8")
    for table in [*metadata.tables.values(), *shared_metadata.tables.values()]:
        label = table.name if table.schema is None else f"{table.schema}.{table.name}"
        assert f'["{label}"]' in text, label
        for fk in table.foreign_keys:
            assert f"`{fk.parent.name}` → `{fk.target_fullname}`" in text, (label, fk)


def test_a_changed_column_makes_the_check_fail(tmp_path: Path) -> None:
    stale = tmp_path / "diagram.md"
    stale.write_text(
        KERNEL_DIAGRAM.read_text(encoding="utf-8").replace("text email", "text mail"),
        encoding="utf-8",
    )
    assert diagram.main(["--check", str(stale)]) == 1
