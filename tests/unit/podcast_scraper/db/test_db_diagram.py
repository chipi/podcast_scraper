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


def test_a_module_diagram_renders_its_references_to_kernel_tables() -> None:
    from tests.unit.podcast_scraper.db import demo_module

    rendered = diagram.render([demo_module.metadata], "Demo")
    assert 'users ||--o{ demo_items : "user_id (cascade)"' in rendered


def test_with_kernel_resolves_a_module_reference_to_users() -> None:
    from podcast_scraper.db.migrate import with_kernel
    from tests.unit.podcast_scraper.db import demo_module

    combined = with_kernel(demo_module.metadata)
    names = [t.name for t in combined.sorted_tables]
    assert names.index("users") < names.index("demo_items")


def test_two_references_from_one_column_render_in_a_fixed_order() -> None:
    from sqlalchemy import Column, ForeignKeyConstraint, MetaData, Table, Text

    md = MetaData()
    Table("b_parent", md, Column("id", Text, primary_key=True), Column("user_id", Text))
    Table(
        "a_child",
        md,
        Column("id", Text, primary_key=True),
        Column("user_id", Text),
        Column("parent_id", Text),
        ForeignKeyConstraint(["user_id"], ["users.user_id"]),
        ForeignKeyConstraint(["parent_id", "user_id"], ["b_parent.id", "b_parent.user_id"]),
    )
    edges = [line for line in diagram.render([md], "t").splitlines() if "--o{ a_child" in line]
    user_edges = [e for e in edges if '"user_id' in e]
    assert user_edges == sorted(user_edges, key=lambda e: e.split()[0]) and len(user_edges) == 2
