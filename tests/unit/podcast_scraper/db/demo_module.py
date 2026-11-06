"""A module that owns one table, for the module-history tests (ADR-172)."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Column, ForeignKey, Integer, MetaData, Table, Text

from podcast_scraper.extensions import DbModule, Extension

metadata = MetaData()

demo_items = Table(
    "demo_items",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Text, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False),
    Column("body", Text, nullable=False),
)

SCRIPTS = Path(__file__).with_name("demo_module_history")


def extension(**module_overrides: object) -> Extension:
    fields: dict[str, object] = {
        "name": "demo",
        "scripts": SCRIPTS,
        "metadata": metadata,
        "planes": ("player",),
    }
    fields.update(module_overrides)
    return Extension(name="demo", database=DbModule(**fields))  # type: ignore[arg-type]
