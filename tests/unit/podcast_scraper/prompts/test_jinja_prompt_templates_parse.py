"""Every committed Jinja prompt template must parse (syntax-only, offline)."""

from __future__ import annotations

from jinja2 import Template
from jinja2.exceptions import TemplateSyntaxError

from podcast_scraper.prompts.store import get_prompt_dir


def test_all_prompt_templates_parse_as_jinja() -> None:
    root = get_prompt_dir()
    paths = sorted(root.rglob("*.j2"))
    assert paths, f"expected *.j2 under {root}"
    failures: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root).as_posix()
        try:
            Template(text)
        except TemplateSyntaxError as exc:
            failures.append(f"{rel}: {exc}")
    assert not failures, "Jinja TemplateSyntaxError:\n" + "\n".join(failures)
