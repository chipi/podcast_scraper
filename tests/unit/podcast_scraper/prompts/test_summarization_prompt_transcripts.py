"""Per-provider summarization templates must embed ``{{ transcript }}`` when user-facing."""

from __future__ import annotations

from podcast_scraper.prompts.store import get_prompt_dir
from tests.unit.podcast_scraper.prompts.prompt_contract_registry import (
    SUMMARIZATION_TRANSCRIPT_EXCLUDE_FILENAMES,
)


def test_non_shared_summarization_templates_include_transcript() -> None:
    root = get_prompt_dir()
    missing: list[str] = []
    for path in sorted(root.glob("**/summarization/*.j2")):
        rel = path.relative_to(root).as_posix()
        if rel.startswith("shared/"):
            continue
        if path.name in SUMMARIZATION_TRANSCRIPT_EXCLUDE_FILENAMES:
            continue
        text = path.read_text(encoding="utf-8")
        if "{{ transcript }}" not in text:
            missing.append(rel)
    assert not missing, (
        "These summarization templates omit {{ transcript }} "
        f"(or add basename to SUMMARIZATION_TRANSCRIPT_EXCLUDE_FILENAMES): {missing}"
    )


def test_summarization_exclude_list_basenames_exist_on_disk() -> None:
    root = get_prompt_dir()
    for name in sorted(SUMMARIZATION_TRANSCRIPT_EXCLUDE_FILENAMES):
        hits = list(root.glob(f"**/summarization/{name}"))
        assert hits, f"Excluded basename {name!r} not found under prompts/**/summarization/"
