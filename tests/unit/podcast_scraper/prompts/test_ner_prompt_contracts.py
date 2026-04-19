"""NER prompt contracts: guest_host is metadata-grounded; system_ner is instruction-only."""

from __future__ import annotations

from podcast_scraper.prompts.store import get_prompt_dir

_TRANSCRIPT = "{{ transcript }}"


def test_ner_guest_host_templates_reference_episode_title() -> None:
    root = get_prompt_dir()
    missing_title: list[str] = []
    missing_description_token: list[str] = []
    for path in sorted(root.glob("**/ner/guest_host_v1.j2")):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        if "{{ episode_title }}" not in text:
            missing_title.append(rel)
        if "{{ episode_description }}" not in text:
            missing_description_token.append(rel)
    assert not missing_title, f"guest_host templates missing episode_title: {missing_title}"
    assert (
        not missing_description_token
    ), f"guest_host templates missing episode_description placeholder: {missing_description_token}"


def test_ner_system_templates_do_not_embed_transcript_placeholder() -> None:
    root = get_prompt_dir()
    offenders: list[str] = []
    for path in sorted(root.glob("**/ner/system_ner_v1.j2")):
        text = path.read_text(encoding="utf-8")
        if _TRANSCRIPT in text:
            offenders.append(path.relative_to(root).as_posix())
    assert not offenders, (
        "system_ner templates must not include transcript placeholder "
        f"(metadata/system path only): {offenders}"
    )
