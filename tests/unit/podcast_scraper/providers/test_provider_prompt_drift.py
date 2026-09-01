"""The shared GI/cleaning prompts must not drift apart between providers.

Gemini's quote prompt was hardened against duplicate quotes ("each quote must be a DIFFERENT
passage") and the other five providers never got it. Nobody chose that divergence — it just
happened, and it stayed invisible because the prompts live inline in seven separate files.

Duplicate quotes collapse into fewer distinct candidates, so the drift showed up as a candidate
supply gap and was read as a model-quality difference.

These prompts are identical by intent. Until they live in one shared template, this test is what
notices when they stop being identical.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
from collections import defaultdict
from typing import Dict, List

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
PROVIDERS_ROOT = _REPO_ROOT / "src" / "podcast_scraper" / "providers"

# prompt site -> the first line of its inline literal block
SITES = {
    "generate_insights.system": r'"Output only the list of key takeaways',
    "extract_quotes.system": r'"Extract all short verbatim quotes',
    "score_entailment.system": r'"You rate how much the premise supports',
    "cleaning.system": r'"You are a transcript cleaning assistant',
}

PROMPTS_ROOT = _REPO_ROOT / "src" / "podcast_scraper" / "prompts"

# prompt site -> the per-provider template subdir it lives in once migrated off inline literals.
# When a site is no longer inline we assert it is TEMPLATED here (not silently deleted) and that
# the copies have not drifted apart — the same "identical by intent" guarantee, moved to templates.
_SITE_TEMPLATE_DIR = {
    "generate_insights.system": "insight_extraction",
    "extract_quotes.system": "evidence/extract_quote",
    "score_entailment.system": "evidence/entailment",
    "cleaning.system": "cleaning",
}


def _templates_for(site: str) -> List[pathlib.Path]:
    """Every per-provider template file backing ``site`` (empty if none migrated yet)."""
    return sorted(PROMPTS_ROOT.glob(f"*/{_SITE_TEMPLATE_DIR[site]}/*.j2"))


def _collect_literal(lines: List[str], start: int) -> str:
    """Join the implicit-concatenation string block beginning at ``start``."""
    parts: List[str] = []
    i = start
    while i < len(lines):
        stripped = lines[i].strip()
        if not (stripped.startswith('"') or stripped.startswith("'")):
            break
        parts.append(re.sub(r'^[\'"]|[\'"],?$', "", stripped))
        i += 1
    return " ".join(parts)


def _variants() -> Dict[str, Dict[str, List[str]]]:
    found: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(PROVIDERS_ROOT.rglob("*_provider.py")):
        lines = path.read_text().splitlines()
        for site, pattern in SITES.items():
            for idx, line in enumerate(lines):
                if re.search(pattern, line):
                    text = _collect_literal(lines, idx)
                    # Non-security digest: a short fingerprint to group identical prompt literals.
                    digest = hashlib.sha1(text.encode(), usedforsecurity=False).hexdigest()[:8]
                    found[site][digest].append(path.parent.name)
                    break
    return found


@pytest.mark.parametrize("site", sorted(SITES))
def test_shared_prompt_has_no_drift(site: str) -> None:
    variants = _variants().get(site, {})
    if not variants:
        # Not inline anymore — the desired end state ("no inline prompts"). Assert it genuinely
        # moved to a shared template (not silently deleted) and the per-provider copies agree, so
        # the "identical by intent" guarantee still holds — templated instead of skipped.
        templates = _templates_for(site)
        assert templates, (
            f"{site} is neither inline nor templated under */{_SITE_TEMPLATE_DIR[site]}/ — the "
            "prompt vanished; restore it inline or as a shared template."
        )
        by_version: Dict[str, set] = defaultdict(set)
        for path in templates:
            by_version[path.name].add(hashlib.sha256(path.read_bytes()).hexdigest())
        drifted = {ver: len(digests) for ver, digests in by_version.items() if len(digests) > 1}
        assert not drifted, (
            f"{site} templates drifted across providers: {drifted}. Identical by intent — "
            "update every provider's copy together."
        )
        return

    if len(variants) > 1:
        detail = "\n".join(
            f"  [{digest}] {', '.join(sorted(provs))}" for digest, provs in sorted(variants.items())
        )
        pytest.fail(
            f"{site} has drifted into {len(variants)} versions across providers:\n{detail}\n"
            "These prompts are identical by intent. Update every provider, or move the prompt "
            "into a shared template so drift becomes impossible."
        )


def test_extract_quotes_prompt_forbids_duplicate_passages() -> None:
    """The anti-duplicate rule is load-bearing for candidate supply — keep it wherever the prompt
    lives (inline today for some providers, templated for the rest)."""
    inline_texts = set(_texts("extract_quotes.system"))
    if inline_texts:
        for text in inline_texts:
            assert "DIFFERENT passage" in text
            assert "never repeat" in text.lower()
        return
    # Migrated to templates — assert the rule survived in every extract_quote template.
    templates = _templates_for("extract_quotes.system")
    assert templates, "extract_quotes prompt is neither inline nor templated"
    for path in templates:
        text = path.read_text()
        assert "DIFFERENT passage" in text, f"{path} lost the anti-duplicate rule"
        assert "never repeat" in text.lower(), f"{path} lost 'never repeat'"


def test_insight_extraction_templates_are_identical_across_providers() -> None:
    """One prompt, seven copies. They must stay byte-identical."""
    paths = sorted(PROMPTS_ROOT.glob("*/insight_extraction/v2.j2"))
    assert len(paths) >= 7, f"expected a v2 template per provider, found {len(paths)}"
    digests = {hashlib.sha256(p.read_bytes()).hexdigest(): p.parent.parent.name for p in paths}
    assert len(digests) == 1, (
        "insight_extraction/v2 has drifted between providers: "
        f"{sorted(digests.values())}. Update every provider, or move it to a shared template."
    )


def test_insight_extraction_prompt_sets_a_value_bar_not_a_quota() -> None:
    """The count must come from the episode, not from us — but the ceiling must still bind.

    v1 said "Extract {{ max_insights }} key takeaways" — a target. Gemini appeared to return
    exactly 12.0 / 25.0 / 50.0 at caps 12 / 25 / 50, which looked like obedience and was actually
    ``cleaned[:max_insights]`` hiding an overproduction of 300+ lines.

    A first attempt at a value bar removed the count anchor entirely ("there is NO target
    number"). Gemini then emitted 270-486 lines, blew max_output_tokens, and the guardrail turned
    that into ZERO insights on 8 of 15 runs. So the prompt needs both halves: a hard ceiling that
    binds, and explicit permission to return fewer.

    Third failure, 2026-09-01: the two halves CONTRADICTED each other. "HARD LIMIT: AT MOST N"
    sat four lines above "Return EVERY insight that clears the bar — no more, and no fewer …
    if it yields thirty, return thirty", which never answered the case the ceiling exists for:
    what to do when MORE than N clear the bar. Measured effects of resolving it are modest and
    model-specific (neutral on DGX Qwen over 8 concurrent pairs, -12% output tokens on the same
    model via OpenRouter, DeepSeek recall 30->52 i.e. UP toward the cap), so the value here is
    coherence, not a headline number.

    Checked across EVERY vendor's SELECTABLE prompt, not just gemini: the first version of this
    test read one file while the change touched fourteen, so thirteen were unguarded.

    v1 is deliberately excluded — it is the superseded quota prompt ("Extract N key takeaways")
    kept for history, and ``gi_insight_prompt_version`` documents only v2/v3 as selectable.
    Asserting the new form on v1 would be asserting that history was rewritten.
    """
    paths = sorted(
        p
        for p in PROMPTS_ROOT.glob("*/insight_extraction/v*.j2")
        if p.stem in {"v2", "v3"}  # the versions gi_insight_prompt_version can select
    )
    assert len(paths) >= 14, f"expected every vendor's v2/v3, found {len(paths)}"
    for path in paths:
        text = path.read_text()
        who = path.relative_to(PROMPTS_ROOT)
        # the ceiling must be stated as a hard limit, not buried
        assert "AT MOST {{ max_insights }}" in text, who
        assert "at most {{ max_insights }} lines" in text, who
        # the count must be neutral: the bar decides, not a target in either direction.
        # Biasing it DOWNWARD ("fewer is expected", "if in doubt leave it out") cost 3 CORE and
        # 13.7 USEFUL insights per episode to remove 6.3 filler — precision up, recall down.
        assert "no more, and no fewer" in text, who
        assert "Do not pad" in text, who
        assert "Do not hold back" in text, who
        # ...and the over-the-limit case must be answered, or the two halves contradict.
        assert "most important and stop" in text, who
        # Worked examples must be TEMPLATED off the limit. A literal ("if it yields thirty,
        # return thirty") silently exceeds it once max_insights is the per-chunk budget, which
        # is 25-38 for every chunked episode — a concrete contradiction replacing the abstract
        # one that was just removed.
        assert "return thirty" not in text, who
        assert "if it yields {{ max_insights }}, return {{ max_insights }}" in text, who
        # the old quota phrasing must not come back
        assert "Extract {{ max_insights }} key takeaways" not in text, who


def _texts(site: str) -> List[str]:
    out: List[str] = []
    for path in sorted(PROVIDERS_ROOT.rglob("*_provider.py")):
        lines = path.read_text().splitlines()
        for idx, line in enumerate(lines):
            if re.search(SITES[site], line):
                out.append(_collect_literal(lines, idx))
                break
    return out
