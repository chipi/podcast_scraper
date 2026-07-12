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
                    digest = hashlib.sha1(text.encode()).hexdigest()[:8]
                    found[site][digest].append(path.parent.name)
                    break
    return found


@pytest.mark.parametrize("site", sorted(SITES))
def test_shared_prompt_has_no_drift(site: str) -> None:
    variants = _variants().get(site, {})
    if not variants:
        pytest.skip(f"no provider defines {site} inline (it may have moved to a template)")

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
    """The anti-duplicate rule is load-bearing for candidate supply — keep it in the prompt."""
    variants = _variants().get("extract_quotes.system", {})
    if not variants:
        pytest.skip("extract_quotes prompt no longer inline")
    (text,) = {t for t in _texts("extract_quotes.system")}
    assert "DIFFERENT passage" in text
    assert "never repeat" in text.lower()


def _texts(site: str) -> List[str]:
    out: List[str] = []
    for path in sorted(PROVIDERS_ROOT.rglob("*_provider.py")):
        lines = path.read_text().splitlines()
        for idx, line in enumerate(lines):
            if re.search(SITES[site], line):
                out.append(_collect_literal(lines, idx))
                break
    return out
