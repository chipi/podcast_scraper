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
    "cleaning.system": r'"You are a transcript cleaning assistant',
}

PROMPTS_ROOT = _REPO_ROOT / "src" / "podcast_scraper" / "prompts"

# prompt site -> the per-provider template subdir it lives in once migrated off inline literals.
# When a site is no longer inline we assert it is TEMPLATED here (not silently deleted) and that
# the copies have not drifted apart — the same "identical by intent" guarantee, moved to templates.
_SITE_TEMPLATE_DIR = {
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


def _texts(site: str) -> List[str]:
    out: List[str] = []
    for path in sorted(PROVIDERS_ROOT.rglob("*_provider.py")):
        lines = path.read_text().splitlines()
        for idx, line in enumerate(lines):
            if re.search(SITES[site], line):
                out.append(_collect_literal(lines, idx))
                break
    return out
