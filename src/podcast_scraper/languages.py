"""Language tags: what the feed said, and the one form the system reasons about.

A feed's ``<language>`` is free-ish text — ``en``, ``en-US``, ``pt_BR``, ``es-ES`` — and the
code that consumes it wants one stable form. This module holds that normalization and nothing
else; the registry of which languages are enabled, per-episode resolution and the per-feed
override arrive in S0.2 (#2174).

DELIBERATELY TRIVIAL (D-21). Lowercase, take the primary subtag, keep the raw value beside it.
No ``und`` / ``zxx`` / ``mul`` policy, no three-letter mapping table, no script parsing. Feeds
are onboarded manually a couple of episodes at a time, so a human inspects every one — a
defensive branch for input that process will never pass is machinery for nothing. A wrong tag
gets corrected by the per-feed override, not by guessing here.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple


def normalize_language_tag(raw: Optional[str]) -> Optional[str]:
    """``"en-US"`` -> ``"en"``, ``"pt_BR"`` -> ``"pt"``, junk -> ``None``.

    Splits on ``-`` and ``_`` and keeps the primary subtag, lowercased. Returns ``None`` for
    anything empty or with no alphabetic primary subtag, so a caller can tell "the feed said
    nothing useful" from "the feed said ``en``" — the raw value is preserved separately.

    ``"en-US"`` lowercased to ``"en-us"`` is the shape that breaks things today:
    ``whisper_utils.py:50`` checks ``language.lower() in ("en", "english")``, so ``"en-us"``
    reads as NOT English and a non-``.en`` Whisper model gets selected for an English episode.
    Every one of the 40 episodes in ``app-validation-corpus/v3`` carries exactly that value.
    """
    if not raw:
        return None
    primary = raw.strip().replace("_", "-").split("-", 1)[0].strip().lower()
    if not primary or not primary.isalpha():
        return None
    return primary


#: How a resolved language was arrived at. Recorded on the artifact because an audit that
#: cannot say WHERE a language came from cannot distinguish a measured corpus from one that
#: silently defaulted every episode — and a uniform 100% ``en`` is exactly what a check
#: measuring the configuration instead of the corpus looks like.
SOURCE_RSS = "rss"
SOURCE_PROFILE_DEFAULT = "profile_default"


def resolve_language(
    raw: Any, profile_default: Optional[str]
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """``(language_raw, language, language_source)`` for one feed.

    Precedence is the feed's own declared tag, then the profile default. The per-feed operator
    override and per-episode resolution land in S0.2 (#2174); this is the two-way case, which
    is all the parse slice needs.

    ``raw`` is typed ``Any`` on purpose. The pipeline hands this ``RssFeed.language``, and a
    large number of tests pass a ``MagicMock`` feed — on which attribute access returns another
    Mock, whose ``.strip()`` returns a Mock in turn. Normalizing that would either raise or
    invent a language out of a test double, so anything that is not a real ``str`` is treated
    as "the feed said nothing".
    """
    declared = raw.strip() if isinstance(raw, str) else ""
    if declared:
        normalized = normalize_language_tag(declared)
        if normalized:
            return declared, normalized, SOURCE_RSS
        # The feed declared something unusable ("", "und", "?"). Keep it visible rather than
        # dropping it: the operator onboarding this feed is the one who decides what it meant.
        return declared, normalize_language_tag(profile_default), SOURCE_PROFILE_DEFAULT
    return None, normalize_language_tag(profile_default), SOURCE_PROFILE_DEFAULT
