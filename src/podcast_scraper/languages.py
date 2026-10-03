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

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


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


#: The language every ANALYSIS stage reads, and therefore the language the canonical
#: ``<base>.txt`` always holds (D-44).
#:
#: One named constant rather than `"en"` scattered through the pipeline, because the two meanings
#: are different facts that happen to share a value today: "English" the language of a particular
#: episode, and "the language our prompts, NER models and ad patterns are written in". Only the
#: second one is this. If analysis ever moves to another language, this is what changes — and the
#: places that legitimately mean the episode's own language keep saying `"en"`.
TARGET_LANGUAGE = "en"


#: Marker field: this episode cannot be served, and the reason why.
#:
#: Written on ``episode`` so it travels with the artifact rather than living in a side table that a
#: restore or a re-run could lose. Read by :func:`episode_is_unusable`, which is the ONE place that
#: decides — the catalog honours it, and the catalog is what feeds the app, the digest and the topic
#: clusters.
UNUSABLE_FIELD = "unusable"
UNUSABLE_REASON_FIELD = "unusable_reason"


def episode_is_unusable(doc: Any) -> Optional[str]:
    """The reason this episode cannot be served, or ``None`` when it can.

    THE CASE THIS EXISTS FOR (D-44). A non-English episode whose translation never completed has a
    SOURCE-language body at the canonical path — the path every generic reader opens believing it
    holds the analysis language. There is no third state: the atomic swap either happened or it did
    not. An episode in the second state cannot be summarised, have insights extracted, or be
    indexed without producing confident nonsense, so it is not served at all.

    Marko's instruction, 2026-10-02: "if there is no english file after translation that means
    pipeline cannot work, therefore we stop and somehow flag this episode is not good and it does
    not show up anywhere."

    EXPLICIT, NOT INFERRED. The marker is written by the stage that discovered the problem, and
    read here. The alternative — every surface re-deriving "is this episode okay" from the files on
    disk — is how a reader ends up with a different answer from its neighbour, which is the whole
    class of bug this arc kept meeting.
    """
    if not isinstance(doc, dict):
        return None
    episode = doc.get("episode")
    if not isinstance(episode, dict):
        return None
    if not episode.get(UNUSABLE_FIELD):
        return None
    reason = episode.get(UNUSABLE_REASON_FIELD)
    return str(reason).strip() if isinstance(reason, str) and reason.strip() else "unspecified"


#: How a resolved language was arrived at. Recorded on the artifact because an audit that
#: cannot say WHERE a language came from cannot distinguish a measured corpus from one that
#: silently defaulted every episode — and a uniform 100% ``en`` is exactly what a check
#: measuring the configuration instead of the corpus looks like.
SOURCE_RSS = "rss"
SOURCE_PROFILE_DEFAULT = "profile_default"
SOURCE_OVERRIDE = "override"


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


# --- the registry -------------------------------------------------------------------------

_DEFAULT_CONFIGURED_PATH = "config/languages.yaml"
_CONTAINER_FALLBACK_PATHS: Tuple[Path, ...] = (Path("/app/config/languages.yaml"),)

_registry_cache: Optional[Dict[str, "LanguageEntry"]] = None


@dataclass(frozen=True)
class LanguageEntry:
    """One row of ``config/languages.yaml``.

    ``enabled`` is the only field with teeth. ``tier`` and ``wer`` inform sequencing and are
    recorded so a decision can cite them, not so code can branch on them.
    """

    code: str
    name: str
    tier: int
    enabled: bool
    script: Optional[str] = None
    wer: Optional[float] = None


def _bundled_path() -> Optional[Path]:
    """Packaged fallback shipped in the wheel (pipeline containers)."""
    try:
        from importlib import resources

        ref = resources.files("podcast_scraper").joinpath("data/languages.yaml")
        with resources.as_file(ref) as extracted:
            p = Path(extracted)
            return p if p.is_file() else None
    except (ImportError, FileNotFoundError, TypeError, OSError):  # pragma: no cover
        return None


def _resolve_registry_path() -> Optional[Path]:
    """Repo ``config/``, then the container path, then the wheel-bundled copy.

    The same three-step search ``providers/known_models.py`` uses. Without the bundled leg the
    container finds nothing and the registry is silently empty — which for known_models meant
    running every cloud call with no allowlist at all.
    """
    for base in [Path.cwd(), *Path.cwd().parents]:
        candidate = base / _DEFAULT_CONFIGURED_PATH
        if candidate.is_file():
            return candidate
    for p in _CONTAINER_FALLBACK_PATHS:
        if p.is_file():
            return p
    return _bundled_path()


def clear_language_registry_cache() -> None:
    """Reset the loader cache (for tests / after editing the YAML at runtime)."""
    global _registry_cache
    _registry_cache = None


def language_registry() -> Dict[str, LanguageEntry]:
    """The registry, keyed by normalized code. Empty dict when the YAML cannot be found.

    An empty registry means nothing is enabled, so every episode would skip. That is the loud
    failure, and it is the right one: silently treating an unknown language as English is the
    hazard this whole phase exists to remove.
    """
    global _registry_cache
    if _registry_cache is not None:
        return _registry_cache

    path = _resolve_registry_path()
    if path is None:
        logger.warning(
            "languages.yaml not found (looked for %s upward from cwd, %s, then the bundled "
            "copy); the language registry is EMPTY, so every language reads as not-enabled",
            _DEFAULT_CONFIGURED_PATH,
            ", ".join(str(p) for p in _CONTAINER_FALLBACK_PATHS),
        )
        _registry_cache = {}
        return _registry_cache

    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (ImportError, OSError, ValueError) as exc:
        logger.warning("languages.yaml at %s could not be read (%s); registry is EMPTY", path, exc)
        _registry_cache = {}
        return _registry_cache

    out: Dict[str, LanguageEntry] = {}
    for raw_code, row in (payload.get("languages") or {}).items():
        code = normalize_language_tag(str(raw_code))
        if not code or not isinstance(row, dict):
            continue
        out[code] = LanguageEntry(
            code=code,
            name=str(row.get("name") or code),
            tier=int(row.get("tier") or 0),
            enabled=bool(row.get("enabled")),
            script=(str(row["script"]) if row.get("script") else None),
            wer=(float(row["wer"]) if row.get("wer") is not None else None),
        )
    _registry_cache = out
    return out


def is_language_enabled(code: Optional[str]) -> bool:
    """Is this language one we ingest? Unknown or absent → False, never a lenient default."""
    if not code:
        return False
    entry = language_registry().get(code)
    return bool(entry and entry.enabled)


def resolve_episode_language(
    *,
    override: Any = None,
    feed_declared: Any = None,
    profile_default: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """``(language_raw, language, language_source)`` with the full precedence (#2174).

    Operator override > the feed's declared tag > the profile default. Keyword-only, because
    three optional string arguments in a row is exactly the signature where a positional
    mix-up would silently invert the precedence and never raise.

    The profile default is the ONLY place the run-global setting may be read. Every other
    ``cfg.language`` reader is what S0.6 removes, and the lint added there whitelists this one.

    Note what this deliberately does NOT do: it does not consult the registry. Resolution
    answers "what language is this episode in"; whether we ingest that language is a separate
    question with a separate answer (:func:`is_language_enabled`, acted on by S0.8). Folding
    them together would mean an unsupported language silently resolving to something else.
    """
    for candidate, source in (
        (override, SOURCE_OVERRIDE),
        (feed_declared, SOURCE_RSS),
    ):
        declared = candidate.strip() if isinstance(candidate, str) else ""
        if declared:
            normalized = normalize_language_tag(declared)
            if normalized:
                return declared, normalized, source
            # Declared but unusable. Keep it visible and fall through to the default rather
            # than inventing a reading of it.
            return declared, normalize_language_tag(profile_default), SOURCE_PROFILE_DEFAULT
    return None, normalize_language_tag(profile_default), SOURCE_PROFILE_DEFAULT


def resolve_config_language(
    cfg: Any, *, feed_language: Any = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """``(language_raw, language, language_source)`` for a run config — provenance included.

    The same resolution :func:`transcription_language` performs, but returning the SOURCE as
    well, for the callers that have to record where the value came from rather than merely act
    on it. Lives here, beside the other one, because this is the only module permitted to read
    ``cfg.language`` (S0.6, enforced by ``scripts/check/lint_language_readers.py``) — a caller
    that wanted the provenance would otherwise have had to read the global itself and earn a
    whitelist entry, which would have made the lint's "one reader" claim a little less true
    every time someone needed it.

    Pass ``feed_language`` when the feed object is in hand: it is what makes the difference
    between a language measured from the feed (``rss``) and one defaulted from the profile
    (``profile_default``), and that distinction is the whole point of recording the source.
    """
    return resolve_episode_language(
        override=getattr(cfg, "language_override", None),
        # An explicit `feed_language=` wins (the seam has the feed object in hand); otherwise
        # fall back to the tag the scraping stage recorded on the per-feed Config. Without this
        # fallback the ONE reader could not see the feed's declared language at all, and
        # answered from the profile default — the split this field exists to close.
        feed_declared=feed_language or getattr(cfg, "feed_declared_language", None),
        profile_default=getattr(cfg, "language", None),
    )


def transcription_language(cfg: Any) -> Optional[str]:
    """The language to transcribe this episode in — THE ONE READER (#2177).

    Every stage that needs a language for a provider call asks here, and this is the only place
    outside :func:`resolve_episode_language` that may consult the run-global ``cfg.language``.
    ``scripts/check/lint_language_readers.py`` enforces that, with an explicit whitelist for the
    handful of legitimate exceptions.

    WHY A FUNCTION RATHER THAN A THREADED ARGUMENT. The alternative was a new parameter through
    ``transcribe_media_to_text`` -> ``_transcribe_with_segments_maybe_chunked`` ->
    ``transcribe_with_sniff_gate`` -> two inner closures, five signatures deep, where the failure
    mode is one path that keeps reading the global and nobody notices. One function, one call per
    site, and the lint proves there is no second reader.

    Returns ``None`` when nothing resolved, which is an honest "let the engine decide". It is NOT
    ``"en"``: that substitution is what made a non-English episode come back as a plausible
    English transcript of the wrong words.
    """
    # `feed_declared` is left unset here on purpose: `resolve_config_language` falls back to
    # `cfg.feed_declared_language`, which the scraping stage sets from the channel tag. An
    # earlier version of this comment claimed the tag "is already in `cfg.language`" — nothing
    # put it there, so this reader answered from the profile default and disagreed with the
    # metadata writer about the same feed.
    _raw, language, _source = resolve_config_language(cfg)
    return language
