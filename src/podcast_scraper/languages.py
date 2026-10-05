"""Language tags: what the feed said, and the one form the system reasons about.

A feed's ``<language>`` is free-ish text — ``en``, ``en-US``, ``pt_BR``, ``es-ES``, ``English``,
``eng`` — and the code that consumes it wants one stable form: an ISO 639-1 code.

STRICT, AND EMPTY MEANS EMPTY (operator, 2026-10-05; supersedes D-21's "deliberately trivial").
A tag becomes a code only when it IS one: an ISO 639-1 code (any region / script subtag dropped),
an ISO 639-2 three-letter code, or a language's English or native name. Anything else — "und",
"?", "x-klingon", a typo — is NO language, and so is a missing tag. No language is never read as
English: the profile default no longer stands in for it (see ``resolve_episode_language``), and
an episode with no language is refused before anything is downloaded. The remedy for a wrong or
missing tag is an operator override (#2283), not a guess here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


#: Every ISO 639-1 code (the two-letter set, as published by the ISO 639 registration authority).
ISO_639_1: frozenset = frozenset(
    "aa ab ae af ak am an ar as av ay az ba be bg bi bm bn bo br bs ca ce ch co cr cs cu cv cy "
    "da de dv dz ee el en eo es et eu fa ff fi fj fo fr fy ga gd gl gn gu gv ha he hi ho hr ht "
    "hu hy hz ia id ie ig ii ik io is it iu ja jv ka kg ki kj kk kl km kn ko kr ks ku kv kw ky "
    "la lb lg li ln lo lt lu lv mg mh mi mk ml mn mr ms mt my na nb "
    "nd ne ng nl nn no nr nv ny "  # codespell:ignore nd
    "oc oj om or os pa pi pl ps pt qu rm rn ro ru rw sa sc sd se sg si sk sl sm sn so sq sr ss "
    "st su sv sw ta te tg th ti tk tl tn to tr ts tt tw ty ug uk ur uz ve vi vo wa wo xh yi yo "
    "za zh zu".split()
)

#: Withdrawn ISO 639-1 codes that feeds still carry, mapped to their current code.
_ISO_639_1_RETIRED = {"iw": "he", "in": "id", "ji": "yi", "jw": "jv", "mo": "ro"}

#: ISO 639-2 (bibliographic AND terminology) -> ISO 639-1, for the languages the registry knows
#: plus the most common others. A three-letter code not listed here is NO language — it may be a
#: real code, but a tag this pipeline cannot place is treated as missing, never guessed.
_ISO_639_2_TO_1 = {
    "eng": "en", "spa": "es", "ita": "it", "por": "pt", "deu": "de", "ger": "de",
    "fra": "fr", "fre": "fr", "nld": "nl", "dut": "nl", "cat": "ca", "swe": "sv",
    "nob": "nb", "nor": "no", "nno": "nn", "rus": "ru", "ron": "ro", "rum": "ro",
    "bul": "bg", "srp": "sr", "jpn": "ja", "kor": "ko", "zho": "zh", "chi": "zh",
    "ara": "ar", "pol": "pl", "tur": "tr", "ukr": "uk", "ces": "cs", "cze": "cs",
    "ell": "el", "gre": "el", "heb": "he", "hin": "hi", "hun": "hu", "fin": "fi",
    "dan": "da", "slk": "sk", "slo": "sk", "slv": "sl", "hrv": "hr", "ind": "id",
    "vie": "vi", "tha": "th",  # codespell:ignore vie,tha
    "fas": "fa", "per": "fa", "glg": "gl", "eus": "eu",
    "baq": "eu", "isl": "is", "ice": "is", "gle": "ga", "cym": "cy",
    "wel": "cy",  # codespell:ignore wel
}  # fmt: skip

#: Language NAMES -> ISO 639-1, English and native spellings, accents optional. Only the
#: registry's languages: a name is the least precise tag there is, so it is honoured only where
#: the pipeline can act on the answer.
_LANGUAGE_NAMES = {
    "english": "en", "spanish": "es", "español": "es", "espanol": "es", "castellano": "es",
    "italian": "it", "italiano": "it", "portuguese": "pt", "português": "pt",
    "portugues": "pt",  # codespell:ignore portugues
    "german": "de", "deutsch": "de", "french": "fr", "français": "fr",
    "francais": "fr", "dutch": "nl", "nederlands": "nl", "catalan": "ca", "català": "ca",
    "catala": "ca", "swedish": "sv", "svenska": "sv", "norwegian": "no", "norsk": "no",
    "russian": "ru", "русский": "ru", "romanian": "ro", "română": "ro", "romana": "ro",
    "bulgarian": "bg", "български": "bg", "serbian": "sr", "српски": "sr", "srpski": "sr",
    "japanese": "ja", "日本語": "ja", "korean": "ko", "한국어": "ko", "chinese": "zh",
    "中文": "zh", "arabic": "ar", "العربية": "ar",
}  # fmt: skip


def normalize_language_tag(raw: Optional[str]) -> Optional[str]:
    """The ISO 639-1 code *raw* names, or ``None`` when it names none.

    ``"en-US"`` / ``"EN"`` / ``"en_gb"`` / ``"eng"`` / ``"English"`` -> ``"en"``;
    ``"pt_BR"`` / ``"por"`` / ``"Português"`` -> ``"pt"``; ``"iw"`` -> ``"he"`` (retired code);
    ``""`` / ``"und"`` / ``"?"`` / ``"xx"`` / ``"klingon"`` -> ``None``.

    ``None`` means NO LANGUAGE, and callers must treat it as such — never as English. The raw
    value is preserved separately by the resolvers so an operator can see what the feed said.

    ``"en-US"`` lowercased to ``"en-us"`` was the shape that broke things before this existed:
    ``whisper_utils.py:50`` checks ``language.lower() in ("en", "english")``, so ``"en-us"`` read
    as NOT English and a non-``.en`` Whisper model was selected for an English episode.
    """
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip().lower()
    if not text:
        return None
    if text in _LANGUAGE_NAMES:
        return _LANGUAGE_NAMES[text]
    primary = text.replace("_", "-").split("-", 1)[0].strip()
    if primary in ISO_639_1:
        return primary
    if primary in _ISO_639_1_RETIRED:
        return _ISO_639_1_RETIRED[primary]
    if primary in _ISO_639_2_TO_1:
        return _ISO_639_2_TO_1[primary]
    return None


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
#: No longer produced (2026-10-05): kept so artifacts written before then still read.
SOURCE_PROFILE_DEFAULT = "profile_default"
SOURCE_OVERRIDE = "override"
#: Nothing declared a language at all — no override and no feed tag.
SOURCE_NONE = "none"


def resolve_language(
    raw: Any, profile_default: Optional[str] = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """``(language_raw, language, language_source)`` for one feed's declared tag alone.

    The feed-only case of :func:`resolve_episode_language`, with the same rule: no usable tag
    means no language (``SOURCE_NONE``); ``profile_default`` is ignored.

    ``raw`` is typed ``Any`` on purpose. The pipeline hands this ``RssFeed.language``, and a
    large number of tests pass a ``MagicMock`` feed — on which attribute access returns another
    Mock, whose ``.strip()`` returns a Mock in turn. Anything that is not a real ``str`` is
    treated as "the feed said nothing".
    """
    return resolve_episode_language(feed_declared=raw, profile_default=profile_default)


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
    """``(language_raw, language, language_source)``: operator override, else the feed's tag.

    NOTHING ELSE (operator, 2026-10-05). The profile default used to stand in when the feed said
    nothing, so a feed with no ``<language>`` was silently treated as English; it no longer
    does. ``profile_default`` is accepted for call-site compatibility and IGNORED. When neither
    declares anything the answer is ``(None, None, SOURCE_NONE)``; when the deciding value is
    not an ISO code it is ``(raw, None, <its source>)``. Either way: no language, and the ingest
    gate refuses the episode before anything is downloaded.

    The FIRST declared value decides, usable or not: an override set to something unusable is
    an operator error and yields no language, rather than quietly falling back to the feed's tag.

    Note what this deliberately does NOT do: it does not consult the registry. Resolution
    answers "what language is this episode in"; whether we ingest that language is a separate
    question with a separate answer (:func:`is_language_enabled`).
    """
    del profile_default
    for candidate, source in (
        (override, SOURCE_OVERRIDE),
        (feed_declared, SOURCE_RSS),
    ):
        declared = candidate.strip() if isinstance(candidate, str) else ""
        if declared:
            normalized = normalize_language_tag(declared)
            # The SOURCE is where the value came from even when it is unusable, so a refusal
            # blames the right thing: a broken override is the override's fault, not the feed's.
            return declared, normalized, source
    return None, None, SOURCE_NONE


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
