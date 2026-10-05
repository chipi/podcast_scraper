"""#652 Part B — deterministic post-extraction validators for GI insights.

Two filters that run on the final insight list regardless of source
(``provider``, transcript chunks, prefilled):

1. Ad filter — drops insights that sit inside a transcript window that
   matches ≥ 2 sponsor-ad regex patterns. Threshold tuned conservatively
   (≥ 2 not ≥ 1) so a CEO legitimately describing their own product
   doesn't trip it.

2. Dialogue filter — drops insights that (a) start with conversational
   filler, (b) exceed a first-person-pronoun density threshold, or
   (c) are dominated by a verbatim quote (quote > 60 % of insight text).

Filters are pure functions — no side effects. Callers (``gi/pipeline.py``)
wire them into :func:`~podcast_scraper.workflow.metrics.Metrics` counters
for observability.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..languages import primary_language, TARGET_LANGUAGE

# ---------------------------------------------------------------------------
# Ad filter (Finding 14-lite)
# ---------------------------------------------------------------------------

#: Sponsor-ad cue vocabulary, KEYED BY THE LANGUAGE OF THE TEXT BEING SCANNED.
#:
#: WHY A MAP AND NOT A LIST. Every pattern below is a phrase in a specific language, so a flat list
#: is a list in ONE language wearing no label. Scanned against Spanish, the English list matches
#: nothing and the caller cannot tell "this episode has no ads" from "I do not speak this episode".
#: Measured on the V.6a fixture: the Spanish source matched ZERO English patterns while its English
#: translation matched the ad read correctly. The shape follows
#: :data:`podcast_scraper.search.query_language._MARKERS`, which is language-keyed for the same
#: reason.
#:
#: WHY THIS IS LATENT, NOT A LIVE BUG. Under D-44 the canonical transcript body is always the
#: ANALYSIS language, so every caller here reads English by construction and
#: :data:`_AD_PATTERNS` below resolves to exactly the list that shipped. The map is what makes a
#: future non-English analysis language a DATA change rather than a rewrite.
#:
#: HONEST PROVENANCE OF THE NON-ENGLISH ROWS. The English row is MEASURED — verified spoken-form
#: patterns hit ad reads in 11/100 episodes of a real corpus at the >= 2-distinct threshold. The
#: other five rows are TRANSLATIONS, authored from the English categories and never run against
#: real non-English podcast audio. They are a starting vocabulary, not a measurement; the first
#: non-English ASR run (#2187) is what can tell whether they fire.
#:
#: Each row keeps the English row's three categories, because that is what the >= 2-distinct
#: threshold assumes: a sponsor disclosure, a spoken/hybrid URL, and a promo/CTA.
#: Accents are written ``(?:ó|o)`` throughout — ASR drops them routinely, and a pattern that
#: requires the accent silently stops matching the moment it does.
_AD_PATTERN_SOURCES: Dict[str, Tuple[str, ...]] = {
    # Patterns are tuned for SPOKEN transcript text (Whisper output) where URLs
    # become phonetic — "bloomberg.com/odd_lots" → "Bloomberg dot com slash Odd
    # Lots". Pre-overnight-#652-stabilization the patterns assumed machine-URL
    # form (``\bvisit\s+[\w.]+/\w+\b``) and caught 0/1200 insights on a 100-ep
    # real corpus. Verified spoken-form patterns hit ad reads in 11/100 eps
    # at the ≥ 2-distinct-patterns threshold.
    "en": (
        # Sponsor disclosure phrases — high-precision standalone signals.
        r"\bbrought to you by\b",
        r"\bsponsored by\b",
        r"\bthis episode is sponsored\b",
        r"\bthis (?:show|podcast) is (?:brought|sponsored)\b",
        r"\bour sponsors?\b",
        r"\btoday'?s sponsor\b",
        r"\bthanks (?:to )?our sponsors?\b",
        r"\bsupport the (?:show|podcast)\b",
        # Spoken-URL signatures (Whisper output).
        r"\b\w+\s+dot\s+com\b",  # "ramp dot com"
        r"\bdot\s+com\s+slash\b",  # "dot com slash promo"
        r"\bslash\s+(?:promo|code|deal|free|trial|save|offer|pod)\b",
        r"\bgo to\s+\w+\s+dot\s+com\b",
        r"\bvisit\s+\w+\s+dot\s+com\b",
        r"\bhead (?:over )?to\s+\w+\s+dot\s+com\b",
        # Hybrid URL forms — modern Whisper-1 keeps literal punctuation (#663).
        # Invest-Like-the-Best-style pre-rolls produce ``ramp.com slash invest``
        # and ``Visit WorkOS.com`` rather than fully-spoken ``dot com``.
        r"\b\w+\.(?:com|ai|io|co)\s+slash\b",  # "ramp.com slash invest"
        r"\bvisit\s+\w+\.(?:com|ai|io|co)\b",  # "Visit WorkOS.com"
        r"\bgo to\s+\w+\.(?:com|ai|io|co)\b",  # "go to ramp.com"
        r"\bcheck out\s+\w+\.(?:com|ai|io|co)\b",  # "check out ramp.com"
        r"\blearn more at\s+\w+\.(?:com|ai|io|co)\b",  # "Learn more at rogo.ai"
        r"\bhead (?:over )?to\s+\w+\.(?:com|ai|io|co)\b",
        # Promo / CTA phrases — common in ad reads.
        r"\b(?:promo|use)\s+code\s+\w+",
        r"\bsave\s+(?:up to\s+)?\d+\s*(?:percent|%)\b",
        r"\bget\s+\d+\s*(?:percent|%)\s*off\b",
        r"\b\d+\s*(?:percent|%)\s*off\b",  # bare "20% off" (discount offer)
        r"\bfree (?:trial|month|shipping|delivery)\b",
        r"\bfor a limited time\b",
        r"\bsign up\s+(?:today|now)\b",
    ),
    "es": (
        # Sponsor disclosure.
        r"\bpatrocinado por\b",
        r"\bcon el patrocinio de\b",
        r"\beste episodio (?:est(?:á|a)|es) patrocinado\b",
        r"\beste (?:programa|podcast) (?:est(?:á|a)|es) patrocinado\b",
        r"\bnuestros? patrocinadores?\b",
        r"\bel patrocinador de hoy\b",
        r"\bgracias a (?:nuestros )?patrocinadores\b",
        r"\bapoya (?:el|este) (?:programa|podcast)\b",
        # Spoken URL — "punto com" is the Spanish "dot com".
        r"\b\w+\s+punto\s+com\b",
        r"\bpunto\s+com\s+barra\b",
        r"\bbarra\s+(?:promo|c(?:ó|o)digo|oferta|gratis|prueba)\b",
        r"\b(?:ve|vayan?|entra|entren)\s+a\s+\w+\s+punto\s+com\b",
        r"\bvisita\w*\s+\w+\s+punto\s+com\b",
        # Hybrid URL forms.
        r"\b\w+\.(?:com|es|ai|io|co)\s+barra\b",
        r"\bvisita\w*\s+\w+\.(?:com|es|ai|io|co)\b",
        r"\b(?:ve|entra)\s+a\s+\w+\.(?:com|es|ai|io|co)\b",
        r"\bm(?:á|a)s informaci(?:ó|o)n en\s+\w+\.(?:com|es|ai|io|co)\b",
        # Promo / CTA.
        r"\bc(?:ó|o)digo (?:promocional|de descuento)\b",
        r"\busa\s+el\s+c(?:ó|o)digo\b",
        r"\bahorra\s+(?:hasta\s+)?\d+\s*(?:por ciento|%)\b",
        r"\b\d+\s*(?:por ciento|%)\s*de descuento\b",
        r"\b(?:prueba|env(?:í|i)o|mes)\s+gratis\b",
        r"\bpor tiempo limitado\b",
        r"\breg(?:í|i)strate\s+(?:hoy|ahora|ya)\b",
    ),
    "it": (
        # Sponsor disclosure.
        r"\bsponsorizzato da(?:l|lla|llo|gli|lle|i)?\b",
        r"\bin collaborazione con\b",
        r"\bquesto episodio (?:è|e'|e) sponsorizzato\b",
        r"\bquesto (?:programma|podcast) (?:è|e'|e) sponsorizzato\b",
        r"\bi nostri sponsor\b",
        r"\blo sponsor di oggi\b",
        r"\bgrazie ai nostri sponsor\b",
        r"\bsostieni (?:il|questo) (?:programma|podcast)\b",
        # Spoken URL.
        r"\b\w+\s+punto\s+com\b",
        r"\bpunto\s+com\s+(?:slash|barra)\b",
        r"\b(?:slash|barra)\s+(?:promo|codice|offerta|gratis|prova)\b",
        r"\b(?:vai|andate)\s+su\s+\w+\s+punto\s+com\b",  # codespell:ignore vai
        r"\bvisita\w*\s+\w+\s+punto\s+com\b",
        # Hybrid URL forms.
        r"\b\w+\.(?:com|it|ai|io|co)\s+(?:slash|barra)\b",
        r"\bvisita\w*\s+\w+\.(?:com|it|ai|io|co)\b",
        r"\b(?:vai|andate)\s+su\s+\w+\.(?:com|it|ai|io|co)\b",  # codespell:ignore vai
        r"\bscopri di pi(?:ù|u)\s+su\s+\w+\.(?:com|it|ai|io|co)\b",
        # Promo / CTA.
        r"\bcodice (?:promozionale|sconto)\b",
        r"\busa\s+il\s+codice\b",
        r"\brisparmia\s+(?:fino a\s+)?\d+\s*(?:per cento|%)\b",
        r"\b\d+\s*(?:per cento|%)\s*di sconto\b",
        r"\b(?:prova|spedizione|mese)\s+gratuit\w+\b",
        r"\bper un (?:periodo|tempo) limitato\b",
        r"\biscriviti\s+(?:oggi|ora|subito)\b",
    ),
    "fr": (
        # Sponsor disclosure.
        r"\bsponsoris(?:é|e) par\b",
        r"\bpr(?:é|e)sent(?:é|e) par\b",
        r"\bcet (?:é|e)pisode est sponsoris(?:é|e)\b",
        r"\bce (?:programme|podcast) est sponsoris(?:é|e)\b",
        r"\bnos (?:sponsors|partenaires)\b",
        r"\ble sponsor du jour\b",
        r"\bmerci (?:à|a) nos sponsors\b",
        r"\bsoutenez (?:le|ce) (?:programme|podcast)\b",
        # Spoken URL — "point com".
        r"\b\w+\s+point\s+com\b",
        r"\bpoint\s+com\s+(?:slash|barre)\b",
        r"\b(?:slash|barre)\s+(?:promo|code|offre|gratuit|essai)\b",
        r"\b(?:allez|rendez-vous)\s+sur\s+\w+\s+point\s+com\b",
        r"\bvisitez\s+\w+\s+point\s+com\b",
        # Hybrid URL forms.
        r"\b\w+\.(?:com|fr|ai|io|co)\s+(?:slash|barre)\b",
        r"\bvisitez\s+\w+\.(?:com|fr|ai|io|co)\b",
        r"\b(?:allez|rendez-vous)\s+sur\s+\w+\.(?:com|fr|ai|io|co)\b",
        r"\ben savoir plus sur\s+\w+\.(?:com|fr|ai|io|co)\b",
        # Promo / CTA.
        r"\bcode (?:promo|promotionnel|de r(?:é|e)duction)\b",
        r"\butilisez\s+le\s+code\b",
        r"\b(?:é|e)conomisez\s+(?:jusqu'(?:à|a)\s+)?\d+\s*(?:pour cent|%)\b",
        r"\b\d+\s*(?:pour cent|%)\s*de (?:r(?:é|e)duction|remise)\b",
        r"\b(?:essai|livraison|mois)\s+gratuit\w*\b",
        r"\bpour une dur(?:é|e)e limit(?:é|e)e\b",
        r"\binscrivez-vous\s+(?:aujourd'hui|maintenant|d(?:è|e)s maintenant)\b",
    ),
    "de": (
        # Sponsor disclosure. ``Werbung`` is here because German podcasts announce paid segments
        # with the bare word — legally required, so it is a high-frequency marker. It is also the
        # ordinary noun for "advertising", which is exactly what the >= 2-distinct threshold is
        # for: on its own it never cuts anything.
        r"\bpr(?:ä|a)sentiert von\b",
        r"\bgesponsert vo(?:n|m)\b",
        r"\bdiese (?:folge|episode) (?:wird|ist).{0,40}?gesponsert\b",
        r"\bunsere? sponsoren?\b",
        r"\bunser sponsor heute\b",
        r"\bdanke an unsere sponsoren\b",
        r"\bunterst(?:ü|u)tz(?:e|t) (?:den|diesen) podcast\b",
        r"\bmit unterst(?:ü|u)tzung von\b",
        r"\bwerbung\b",
        # Spoken URL — "punkt de" as often as "punkt com".
        r"\b\w+\s+punkt\s+(?:com|de)\b",
        r"\bpunkt\s+(?:com|de)\s+(?:slash|schr(?:ä|a)gstrich)\b",
        r"\b(?:slash|schr(?:ä|a)gstrich)\s+(?:promo|code|angebot|gratis|test)\b",
        r"\b(?:geh|geht|gehen sie)\s+auf\s+\w+\s+punkt\s+(?:com|de)\b",  # codespell:ignore sie
        r"\bbesuch(?:e|t|en sie)?\s+\w+\s+punkt\s+(?:com|de)\b",  # codespell:ignore sie
        # Hybrid URL forms.
        r"\b\w+\.(?:com|de|ai|io|co)\s+(?:slash|schr(?:ä|a)gstrich)\b",
        r"\bbesuch(?:e|t|en sie)?\s+\w+\.(?:com|de|ai|io|co)\b",  # codespell:ignore sie
        r"\bmehr (?:dazu|infos|informationen) (?:unter|auf)"  # codespell:ignore unter
        r"\s+\w+\.(?:com|de|ai|io|co)\b",
        # Promo / CTA.
        r"\b(?:gutschein|rabatt|promo)code\b",
        r"\bmit dem code\b",
        r"\b\d+\s*(?:prozent|%)\s*(?:rabatt|sparen)\b",
        r"\bspar(?:e|t|en sie)\s+(?:bis zu\s+)?\d+\s*(?:prozent|%)\b",  # codespell:ignore sie
        r"\bkostenlos(?:e|er)?\s+(?:testphase|versand|monat)\b",
        r"\bgratis\s+testen\b",
        r"\bnur f(?:ü|u)r kurze zeit\b",
        r"\bjetzt\s+(?:anmelden|registrieren)\b",
    ),
    "pt": (
        # Sponsor disclosure.
        r"\bpatrocinado p(?:or|el[oa])\b",
        r"\boferecid[oa] p(?:or|el[oa])\b",
        r"\beste epis(?:ó|o)dio (?:é|e) patrocinado\b",
        r"\beste (?:programa|podcast) (?:é|e) patrocinado\b",
        r"\bnossos? patrocinadores?\b",
        r"\bo patrocinador de hoje\b",
        r"\bobrigado aos nossos patrocinadores\b",
        r"\bapoie (?:o|este) (?:programa|podcast)\b",
        # Spoken URL — "ponto com".
        r"\b\w+\s+ponto\s+com\b",
        r"\bponto\s+com\s+barra\b",
        r"\bbarra\s+(?:promo|c(?:ó|o)digo|oferta|gr(?:á|a)tis|teste)\b",
        r"\b(?:v(?:á|a)|acesse|entre em)\s+\w+\s+ponto\s+com\b",
        r"\bvisite\s+\w+\s+ponto\s+com\b",
        # Hybrid URL forms.
        r"\b\w+\.(?:com|br|ai|io|co)\s+barra\b",
        r"\b(?:visite|acesse)\s+\w+\.(?:com|br|ai|io|co)\b",
        r"\bsaiba mais em\s+\w+\.(?:com|br|ai|io|co)\b",
        # Promo / CTA.
        r"\bc(?:ó|o)digo (?:promocional|de desconto)\b",
        r"\buse o c(?:ó|o)digo\b",
        r"\beconomize\s+(?:at(?:é|e)\s+)?\d+\s*(?:por cento|%)\b",
        r"\b\d+\s*(?:por cento|%)\s*de desconto\b",
        r"\b(?:teste|frete|m(?:ê|e)s)\s+gr(?:á|a)tis\b",
        r"\bpor tempo limitado\b",
        r"\b(?:inscreva-se|cadastre-se)\s+(?:hoje|agora|j(?:á|a))\b",
    ),
}

AD_PATTERNS_BY_LANGUAGE: Dict[str, Tuple[re.Pattern[str], ...]] = {
    lang: tuple(re.compile(p, re.IGNORECASE) for p in patterns)
    for lang, patterns in _AD_PATTERN_SOURCES.items()
}


def ad_patterns_for(language: Optional[str]) -> Tuple[re.Pattern[str], ...]:
    """The ad-cue patterns for ``language`` — EMPTY when we have no vocabulary for it.

    Empty rather than an English fallback, deliberately. Scanning Spanish with English regexes
    returns zero hits, which reads downstream as "this episode is clean" — a confident wrong
    answer. Empty is the same zero, but :data:`AD_PATTERNS_LANGUAGES` lets a caller that cares
    tell the two apart instead of being lied to.
    """
    if not language:
        return ()
    return AD_PATTERNS_BY_LANGUAGE.get(primary_language(language), ())


#: The languages whose ad vocabulary exists at all. A caller deciding whether ad excision is
#: MEANINGFUL for an episode asks this, not ``ad_patterns_for(...) != ()``.
AD_PATTERNS_LANGUAGES = frozenset(AD_PATTERNS_BY_LANGUAGE)

#: The analysis-language row, resolved once. Under D-44 the canonical transcript body is always
#: the analysis language, so this is what every caller in this package reads — and it is byte-for
#: -byte the list that shipped before the map existed.
_AD_PATTERNS: Tuple[re.Pattern[str], ...] = AD_PATTERNS_BY_LANGUAGE[TARGET_LANGUAGE]


_AD_HITS_THRESHOLD = 2


def _ad_pattern_hits(text: str) -> int:
    if not text:
        return 0
    return sum(1 for p in _AD_PATTERNS if p.search(text))


def insight_looks_like_ad(insight_text: str, source_text: Optional[str] = None) -> bool:
    """Return True when ``source_text`` (or the insight itself) matches ≥ 2
    sponsor-ad regex patterns.

    ``source_text`` should be the transcript window the insight was distilled
    from (quote context) when available; we fall back to scanning the insight
    text directly. The ≥ 2-pattern threshold keeps false positives low — a
    single "go to example.com/X" on its own can appear in genuine content, but
    two or more ad-phrase hits within the same passage is reliably a sponsor
    read.
    """
    hits = _ad_pattern_hits(insight_text)
    if source_text and source_text != insight_text:
        hits += _ad_pattern_hits(source_text)
    return hits >= _AD_HITS_THRESHOLD


# ---------------------------------------------------------------------------
# Dialogue filter (Finding 12)
# ---------------------------------------------------------------------------

_FILLER_PREFIXES: Tuple[str, ...] = (
    # Post-#652 audit: dropped "so", "and", "but" — they're natural sentence
    # connectors used by genuine substantive insights ("So stable coins like
    # USDC ..." was a real technical claim, not filler). The remaining prefixes
    # are pure conversational filler (yeah/uh/well/etc.) with low collision
    # risk on real insight starts.
    "yeah",
    "yep",
    "nope",
    "okay",
    "ok",
    "well",
    "i mean",
    "you know",
    "um",
    "uh",
    "right",
    "exactly",
)

_FIRST_PERSON_PRONOUNS: frozenset[str] = frozenset(
    {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
)

# Bumped from 0.15 → 0.25 after #652 audit. The lower threshold dropped
# substantive first-person CEO/expert claims like "We made our first
# investment before the AI trade started" (3/12 = 0.25 pronoun density —
# right at the old threshold but legitimate analysis content). The higher
# threshold still catches dialogue-heavy insights without false-positiving
# on opinion/analysis bullets.
_PRONOUN_DENSITY_THRESHOLD = 0.25
_QUOTE_COVERAGE_THRESHOLD = 0.60


def _normalize_first_words(text: str, n: int = 3) -> List[str]:
    """Return lowercase first ``n`` tokens (strip punctuation)."""
    tokens = re.findall(r"[A-Za-z']+", text or "")
    return [t.lower() for t in tokens[:n]]


def _starts_with_filler(text: str) -> bool:
    if not text:
        return False
    first_words = _normalize_first_words(text, 3)
    if not first_words:
        return False
    for pref in _FILLER_PREFIXES:
        pref_tokens = pref.split()
        if len(pref_tokens) <= len(first_words):
            if first_words[: len(pref_tokens)] == pref_tokens:
                return True
    return False


def _first_person_density(text: str) -> float:
    tokens = re.findall(r"[A-Za-z']+", text or "")
    if not tokens:
        return 0.0
    pronouns = sum(1 for t in tokens if t.lower() in _FIRST_PERSON_PRONOUNS)
    return pronouns / len(tokens)


def _quote_coverage(insight_text: str, quote_text: Optional[str]) -> float:
    """Fraction of ``insight_text`` character-length that ``quote_text`` covers
    (case-insensitive substring length). Returns 0 when quote is absent."""
    if not insight_text or not quote_text:
        return 0.0
    insight_len = len(insight_text.strip())
    if insight_len == 0:
        return 0.0
    q = quote_text.strip()
    if not q:
        return 0.0
    if q.lower() in insight_text.lower():
        return len(q) / insight_len
    return 0.0


def insight_looks_like_dialogue(insight_text: str, quote_text: Optional[str] = None) -> bool:
    """Return True when an insight is likely dialogue/filler rather than a
    distilled third-person claim.

    Any one of the three rules is sufficient:

    * Starts with a conversational filler token (yeah/okay/well/so/…).
    * First-person pronoun density > 0.15.
    * A verbatim quote covers > 60 % of the insight text.
    """
    if not insight_text:
        return False
    if _starts_with_filler(insight_text):
        return True
    if _first_person_density(insight_text) > _PRONOUN_DENSITY_THRESHOLD:
        return True
    if _quote_coverage(insight_text, quote_text) > _QUOTE_COVERAGE_THRESHOLD:
        return True
    return False


# ---------------------------------------------------------------------------
# Public entry points used by gi/pipeline.py
# ---------------------------------------------------------------------------


def apply_insight_filters(
    insights: Sequence[dict],
    *,
    transcript_window_by_index: Optional[dict[int, str]] = None,
) -> Tuple[List[dict], int, int]:
    """Apply the two filters to a list of insight dicts.

    Args:
        insights: Each dict has at least ``text``; may also have ``quote`` /
            ``quote_text`` for the dialogue filter and a resolvable transcript
            source window via ``transcript_window_by_index``.
        transcript_window_by_index: Optional mapping of positional index → the
            transcript window the insight came from (used by the ad filter).

    Returns:
        ``(kept_insights, ads_dropped_count, dialogue_dropped_count)``.
    """
    kept: List[dict] = []
    ads_dropped = 0
    dialogue_dropped = 0
    for i, ins in enumerate(insights):
        text = str(ins.get("text") or "").strip()
        if not text:
            kept.append(ins)  # let downstream validation drop empties
            continue
        window = None
        if transcript_window_by_index is not None:
            window = transcript_window_by_index.get(i)
        if insight_looks_like_ad(text, window):
            ads_dropped += 1
            continue
        quote = ins.get("quote") or ins.get("quote_text")
        if insight_looks_like_dialogue(text, quote):
            dialogue_dropped += 1
            continue
        kept.append(ins)
    return kept, ads_dropped, dialogue_dropped


__all__ = [
    "apply_insight_filters",
    "insight_looks_like_ad",
    "insight_looks_like_dialogue",
]
