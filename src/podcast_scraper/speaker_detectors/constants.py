"""Thresholds, defaults, and pattern lists for NER-based speaker detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..languages import primary_language, TARGET_LANGUAGE

# Default speaker names when detection fails (Issue #428: use typed placeholder, not "Guest")
DEFAULT_SPEAKER_NAMES = ["Host", "unknown_guest_1"]

_VALID_MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")

MAX_MODEL_NAME_LENGTH = 100
MIN_NAME_LENGTH = 2
MIN_RAW_NAME_LENGTH = 2
MIN_SEGMENT_LENGTH = 2

DEFAULT_CONFIDENCE_SCORE = 1.0
PATTERN_BASED_CONFIDENCE_SCORE = 0.7

DESCRIPTION_SNIPPET_LENGTH = 500
# Transcript-intro window scanned for guests with the SAME NER + interview-indicator logic used on
# the feed description — the opening few minutes name the guests the feed metadata often omits.
INTRO_SNIPPET_LENGTH = 3000
DEFAULT_SAMPLE_SIZE = 5
MIN_SPEAKERS_REQUIRED = 2


@dataclass(frozen=True)
class SpeakerCuePatterns:
    """One language's four guest-cue lists, which are only safe to use together.

    Bundled rather than fetched one at a time because the pair that matters is RECALL
    (``leading`` / ``trailing`` / ``trailing_gapped``) against GUARD (``mentioned_only``): take
    the cues without the guard and every name a description merely mentions becomes a candidate
    guest. A single lookup makes the half-use awkward to write by accident.
    """

    leading: Tuple[str, ...]
    trailing: Tuple[str, ...]
    trailing_gapped: Tuple[str, ...]
    mentioned_only: Tuple[str, ...]


#: Guest-introduction cues, KEYED BY THE LANGUAGE OF THE TEXT BEING SCANNED.
#:
#: WHY A MAP. Each entry is a phrase in one language, so a flat list is a list in English wearing
#: no label — and read against a Spanish description it matches nothing, which the caller cannot
#: distinguish from "this episode has no guest". Shape follows
#: :data:`podcast_scraper.search.query_language._MARKERS` and
#: :data:`podcast_scraper.gi.filters.AD_PATTERNS_BY_LANGUAGE`.
#:
#: WHY THIS IS LATENT, NOT A LIVE BUG. Under D-44 the canonical transcript body is always the
#: ANALYSIS language, and feed descriptions reach this code after the same path, so the four
#: module-level names below resolve to the English rows — byte-for-byte the lists that shipped.
#: The map is what makes a non-English analysis language a DATA change.
#:
#: HONEST PROVENANCE. The English rows are MEASURED, and the measurements are quoted in the
#: comments inside them (fire counts and precision against a 1,400-episode ground-truth set). The
#: five non-English rows are TRANSLATIONS I authored from the English categories; no non-English
#: description or ASR intro has ever been run against them. They are a starting vocabulary.
#:
#: THE FOUR LISTS MOVE TOGETHER — see :data:`SPEAKER_CUE_LANGUAGES`. Adding cues for a language
#: without adding its mentioned-only GUARD would strip the precision half and keep the recall
#: half, which is how a person the episode is merely about becomes a diarized voice's name.
INTERVIEW_INDICATOR_PATTERNS_BY_LANGUAGE: Dict[str, List[str]] = {
    "en": [
        r"interview(?:ed|ing|s)?\s+(?:with\s+)?",
        # "we ARE joined by" / "I AM joined by" — the plain forms were missing, and they are the
        # most common way a description introduces a guest.
        r"(?:we(?:'re|'ve|\s+are|\s+have)?|i(?:'m|'ve|\s+am|\s+have)?)\s+"
        r"(?:been\s+)?joined\s+by\s+",
        r"speaks?\s+(?:with|to)\s+",
        r"speaking\s+(?:with|to)\s+",
        r"talking\s+(?:with|to)\s+",
        r"talks?\s+(?:with|to)\s+",
        r"conversation\s+with\s+",
        r"guest(?:s)?(?:\s*:|\s+is|\s+are)?\s*",
        r"featuring\s+",
        r"(?:special\s+)?guest\s+",
        r"welcomes?\s+",
        r"sits?\s+down\s+with\s+",
        r"chats?\s+with\s+",
        r"joining\s+us\s+",
        # Third-person passive — "Elena Burger is joined by a16z's Andy McCall". The list only had
        # the first-person form ("we're joined by"), so every show that writes its blurb in the
        # third person was invisible. 48 fires, 77.1% a real speaker on the ground-truth set.
        r"(?:is|are|was|were)\s+joined\s+by\s+",
        r"deep\s+dive\s+with\s+",
        # A panel: "speaks with Chris Miller, author of Chip War, AND WITH analyst Stacy Rasgon".
        # The leading cue only reaches the first name; the second guest is coordinated onto it.
        r"(?:and|along)\s+with\s+",
    ],
    "es": [
        r"entrevista\s+(?:con|a)\s+",
        r"entrevista(?:mos|n)\s+a\s+",
        r"(?:nos\s+)?acompa(?:ñ|n)a\s+",
        r"habla(?:mos|n)?\s+con\s+",
        r"conversa(?:mos|n)?\s+con\s+",
        r"charla(?:mos|n)?\s+con\s+",
        r"(?:en\s+)?conversaci(?:ó|o)n\s+con\s+",
        r"(?:nuestr[oa]s?\s+)?invitad[oa]s?(?:\s*:|\s+es|\s+son)?\s*",
        r"con\s+la\s+participaci(?:ó|o)n\s+de\s+",
        r"recibimos\s+a\s+",
        r"damos\s+la\s+bienvenida\s+a\s+",
        r"se\s+une\s+a\s+nosotros\s+",
        r"junto\s+(?:a|con)\s+",
        # FIRST PERSON SINGULAR. Every fixture introduces the guest this way — "con me c'è
        # Marco", "bei mir ist Stefan", "comigo está Rafael" — and every row here originally
        # had only the plural form, so four of five languages scored ZERO introduction cues on
        # content that plainly introduces a guest. Added 2026-10-02 from that measurement.
        r"conmigo\s+(?:est(?:á|a)|hoy)\s+",
        # The coordinated second guest, as English's "(?:and|along) with": "hablamos con A, de X,
        # y con B" — the leading cue reaches only A. MEASURED on 714 real El Hilo / Radio
        # Ambulante descriptions (2026-10-09): 34 names newly introduced, 24 that episode's
        # guests, 10 organisations (refused by the person check), none a merely-mentioned person.
        r"(?:y|tambi(?:é|e)n)\s+con\s+",
    ],
    "it": [
        r"intervista\s+(?:con|a)\s+",
        r"intervist(?:iamo|a)\s+",
        r"insieme\s+a\s+",
        r"parl(?:iamo|a)\s+con\s+",
        r"(?:in\s+)?conversazione\s+con\s+",
        r"chiacchier(?:iamo|a)\s+con\s+",
        r"(?:i\s+nostri\s+|il\s+nostro\s+)?ospit[ei](?:\s*:|\s+(?:è|e'|e)|\s+sono)?\s*",
        r"con\s+la\s+partecipazione\s+di\s+",
        r"diamo\s+il\s+benvenuto\s+a\s+",
        r"si\s+unisce\s+a\s+noi\s+",
        r"ospite\s+di\s+oggi\s+",
        r"con\s+me\s+c'(?:è|e)\s+",
    ],
    "fr": [
        r"(?:entretien|interview)\s+avec\s+",
        r"nous\s+recevons\s+",
        r"(?:nous\s+)?parl(?:ons|e)\s+avec\s+",  # codespell:ignore ons
        r"(?:en\s+)?conversation\s+avec\s+",
        r"(?:notre|nos)\s+invit(?:é|e)e?s?(?:\s*:|\s+est|\s+sont)?\s*",
        r"invit(?:é|e)e?\s*:\s*",
        r"avec\s+la\s+participation\s+de\s+",
        r"nous\s+accueillons\s+",
        r"(?:nous\s+)?rejoint\s+",
        r"(?:s'entretient|discute)\s+avec\s+",
        r"aux\s+c(?:ô|o)t(?:é|e)s\s+de\s+",
        r"je\s+re(?:ç|c)ois\s+",  # codespell:ignore ois
    ],
    # GERMAN IS V2, so the subject follows the verb whenever anything is fronted: "in dieser Folge
    # SPRECHEN WIR MIT Brian Chesky" is the ordinary shape, not an inversion to handle later. Every
    # verb cue here therefore allows both orders — the subject-first form alone missed the sentence
    # German descriptions actually write.
    "de": [
        r"(?:interview|gespr(?:ä|a)ch)\s+mit\s+",
        r"im\s+gespr(?:ä|a)ch\s+mit\s+",
        r"zu\s+gast\s+ist\s+",
        r"(?:unser|unsere)\s+"  # codespell:ignore unser
        r"g(?:ä|a)st(?:e|in)?(?:\s*:|\s+ist|\s+sind)?\s*",
        r"(?:wir\s+sprechen|sprechen\s+wir|spricht)\s+mit\s+",
        r"(?:wir\s+begr(?:ü|u)(?:ß|ss)en|begr(?:ü|u)(?:ß|ss)en\s+wir|begr(?:ü|u)(?:ß|ss)t)\s+",
        r"(?:gemeinsam|zusammen)\s+mit\s+",
        r"unterh(?:ä|a)lt\s+sich\s+mit\s+",
        r"(?:wir\s+haben|haben\s+wir)\s+(?:heute\s+)?zu\s+gast\s+",
        r"g(?:a|ä)ste?\s*:\s*",
        r"bei\s+(?:mir|uns)\s+ist\s+",
    ],
    "pt": [
        r"entrevista\s+com\s+",
        r"entrevista(?:mos|m)\s+",
        r"convers(?:amos|a)\s+com\s+",  # codespell:ignore convers
        r"fala(?:mos|m)?\s+com\s+",
        r"(?:em\s+)?conversa\s+com\s+",
        r"(?:nossos?\s+|nossas?\s+)?convidad[oa]s?(?:\s*:|\s+(?:é|e)|\s+s(?:ã|a)o)?\s*",
        r"com\s+a\s+participa(?:ç|c)(?:ã|a)o\s+de\s+",
        r"recebemos\s+",
        r"damos\s+as\s+boas-vindas\s+a\s+",
        r"se\s+junta\s+a\s+n(?:ó|o)s\s+",
        r"ao\s+lado\s+de\s+",
        r"(?:comigo|conosco)\s+est(?:á|a)\s+",
    ],
}

#: Cues that come AFTER the name ("Dr. Adam Rodman ... returns"), glued directly to it.
#:
#: The leading map above only matches a cue BEFORE the name, which is why the real guest of
#: "OpenAI's Big Reset" was invisible to the safe path — the description introduces him as "the
#: A.I. researcher Dr. Adam Rodman, of Harvard Medical School, returns to discuss...". A guest the
#: detector cannot see leaves a voice cluster free for a mentioned celebrity to claim.
INTERVIEW_TRAILING_PATTERNS_BY_LANGUAGE: Dict[str, List[str]] = {
    "en": [
        r"\s*,?\s*(?:of|from|at)\s+[\w .'-]{2,40},?\s+returns?\b",
        r"\s*,?\s*returns?\s+to\s+(?:discuss|talk|explain|join)",
        r"\s*,?\s*(?:is\s+back|rejoins?|comes?\s+back)\b",
        r"\s*,?\s*joins?\s+(?:us|the\s+show|me)\b",
    ],
    "es": [
        r"\s*,?\s*(?:de|desde|en)\s+[\w .'-]{2,40},?\s+(?:vuelve|regresa)\b",
        r"\s*,?\s*(?:vuelve|regresa)\s+para\s+(?:hablar|contar|explicar|comentar)",
        r"\s*,?\s*(?:est(?:á|a)\s+de\s+vuelta|regresa|vuelve)\b",
        r"\s*,?\s*(?:se\s+une\s+a\s+nosotros|nos\s+acompa(?:ñ|n)a)\b",
    ],
    "it": [
        r"\s*,?\s*(?:di|da|presso)\s+[\w .'-]{2,40},?\s+(?:torna|ritorna)\b",
        r"\s*,?\s*(?:torna|ritorna)\s+(?:per|a)\s+(?:parlare|raccontare|spiegare)",
        r"\s*,?\s*(?:(?:è|e'|e)\s+di\s+nuovo\s+con\s+noi|torna|ritorna)\b",
        r"\s*,?\s*si\s+unisce\s+a\s+noi\b",
    ],
    "fr": [
        r"\s*,?\s*(?:de|du|chez)\s+[\w .'-]{2,40},?\s+revient\b",
        r"\s*,?\s*revient\s+(?:pour|nous)\s+(?:parler|raconter|expliquer)",
        r"\s*,?\s*(?:est\s+de\s+retour|revient)\b",
        r"\s*,?\s*nous\s+rejoint\b",
    ],
    "de": [
        r"\s*,?\s*(?:von|aus|bei)\s+[\w .'-]{2,40},?\s+"
        r"(?:ist\s+zur(?:ü|u)ck|kehrt\s+zur(?:ü|u)ck)\b",
        r"\s*,?\s*(?:ist|kehrt)\s+zur(?:ü|u)ck,?\s+um\s+(?:(?:ü|u)ber|zu)\b",
        r"\s*,?\s*(?:ist\s+zur(?:ü|u)ck|kehrt\s+zur(?:ü|u)ck)\b",
        r"\s*,?\s*ist\s+(?:zu\s+gast|bei\s+uns)\b",
    ],
    "pt": [
        r"\s*,?\s*(?:de|do|da|em)\s+[\w .'-]{2,40},?\s+(?:volta|retorna)\b",
        r"\s*,?\s*(?:volta|retorna)\s+para\s+(?:falar|contar|explicar|comentar)",
        r"\s*,?\s*(?:est(?:á|a)\s+de\s+volta|volta|retorna)\b",
        r"\s*,?\s*se\s+junta\s+a\s+n(?:ó|o)s\b",
    ],
}

#: Trailing cues matched with a BOUNDED GAP after the name — ``NAME <role clause> CUE``.
#:
#: WHY A SECOND LIST. Every pattern in the map above is glued to the name (``name + pattern``),
#: which only works when the cue is immediately adjacent. Real episode descriptions put the guest's
#: job title in between: "Sarah Laszlo, senior director of Visa's machine learning platform, joins
#: the AI Podcast", "Mike Pritchard, Director of Climate Simulation Research at NVIDIA, discusses".
#: These are matched as ``name + gap + cue`` instead.
#:
#: DIRECTION IS WHAT MAKES ``discusses`` SAFE HERE. The same verb appears in
#: :data:`MENTIONED_ONLY_PATTERNS`, and that is not a contradiction: mentioned-only is matched
#: cue-BEFORE-name ("discusses Mike Pritchard" — he is the topic), this is matched
#: name-BEFORE-cue ("Mike Pritchard ... discusses" — he is speaking). The two can never fire on the
#: same text in the same direction. Every non-English row below preserves that same direction
#: split, which is the property that makes the pair safe rather than the specific verbs.
#:
#: MEASURED against 1,400 episodes whose roster already names a real guest, so a wrong pick is a
#: genuine error rather than a roster gap:
#:     NAME joins ...................... 46 fires, 82.6% a real speaker
#:     NAME, <role>, discusses ......... 49 fires, 75.5%
#: The looser "NAME ... discusses anywhere within 60 chars" scored 58.1% and is NOT included.
INTERVIEW_TRAILING_GAPPED_PATTERNS_BY_LANGUAGE: Dict[str, List[str]] = {
    "en": [
        r",?\s*joins?\b",
        r",?\s*(?:discusses|explains|shares|unpacks|breaks\s+down)\b",
        r",?\s*(?:tells|speaks?\s+(?:with|to)|sits?\s+down\s+with)\b",
    ],
    "es": [
        r",?\s*(?:se\s+une|nos\s+acompa(?:ñ|n)a)\b",
        r",?\s*(?:explica|analiza|comparte|cuenta|desgrana)\b",
        r",?\s*(?:nos\s+dice|habla\s+con|conversa\s+con)\b",
    ],
    "it": [
        r",?\s*si\s+unisce\b",
        r",?\s*(?:spiega|analizza|condivide|racconta)\b",
        r",?\s*(?:ci\s+dice|parla\s+con|chiacchiera\s+con)\b",
    ],
    "fr": [
        r",?\s*(?:nous\s+)?rejoint\b",
        r",?\s*(?:explique|analyse|partage|raconte|d(?:é|e)crypte)\b",
        r",?\s*(?:nous\s+dit|parle\s+avec|s'entretient\s+avec)\b",
    ],
    "de": [
        r",?\s*(?:ist\s+dabei|kommt\s+dazu)\b",
        r",?\s*(?:erkl(?:ä|a)rt|analysiert|teilt|erz(?:ä|a)hlt|berichtet)\b",
        r",?\s*(?:sagt\s+uns|spricht\s+mit|unterh(?:ä|a)lt\s+sich\s+mit)\b",
    ],
    "pt": [
        r",?\s*se\s+junta\b",
        r",?\s*(?:explica|analisa|compartilha|conta|destrincha)\b",
        r",?\s*(?:nos\s+diz|fala\s+com|conversa\s+com)\b",
    ],
}

#: The PRECISION GUARD: cues that mark a name as talked ABOUT rather than talking. Matched
#: cue-BEFORE-name, the opposite direction from the gapped map above.
#:
#: This is the half that must never be missing for a language whose cue list exists — see
#: :data:`SPEAKER_CUE_LANGUAGES`. Cues without a guard is pure recall with nothing to stop a
#: lawsuit defendant becoming a podcast guest (#876).
MENTIONED_ONLY_PATTERNS_BY_LANGUAGE: Dict[str, List[str]] = {
    "en": [
        r"about\s+",
        r"on\s+\w+(?:'s)?\s+",
        r"discuss(?:es|ing|ed)?\s+",
        r"analysis\s+of\s+",
        r"according\s+to\s+",
        r"(?:he|she|they)\s+says?\s+",
        r"'s\s+(?:\w+\s+)*(?:policy|plan|speech|decision|statement)",
        r"(?:the\s+)?(?:president|ceo|senator|governor)\s+",
        r"covers?\s+",
        r"examines?\s+",
        r"looks?\s+at\s+",
        r"(?:news|story|report)\s+(?:about|on)\s+",
    ],
    "es": [
        r"sobre\s+",
        r"acerca\s+de\s+",
        r"(?:analiza|analizamos)\s+(?:a\s+)?",
        r"an(?:á|a)lisis\s+de\s+",
        r"seg(?:ú|u)n\s+",
        r"(?:(?:é|e)l|ella|ellos)\s+dice[ns]?\s+",
        r"(?:la|el)\s+(?:pol(?:í|i)tica|plan|discurso|decisi(?:ó|o)n|declaraci(?:ó|o)n)\s+de\s+",
        r"(?:el\s+|la\s+)?" r"(?:presidente|presidenta|director[a]?|senador[a]?|gobernador[a]?)\s+",
        r"(?:cubre|examina|repasa)\s+",
        r"(?:noticia|historia|informe)\s+sobre\s+",
    ],
    "it": [
        r"su\s+",
        r"a\s+proposito\s+di\s+",
        r"(?:analizza|analizziamo)\s+",
        r"analisi\s+di\s+",
        r"secondo\s+",
        r"(?:lui|lei|loro)\s+dic(?:e|ono)\s+",
        r"(?:la|il)\s+(?:politica|piano|discorso|decisione|dichiarazione)\s+di\s+",
        r"(?:il\s+|la\s+)?(?:presidente|amministratore|senatore|governatore)\s+",
        r"(?:copre|esamina|ripercorre)\s+",
        r"(?:notizia|storia|rapporto)\s+su\s+",
    ],
    "fr": [
        r"(?:à|a)\s+propos\s+de\s+",
        r"au\s+sujet\s+de\s+",
        r"(?:analyse|analysons)\s+",
        r"analyse\s+de\s+",
        r"selon\s+",
        r"(?:il|elle|ils|elles)\s+di(?:t|sent)\s+",
        r"(?:la|le)\s+(?:politique|plan|discours|d(?:é|e)cision|d(?:é|e)claration)\s+de\s+",
        r"(?:le\s+|la\s+)?"
        r"(?:pr(?:é|e)sident[e]?|directeur|directrice|s(?:é|e)nateur|gouverneur)\s+",
        r"(?:couvre|examine|revient\s+sur)\s+",
        r"(?:nouvelle|histoire|rapport)\s+sur\s+",
    ],
    "de": [
        r"(?:ü|u)ber\s+",
        r"zum\s+thema\s+",
        r"(?:analysiert|analysieren)\s+",
        r"analyse\s+von\s+",
        r"laut\s+",
        r"(?:er|sie)\s+sagt\s+",  # codespell:ignore sie
        r"(?:die|der|das)\s+(?:politik|plan|rede|entscheidung|erkl(?:ä|a)rung)\s+von\s+",
        r"(?:der\s+|die\s+)?(?:pr(?:ä|a)sident(?:in)?"
        r"|gesch(?:ä|a)ftsf(?:ü|u)hrer(?:in)?|senator(?:in)?|gouverneur(?:in)?)\s+",
        r"(?:behandelt|untersucht|blickt\s+auf)\s+",
        r"(?:nachricht|geschichte|bericht)\s+(?:(?:ü|u)ber|zu)\s+",
    ],
    "pt": [
        r"sobre\s+",
        r"a\s+respeito\s+de\s+",
        r"(?:analisa|analisamos)\s+",
        r"an(?:á|a)lise\s+de\s+",
        r"segundo\s+",
        r"(?:ele|ela|eles|elas)\s+diz(?:em)?\s+",  # codespell:ignore eles
        r"(?:a|o)\s+(?:pol(?:í|i)tica|plano|discurso|decis(?:ã|a)o|declara(?:ç|c)(?:ã|a)o)\s+de\s+",
        r"(?:o\s+|a\s+)?(?:presidente|diretor[a]?|senador[a]?|governador[a]?)\s+",
        r"(?:cobre|examina|revisita)\s+",
        r"(?:not(?:í|i)cia|hist(?:ó|o)ria|relat(?:ó|o)rio)\s+sobre\s+",
    ],
}

#: Languages where ALL FOUR cue lists exist, so the recall cues and the mentioned-only guard are
#: both present. Derived by intersection rather than written down: a language half-added to the
#: maps above is simply not advertised here, which fails closed. The alternative — a list of names
#: — would claim support the data does not back.
SPEAKER_CUE_LANGUAGES = frozenset(
    set(INTERVIEW_INDICATOR_PATTERNS_BY_LANGUAGE)
    & set(INTERVIEW_TRAILING_PATTERNS_BY_LANGUAGE)
    & set(INTERVIEW_TRAILING_GAPPED_PATTERNS_BY_LANGUAGE)
    & set(MENTIONED_ONLY_PATTERNS_BY_LANGUAGE)
)


def interview_cue_patterns_for(language: Optional[str]) -> Optional[SpeakerCuePatterns]:
    """All four cue lists for ``language``, or ``None`` when we have no vocabulary for it.

    ``None`` rather than an English fallback: English cues read against Spanish match nothing,
    and "no guest found" is indistinguishable from "I do not speak this episode" to every caller
    downstream. ``None`` is a state a caller can branch on.
    """
    if not language:
        return None
    code = primary_language(language)
    if code not in SPEAKER_CUE_LANGUAGES:
        return None
    return SpeakerCuePatterns(
        leading=tuple(INTERVIEW_INDICATOR_PATTERNS_BY_LANGUAGE[code]),
        trailing=tuple(INTERVIEW_TRAILING_PATTERNS_BY_LANGUAGE[code]),
        trailing_gapped=tuple(INTERVIEW_TRAILING_GAPPED_PATTERNS_BY_LANGUAGE[code]),
        mentioned_only=tuple(MENTIONED_ONLY_PATTERNS_BY_LANGUAGE[code]),
    )


#: The analysis-language rows, resolved once: what a caller that passes no language reads. The
#: canonical body is always the analysis language under D-44, but the FEED DESCRIPTION is not —
#: D-44 translates the transcript, not the feed — so description readers take the feed's language
#: (`guests._cues`). Assuming otherwise rejected both of El Hilo's guests (2026-10-09).
INTERVIEW_INDICATOR_PATTERNS = INTERVIEW_INDICATOR_PATTERNS_BY_LANGUAGE[TARGET_LANGUAGE]
INTERVIEW_TRAILING_PATTERNS = INTERVIEW_TRAILING_PATTERNS_BY_LANGUAGE[TARGET_LANGUAGE]
INTERVIEW_TRAILING_GAPPED_PATTERNS = INTERVIEW_TRAILING_GAPPED_PATTERNS_BY_LANGUAGE[TARGET_LANGUAGE]
MENTIONED_ONLY_PATTERNS = MENTIONED_ONLY_PATTERNS_BY_LANGUAGE[TARGET_LANGUAGE]
