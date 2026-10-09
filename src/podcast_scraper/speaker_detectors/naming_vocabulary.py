"""Per-language vocabulary for speaker naming — one home for every word list and cue phrase.

WHY THIS MODULE EXISTS. `hosts.py` grew to 3,200 lines carrying ~35 flat English collections:
honorifics, job titles, role words, network names, sentence-opener stopwords, host-statement
phrases. That is the correct shape for a one-language corpus and the wrong one here — the corpus
carries fifteen non-English episodes, and `constants.py` already established the house pattern of
a `*_BY_LANGUAGE` map in its own module for the interview cues.

WHY IT IS NOT LATENT. Under D-44 the canonical transcript body is always the ANALYSIS language,
so transcript-path detectors are right to read the English rows. The DESCRIPTION path is not:
nothing translates feed metadata (the same reason S2.14 exists), so a Spanish show's title,
description and author tags reach these collections in Spanish. Measured on 2026-10-03, before
this existed: `is_publishable_speaker_name` rejected "Host Mike", "Host" and "Tech Summit" and
ACCEPTED "Anfitrión Miguel", "Anfitrión" and "Cumbre Tecnología" — so a Spanish feed minted a
person called "Anfitrión Miguel", and a bare role word became a person with a `SPOKEN_BY` edge.
That is §5.2's phantom-person failure reached through a path nobody had looked at.

WHAT IS MEASURED AND WHAT IS NOT, because the distinction decides how much to trust a row:

* the ENGLISH row of every map is the measured one. It is what #2269's gold development set was
  scored against, and the comments in `hosts.py` that cite a show by name are describing it.
* the five non-English rows are TRANSLATIONS OF THE ENGLISH CATEGORIES. No non-English
  conversation has been scored against any of them. They are authored, not observed.

Translated, not transliterated. Gendered forms are spelled out (`anfitrión`/`anfitriona`,
`modérateur`/`modératrice`, `fondatore`/`fondatrice`) because a row carrying only the masculine
form sees half the people. Where a construction does not exist in a language the value is `None`
or the English row, stated at the map rather than left to be inferred.

WHAT BELONGS HERE AND WHAT DOES NOT. Prose belongs here: words and phrases a human says. Structure
does not: `_STATED_UC` (a Unicode capital), `_SHOW_TITLE_PAREN` (a parenthesis), `_NOT_IN_A_NAME`
(punctuation) are language-independent and stay in `hosts.py`. A proper noun is a third case —
brand and network names are spelled the same everywhere, so those rows are the English row PLUS
each market's own outlets, not a translation.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Optional, Tuple

from ..languages import primary_language, TARGET_LANGUAGE

#: Tier-1: the five enabled non-English languages plus the analysis language. Every map below
#: carries a row for each, and `hosts.NAMING_VOCABULARY_LANGUAGES` derives the advertised set as
#: an INTERSECTION so a half-added language fails closed rather than half-working.
TIER_1_LANGUAGES: Tuple[str, ...] = ("en", "es", "it", "fr", "de", "pt")


def primary_subtag(language: Optional[str]) -> str:
    """``"es-ES"`` -> ``"es"``. NORMALISED HERE so no call site has to remember.

    Feeds carry a full BCP-47 tag and these maps are keyed by the primary subtag. A caller that
    forgot the split would silently receive the English row for a Spanish feed, which is the exact
    failure this module exists to close — so the split happens once, on the way in.
    """
    return primary_language(language)


def vocabulary_row(rows: Dict[str, Any], language: Optional[str], *, default: Any = None) -> Any:
    """One language's row out of *rows*, for a module that has to DO something with it.

    TWO CASES, and conflating them is the bug this function exists to prevent:

    * ``language`` is None — nothing resolved for this episode. `transcription_language` returns
      None as an honest "let the engine decide", and the transcript we are reading is then
      whatever the analysis language is, so the TARGET_LANGUAGE row is the right answer and is
      also exactly what every one of these call sites did before it took a language at all.
    * ``language`` resolved to something with NO row — "ja", "ko", "ar". Here the answer is
      *default* (``None``, or an empty set where the caller needs a container), NOT English.
      Running English cue patterns over Japanese is how a confident wrong name gets made; the
      English rows are gold-gated against an English development set and mean nothing off it.

    This is deliberately NOT :func:`hosts.naming_vocabulary_for`, which answers the different
    question "does this language have vocabulary at all" and so returns None in both cases.
    """
    if language is None:
        return rows.get(TARGET_LANGUAGE, default)
    row = rows.get(primary_subtag(language))
    return default if row is None else row


# =============================================================================================
# ROLE AND JOB WORDS
# =============================================================================================

#: Role words that may stand in FRONT of a name ("Host Mike", "Guest Host Tim"). A name made only
#: of these is not a person. Inside a longer real name they are NOT rejected ("Christopher
#: Guest" ends with one, so it is unaffected) — the check is positional, not a substring scan.
LEADING_ROLE_WORDS: Dict[str, FrozenSet[str]] = {
    "en": frozenset(
        {
            "host",
            "co-host",
            "cohost",
            "guest",
            "speaker",
            "narrator",
            "announcer",
            "reporter",
            "producer",
            "mister",
            "presenter",
            "moderator",
        }
    ),
    "es": frozenset(
        {
            "anfitrión",
            "anfitriona",
            "coanfitrión",
            "coanfitriona",
            "presentador",
            "presentadora",
            "conductor",
            "conductora",
            "invitado",
            "invitada",
            "ponente",
            "narrador",
            "narradora",
            "locutor",
            "locutora",
            "reportero",
            "reportera",
            "productor",
            "productora",
            "señor",
            "moderador",
            "moderadora",
        }
    ),
    "it": frozenset(
        {
            "conduttore",
            "conduttrice",
            "coconduttore",
            "coconduttrice",
            "presentatore",
            "presentatrice",
            "ospite",
            "relatore",
            "relatrice",
            "narratore",
            "narratrice",
            "annunciatore",
            "annunciatrice",
            "giornalista",
            "produttore",
            "produttrice",
            "signor",
            "signore",
            "moderatore",
            "moderatrice",
        }
    ),
    "fr": frozenset(
        {
            "animateur",
            "animatrice",
            "coanimateur",
            "coanimatrice",
            "présentateur",
            "présentatrice",
            "invité",
            "invitée",
            "intervenant",
            "intervenante",
            "narrateur",
            "narratrice",
            "annonceur",
            "reporter",
            "producteur",
            "productrice",
            "monsieur",
            "modérateur",
            "modératrice",
        }
    ),
    "de": frozenset(
        {
            "gastgeber",
            "gastgeberin",
            "moderator",
            "moderatorin",
            "komoderator",
            "komoderatorin",
            "gast",
            "sprecher",
            "sprecherin",
            "erzähler",
            "erzählerin",
            "ansager",
            "ansagerin",
            "reporter",
            "reporterin",
            "produzent",
            "produzentin",
            "herr",
            "präsentator",
            "präsentatorin",
        }
    ),
    "pt": frozenset(
        {
            "anfitrião",
            "anfitriã",
            "coanfitrião",
            "coanfitriã",
            "apresentador",
            "apresentadora",
            "convidado",
            "convidada",
            "orador",
            "oradora",
            "narrador",
            "narradora",
            "locutor",
            "locutora",
            "repórter",
            "produtor",
            "produtora",
            "senhor",
            "moderador",
            "moderadora",
        }
    ),
}

#: Role words PLUS conversational filler. A one-word name made of any of these is not a person,
#: and neither is a name made of nothing else — but inside a longer name they are real surnames
#: and given names (Christopher Guest, Ok Taecyeon), which is why the check is "all tokens".
#:
#: The ENGLISH row is MEASURED, on the published corpus 2026-10-02: "Host" reached 20 voices
#: of The Flip (the episode description says "Host: ..." and the label was read as a name),
#: plus "OK" x3, "Thank" and "Right" from self-introductions. The five other rows are
#: AUTHORED — the equivalent measurement needs a non-English corpus we have not run yet.
ROLE_OR_FILLER_TOKENS: Dict[str, FrozenSet[str]] = {
    "en": frozenset(
        {
            "host",
            "hosts",
            "cohost",
            "co-host",
            "guest",
            "guests",
            "speaker",
            "narrator",
            "announcer",
            "moderator",
            "interviewer",
            "ok",
            "okay",
            "right",
            "sorry",
            "sure",
            "thank",
            "thanks",
            "yeah",
            "yep",
            "hello",
            "hi",
        }
    ),
    "es": frozenset(
        {
            "anfitrión",
            "anfitriona",
            "presentador",
            "presentadora",
            "invitado",
            "invitada",
            "invitados",
            "ponente",
            "narrador",
            "locutor",
            "moderador",
            "entrevistador",
            "entrevistadora",
            "vale",
            "bueno",
            "claro",
            "perdón",
            "perdona",
            "gracias",
            "sí",
            "ya",
            "hola",
        }
    ),
    "it": frozenset(
        {
            "conduttore",
            "conduttrice",
            "presentatore",
            "presentatrice",
            "ospite",
            "ospiti",
            "relatore",
            "narratore",
            "annunciatore",
            "moderatore",
            "intervistatore",
            "intervistatrice",
            "ok",
            "bene",
            "certo",
            "scusa",
            "scusi",
            "grazie",
            "sì",
            "già",
            "ciao",
            "salve",
        }
    ),
    "fr": frozenset(
        {
            "animateur",
            "animatrice",
            "présentateur",
            "présentatrice",
            "invité",
            "invitée",
            "invités",
            "intervenant",
            "narrateur",
            "annonceur",
            "modérateur",
            "intervieweur",
            "intervieweuse",
            "ok",
            "bien",
            "pardon",
            "désolé",
            "désolée",
            "merci",
            "oui",
            "ouais",
            "bonjour",
            "salut",
        }
    ),
    "de": frozenset(
        {
            "gastgeber",
            "gastgeberin",
            "moderator",
            "moderatorin",
            "gast",
            "gäste",
            "sprecher",
            "sprecherin",
            "erzähler",
            "ansager",
            "interviewer",
            "interviewerin",
            "ok",
            "okay",
            "gut",
            "entschuldigung",
            "sorry",
            "klar",
            "danke",
            "ja",
            "genau",
            "hallo",
            "hi",
        }
    ),
    "pt": frozenset(
        {
            "anfitrião",
            "anfitriã",
            "apresentador",
            "apresentadora",
            "convidado",
            "convidada",
            "convidados",
            "orador",
            "narrador",
            "locutor",
            "moderador",
            "entrevistador",
            "entrevistadora",
            "ok",
            "bem",
            "claro",
            "desculpa",
            "desculpe",
            "obrigado",
            "obrigada",
            "sim",
            "pois",
            "olá",
            "oi",
        }
    ),
}

#: Job words that may appear INSIDE a stated name, before the person
#: ("Senior User Experience Specialist Therese Fessenden").
JOB_TITLE_TOKENS: Dict[str, FrozenSet[str]] = {
    "en": frozenset(
        {
            "specialist",
            "director",
            "manager",
            "president",
            "chairman",
            "columnist",
            "correspondent",
            "editor",
            "founder",
            "partner",
            "analyst",
            "senior",
            "executive",
        }
    ),
    "es": frozenset(
        {
            "especialista",
            "director",
            "directora",
            "gerente",
            "presidente",
            "presidenta",
            "columnista",
            "corresponsal",
            "editor",
            "editora",
            "fundador",
            "fundadora",
            "socio",
            "socia",
            "analista",
            "senior",
            "ejecutivo",
            "ejecutiva",
            "redactor",
            "redactora",
        }
    ),
    "it": frozenset(
        {
            "specialista",
            "direttore",
            "direttrice",
            "responsabile",
            "presidente",
            "editorialista",
            "corrispondente",
            "redattore",
            "redattrice",
            "fondatore",
            "fondatrice",
            "socio",
            "socia",
            "analista",
            "senior",
            "dirigente",
        }
    ),
    "fr": frozenset(
        {
            "spécialiste",
            "directeur",
            "directrice",
            "responsable",
            "président",
            "présidente",
            "chroniqueur",
            "chroniqueuse",
            "correspondant",
            "correspondante",
            "rédacteur",
            "rédactrice",
            "fondateur",
            "fondatrice",
            "associé",
            "associée",
            "analyste",
            "senior",
            "cadre",
        }
    ),
    "de": frozenset(
        {
            "spezialist",
            "spezialistin",
            "direktor",
            "direktorin",
            "leiter",
            "leiterin",
            "präsident",
            "präsidentin",
            "kolumnist",
            "kolumnistin",
            "korrespondent",
            "korrespondentin",
            "redakteur",
            "redakteurin",
            "gründer",
            "gründerin",
            "partner",
            "partnerin",
            "analyst",
            "analystin",
            "senior",
            "geschäftsführer",
            "geschäftsführerin",
        }
    ),
    "pt": frozenset(
        {
            "especialista",
            "diretor",
            "diretora",
            "gerente",
            "presidente",
            "colunista",
            "correspondente",
            "editor",
            "editora",
            "fundador",
            "fundadora",
            "sócio",
            "sócia",
            "analista",
            "sénior",
            "executivo",
            "executiva",
            "redator",
            "redatora",
        }
    ),
}

#: Job words that TRAIL a stated name ("Celestin Ntawirema CEO and founder of...").
#:
#: The C-suite abbreviations are borrowed unchanged in all five — a Spanish bio trailing a name
#: says "CEO", not "director ejecutivo" — so every row carries them. Only the founder words move.
#: The English row came from main, 2026-10-03.
TRAILING_JOB_TOKENS: Dict[str, FrozenSet[str]] = {
    "en": frozenset({"ceo", "cto", "coo", "cfo", "founder", "cofounder", "co-founder"}),
    "es": frozenset(
        {"ceo", "cto", "coo", "cfo", "fundador", "fundadora", "cofundador", "cofundadora"}
    ),
    "it": frozenset(
        {"ceo", "cto", "coo", "cfo", "fondatore", "fondatrice", "cofondatore", "cofondatrice"}
    ),
    "fr": frozenset(
        {"ceo", "cto", "coo", "cfo", "fondateur", "fondatrice", "cofondateur", "cofondatrice"}
    ),
    "de": frozenset(
        {"ceo", "cto", "coo", "cfo", "gründer", "gründerin", "mitgründer", "mitgründerin"}
    ),
    "pt": frozenset(
        {"ceo", "cto", "coo", "cfo", "fundador", "fundadora", "cofundador", "cofundadora"}
    ),
}

#: How someone is ADDRESSED, which is not part of their name. "Professor Hannah Fry" -> "Hannah
#: Fry": the roster snaps a self-introduction onto the pool by first name, and "Professor" is not
#: one. Abbreviated and spelled-out forms both, with and without the period stripped by the caller.
HONORIFIC_TITLES: Dict[str, FrozenSet[str]] = {
    "en": frozenset(
        {
            "capt",
            "captain",
            "col",
            "dame",
            "doctor",
            "dr",
            "father",
            "fr",
            "gen",
            "gov",
            "governor",
            "judge",
            "justice",
            "lady",
            "lord",
            "miss",
            "mr",
            "mrs",
            "ms",
            "pres",
            "president",
            "prof",
            "professor",
            "rep",
            "rev",
            "reverend",
            "sen",
            "senator",
            "sgt",
            "sir",
        }
    ),
    "es": frozenset(
        {
            "dr",
            "dra",
            "doctor",
            "doctora",
            "sr",
            "sra",
            "srta",
            "señor",
            "señora",
            "señorita",
            "don",
            "doña",
            "prof",
            "profesor",
            "profesora",
            "padre",
            "presidente",
            "presidenta",
            "senador",
            "senadora",
            "juez",
            "jueza",
            "gobernador",
            "gobernadora",
            "capitán",
            "coronel",
            "general",
            "sargento",
            "reverendo",
            "monseñor",
        }
    ),
    "it": frozenset(
        {
            "dott",
            "dottore",
            # APOCOPATED FORMS. Italian drops the final -e on a title that stands directly in
            # front of a name — "Dottor Rossi", never "Dottore Rossi" — so these are the only
            # forms that ever appear in the position this set is checked against. With just
            # the full forms, "Dottor Luca Moretti Rossi" kept its title through every pass.
            "dottor",
            "professor",
            "monsignor",
            "ingegner",
            "ingegnere",
            "avvocato",
            "avvocata",
            "dottoressa",
            "sig",
            "signor",
            "signora",
            "signorina",
            "don",
            "prof",
            "professore",
            "professoressa",
            "padre",
            "presidente",
            "senatore",
            "senatrice",
            "giudice",
            "governatore",
            "capitano",
            "colonnello",
            "generale",
            "sergente",
            "reverendo",
            "monsignore",
            "onorevole",
        }
    ),
    "fr": frozenset(
        {
            "dr",
            "docteur",
            "docteure",
            "m",
            "mme",
            "mlle",
            "monsieur",
            "madame",
            "mademoiselle",
            "prof",
            "professeur",
            "professeure",
            "père",
            "président",
            "présidente",
            "sénateur",
            "sénatrice",
            "juge",
            "gouverneur",
            "gouverneure",
            "capitaine",
            "colonel",
            "général",
            "sergent",
            "révérend",
            "maître",
        }
    ),
    "de": frozenset(
        {
            "dr",
            "doktor",
            "doktorin",
            "hr",
            "herr",
            "frau",
            "prof",
            "professor",
            "professorin",
            "pfarrer",
            "pater",
            "präsident",
            "präsidentin",
            "senator",
            "senatorin",
            "richter",
            "richterin",
            "gouverneur",
            "hauptmann",
            "oberst",
            "general",
            "feldwebel",
            "reverend",
            "monsignore",
        }
    ),
    "pt": frozenset(
        {
            "dr",
            "dra",
            "doutor",
            "doutora",
            "sr",
            "sra",
            "srta",
            "senhor",
            "senhora",
            "senhorita",
            "dom",
            "dona",
            "prof",
            "professor",
            "professora",
            "padre",
            "presidente",
            "senador",
            "senadora",
            "juiz",
            "juíza",
            "governador",
            "governadora",
            "capitão",
            "coronel",
            "general",
            "sargento",
            "reverendo",
            "monsenhor",
        }
    ),
}

#: Suffixes that trail a name and are not part of it. Roman numerals and the academic
#: abbreviations are shared; the GENERATIONAL words are the ones that move — Spanish and
#: Portuguese say `hijo`/`filho` where English says `Jr`, and Portuguese also uses `Neto`.
NAME_SUFFIXES: Dict[str, FrozenSet[str]] = {
    "en": frozenset(
        {"esq", "esq.", "ii", "iii", "iv", "jr", "jr.", "m.d.", "md", "ph.d.", "phd", "sr", "sr."}
    ),
    "es": frozenset(
        {
            "ii",
            "iii",
            "iv",
            "jr",
            "jr.",
            "sr",
            "sr.",
            "hijo",
            "h.",
            "dr.",
            "lic.",
            "ing.",
            "mtro.",
        }
    ),
    "it": frozenset({"ii", "iii", "iv", "jr", "jr.", "sr", "sr.", "dott.", "ing.", "avv."}),
    "fr": frozenset({"ii", "iii", "iv", "jr", "jr.", "sr", "fils", "dr.", "me", "mtre"}),
    "de": frozenset({"ii", "iii", "iv", "jr", "jr.", "sen", "sen.", "dr.", "dipl.", "mba"}),
    "pt": frozenset(
        {
            "ii",
            "iii",
            "iv",
            "jr",
            "jr.",
            "sr",
            "sr.",
            "júnior",
            "filho",
            "neto",
            "sobrinho",
            "dr.",
            "eng.",
        }
    ),
}


# =============================================================================================
# WORDS THAT ARE NOT NAMES
# =============================================================================================

#: Sentence openers, pronouns, auxiliaries and filler that the ASR capitalised at a turn boundary
#: ("But Sun", "So Nick", bare "But"). A capitalised token from this set is not a person.
#:
#: TRANSLATED BY CATEGORY, NOT WORD FOR WORD. The English row grew from measured ASR artefacts and
#: runs to 161 entries including every apostrophe variant of every contraction. Reproducing that
#: shape in five languages would be guessing at artefacts nobody has observed; what carries over
#: is the CATEGORIES — articles, personal pronouns, the auxiliaries of "to be" and "to have",
#: conjunctions, the common prepositions, and the small set of feeling words a guest opens with
#: ("glad", "excited" -> "encantado", "contento"). Each row covers those and nothing invented
#: beyond them. The apostrophe-variant explosion is deliberately absent: Spanish, Italian,
#: Portuguese and German do not contract with apostrophes the way English does, and French
#: elision (`j'`, `l'`, `d'`) attaches to the NEXT word rather than standing alone, so a
#: French row lists the elided forms that can appear as a standalone token and no more.
NOT_A_NAME_TOKEN: Dict[str, FrozenSet[str]] = {
    "en": frozenset(
        {
            "a",
            "about",
            "afraid",
            "after",
            "also",
            "always",
            "am",
            "an",
            "and",
            "anyway",
            "are",
            "aren't",
            "aren’t",
            "as",
            "at",
            "back",
            "be",
            "because",
            "been",
            "before",
            "but",
            "by",
            "can't",
            "can’t",
            "coming",
            "couldn't",
            "couldn’t",
            "curious",
            "did",
            "didn't",
            "didn’t",
            "does",
            "doesn't",
            "doesn’t",
            "don't",
            "don’t",
            "excited",
            "fine",
            "for",
            "from",
            "glad",
            "going",
            "gonna",
            "good",
            "great",
            "had",
            "happy",
            "has",
            "hasn't",
            "hasn’t",
            "have",
            "haven't",
            "haven’t",
            "he",
            "her",
            "here",
            "here's",
            "here’s",
            "him",
            "his",
            "i",
            "i'd",
            "i'll",
            "i'm",
            "i've",
            "in",
            "into",
            "is",
            "isn't",
            "isn’t",
            "it",
            "it's",
            "its",
            "it’s",
            "i’d",
            "i’ll",
            "i’m",
            "i’ve",
            "just",
            "let's",
            "let’s",
            "look",
            "looking",
            "me",
            "my",
            "not",
            "now",
            "of",
            "okay",
            "on",
            "or",
            "our",
            "out",
            "plus",
            "really",
            "saying",
            "she",
            "shouldn't",
            "shouldn’t",
            "so",
            "sorry",
            "still",
            "sure",
            "talking",
            "telling",
            "that",
            "that's",
            "that’s",
            "the",
            "their",
            "them",
            "then",
            "there",
            "there's",
            "there’s",
            "these",
            "they",
            "they're",
            "they've",
            "they’re",
            "they’ve",
            "thinking",
            "this",
            "those",
            "to",
            "trying",
            "us",
            "very",
            "was",
            "wasn't",
            "wasn’t",
            "we",
            "we'll",
            "we're",
            "we've",
            "well",
            "were",
            "weren't",
            "weren’t",
            "we’ll",
            "we’re",
            "we’ve",
            "with",
            "won't",
            "wondering",
            "won’t",
            "working",
            "worried",
            "wouldn't",
            "wouldn’t",
            "yeah",
            "you",
            "you'd",
            "you'll",
            "you're",
            "you've",
            "your",
            "you’d",
            "you’ll",
            "you’re",
            "you’ve",
        }
    ),
    "es": frozenset(
        {
            # articles and determiners
            "el",
            "la",
            "los",
            "las",
            "un",
            "una",
            "unos",
            "unas",
            "este",
            "esta",
            "estos",
            "estas",
            "ese",
            "esa",
            "esos",
            "esas",
            "mi",
            "mis",
            "tu",
            "tus",
            "su",
            "sus",
            "nuestro",
            "nuestra",
            "nuestros",
            "nuestras",
            # pronouns
            "yo",
            "tú",
            "él",
            "ella",
            "usted",
            "nosotros",
            "nosotras",
            "vosotros",
            "ellos",
            "ellas",
            "ustedes",
            "me",
            "te",
            "se",
            "nos",
            "le",
            "les",
            "lo",
            # to be / to have / to go
            "soy",
            "eres",
            "es",
            "somos",
            "son",
            "era",
            "fue",
            "fui",
            "estoy",
            "estás",
            "está",
            "estamos",
            "están",
            "estaba",
            "he",
            "has",
            "ha",
            "hemos",
            "han",
            "había",
            "tengo",
            "tienes",
            "tiene",
            "tenemos",
            "tienen",
            "voy",
            "vamos",
            "van",
            # conjunctions, prepositions, adverbs
            "y",
            "e",
            "o",
            "u",
            "pero",
            "porque",
            "que",
            "como",
            "cuando",
            "si",
            "no",
            "ni",
            "de",
            "del",
            "a",
            "al",
            "en",
            "con",
            "por",
            "para",
            "sin",
            "sobre",
            "entre",
            "desde",
            "hasta",
            "muy",
            "más",
            "menos",
            "también",
            "siempre",
            "ahora",
            "aquí",
            "allí",
            "entonces",
            "bueno",
            "bien",
            "claro",
            "pues",
            "vale",
            "solo",
            "sólo",
            "todo",
            "todos",
            "nada",
            "algo",
            "ya",
            "aún",
            "todavía",
            # feeling words a guest opens with
            "encantado",
            "encantada",
            "contento",
            "contenta",
            "feliz",
            "gracias",
            "perdón",
            "lamento",
            "curioso",
            "preocupado",
            "preocupada",
            "hablando",
            "pensando",
            "trabajando",
            "viniendo",
            "diciendo",
        }
    ),
    "it": frozenset(
        {
            "il",
            "lo",
            "la",
            "i",
            "gli",
            "le",
            "un",
            "uno",
            "una",
            "questo",
            "questa",
            "questi",
            "queste",
            "quello",
            "quella",
            "mio",
            "mia",
            "tuo",
            "tua",
            "suo",
            "sua",
            "nostro",
            "nostra",
            "io",
            "tu",
            "lui",
            "lei",
            "noi",
            "voi",
            "loro",
            "mi",
            "ti",
            "si",
            "ci",
            "vi",
            "ne",
            "sono",
            "sei",
            "è",
            "siamo",
            "siete",
            "era",
            "fu",
            "sto",
            "stai",
            "sta",
            "stiamo",
            "stanno",
            "ho",
            "hai",
            "ha",
            "abbiamo",
            "hanno",
            "aveva",
            "vado",
            "andiamo",
            "e",
            "ed",
            "o",
            "od",
            "ma",
            "perché",
            "che",
            "come",
            "quando",
            "se",
            "non",
            "né",
            "di",
            "del",
            "della",
            "a",
            "al",
            "alla",
            "in",
            "nel",
            "nella",
            "con",
            "per",
            "senza",
            "su",
            "tra",
            "fra",
            "da",
            "dal",
            "molto",
            "più",
            "meno",
            "anche",
            "sempre",
            "ora",
            "adesso",
            "qui",
            "qua",
            "lì",
            "allora",
            "bene",
            "certo",
            "solo",
            "tutto",
            "tutti",
            "niente",
            "qualcosa",
            "già",
            "ancora",
            "felice",
            "contento",
            "contenta",
            "grazie",
            "scusa",
            "scusi",
            "spiacente",
            "curioso",
            "preoccupato",
            "preoccupata",
            "parlando",
            "pensando",
            "lavorando",
            "venendo",
            "dicendo",
        }
    ),
    "fr": frozenset(
        {
            "le",
            "la",
            "les",
            "un",
            "une",
            "des",
            "ce",
            "cet",
            "cette",
            "ces",
            "mon",
            "ma",
            "mes",
            "ton",
            "ta",
            "tes",
            "son",
            "sa",
            "ses",
            "notre",
            "nos",
            "votre",
            "vos",
            "je",
            "tu",
            "il",
            "elle",
            "on",
            "nous",
            "vous",
            "ils",
            "elles",
            "me",
            "te",
            "se",
            "lui",
            "leur",
            "en",
            "y",
            "suis",
            "es",
            "est",
            "sommes",
            "êtes",
            "sont",
            "était",
            "fut",
            "ai",
            "as",
            "a",
            "avons",
            "avez",
            "ont",
            "avait",
            "vais",
            "allons",
            "vont",
            "et",
            "ou",
            "mais",
            "parce",
            "que",
            "comme",
            "quand",
            "si",
            "ne",
            "pas",
            "ni",
            "de",
            "du",
            "des",
            "à",
            "au",
            "aux",
            "dans",
            "avec",
            "par",
            "pour",
            "sans",
            "sur",
            "entre",
            "depuis",
            "très",
            "plus",
            "moins",
            "aussi",
            "toujours",
            "maintenant",
            "ici",
            "là",
            "alors",
            "bien",
            "bon",
            "sûr",
            "donc",
            "seulement",
            "tout",
            "tous",
            "rien",
            "quelque",
            "déjà",
            "encore",
            # elided forms that can stand as a token after tokenisation
            "j'",
            "l'",
            "d'",
            "n'",
            "qu'",
            "c'",
            "s'",
            "t'",
            "m'",
            "ravi",
            "ravie",
            "content",
            "contente",
            "heureux",
            "heureuse",
            "merci",
            "pardon",
            "désolé",
            "désolée",
            "curieux",
            "inquiet",
            "inquiète",
            "parlant",
            "pensant",
            "travaillant",
            "venant",
            "disant",
        }
    ),
    "de": frozenset(
        {
            "der",
            "die",
            "das",
            "den",
            "dem",
            "des",
            "ein",
            "eine",
            "einen",
            "einem",
            "einer",
            "eines",
            "dieser",
            "diese",
            "dieses",
            "mein",
            "meine",
            "dein",
            "deine",
            "sein",
            "seine",
            "ihr",
            "ihre",
            "unser",
            "unsere",
            "euer",
            "eure",
            "ich",
            "du",
            "er",
            "sie",
            "es",
            "wir",
            "ihr",
            "mich",
            "dich",
            "sich",
            "uns",
            "mir",
            "dir",
            "ihm",
            "ihnen",
            "bin",
            "bist",
            "ist",
            "sind",
            "seid",
            "war",
            "waren",
            "habe",
            "hast",
            "hat",
            "haben",
            "habt",
            "hatte",
            "hatten",
            "werde",
            "wird",
            "werden",
            "gehe",
            "gehen",
            "und",
            "oder",
            "aber",
            "weil",
            "dass",
            "wie",
            "wann",
            "wenn",
            "nicht",
            "kein",
            "keine",
            "von",
            "vom",
            "zu",
            "zur",
            "zum",
            "in",
            "im",
            "mit",
            "durch",
            "für",
            "ohne",
            "über",
            "zwischen",
            "seit",
            "bei",
            "auf",
            "aus",
            "an",
            "sehr",
            "mehr",
            "weniger",
            "auch",
            "immer",
            "jetzt",
            "hier",
            "da",
            "dort",
            "dann",
            "gut",
            "klar",
            "also",
            "nur",
            "alles",
            "alle",
            "nichts",
            "etwas",
            "schon",
            "noch",
            "froh",
            "glücklich",
            "danke",
            "entschuldigung",
            "leid",
            "neugierig",
            "besorgt",
            "sprechend",
            "denkend",
            "arbeitend",
            "kommend",
            "sagend",
        }
    ),
    "pt": frozenset(
        {
            "o",
            "a",
            "os",
            "as",
            "um",
            "uma",
            "uns",
            "umas",
            "este",
            "esta",
            "estes",
            "estas",
            "esse",
            "essa",
            "aquele",
            "aquela",
            "meu",
            "minha",
            "teu",
            "tua",
            "seu",
            "sua",
            "nosso",
            "nossa",
            "eu",
            "tu",
            "ele",
            "ela",
            "você",
            "nós",
            "vós",
            "eles",
            "elas",
            "vocês",
            "me",
            "te",
            "se",
            "nos",
            "lhe",
            "lhes",
            "sou",
            "és",
            "é",
            "somos",
            "são",
            "era",
            "foi",
            "fui",
            "estou",
            "estás",
            "está",
            "estamos",
            "estão",
            "estava",
            "tenho",
            "tens",
            "tem",
            "temos",
            "têm",
            "tinha",
            "vou",
            "vamos",
            "vão",
            "e",
            "ou",
            "mas",
            "porque",
            "que",
            "como",
            "quando",
            "se",
            "não",
            "nem",
            "de",
            "do",
            "da",
            "dos",
            "das",
            "a",
            "ao",
            "à",
            "em",
            "no",
            "na",
            "com",
            "por",
            "para",
            "sem",
            "sobre",
            "entre",
            "desde",
            "até",
            "muito",
            "mais",
            "menos",
            "também",
            "sempre",
            "agora",
            "aqui",
            "ali",
            "então",
            "bem",
            "bom",
            "claro",
            "pois",
            "só",
            "tudo",
            "todos",
            "nada",
            "algo",
            "já",
            "ainda",
            "encantado",
            "encantada",
            "contente",
            "feliz",
            "obrigado",
            "obrigada",
            "desculpa",
            "desculpe",
            "curioso",
            "preocupado",
            "preocupada",
            "falando",
            "pensando",
            "trabalhando",
            "vindo",
            "dizendo",
        }
    ),
}

#: Words that are adjectives of nationality, religion or politics. One of these alone is never a
#: person ("American", "Catholic", "Republican") even though it is capitalised and name-shaped.
#:
#: GENDERED AND PLURAL FORMS SPELLED OUT in the Romance rows, because "española" and "españolas"
#: are as common in a description as "español" and a row with only the masculine singular would
#: see one in three.
NOT_A_MONONYM: Dict[str, FrozenSet[str]] = {
    "en": frozenset(
        {
            "african",
            "american",
            "asian",
            "atheist",
            "australian",
            "brazilian",
            "british",
            "buddhist",
            "canadian",
            "catholic",
            "chinese",
            "christian",
            "conservative",
            "danish",
            "democrat",
            "democratic",
            "dutch",
            "english",
            "european",
            "french",
            "german",
            "hindu",
            "hispanic",
            "independent",
            "indian",
            "irish",
            "italian",
            "japanese",
            "jewish",
            "korean",
            "latina",
            "latino",
            "liberal",
            "mexican",
            "muslim",
            "norwegian",
            "portuguese",
            "progressive",
            "protestant",
            "republican",
            "russian",
            "scottish",
            "spanish",
            "swedish",
            "welsh",
        }
    ),
    "es": frozenset(
        {
            "africano",
            "africana",
            "americano",
            "americana",
            "asiático",
            "asiática",
            "ateo",
            "atea",
            "australiano",
            "australiana",
            "brasileño",
            "brasileña",
            "británico",
            "británica",
            "budista",
            "canadiense",
            "católico",
            "católica",
            "chino",
            "china",
            "cristiano",
            "cristiana",
            "conservador",
            "conservadora",
            "danés",
            "danesa",
            "demócrata",
            "holandés",
            "holandesa",
            "inglés",
            "inglesa",
            "europeo",
            "europea",
            "francés",
            "francesa",
            "alemán",
            "alemana",
            "hindú",
            "hispano",
            "hispana",
            "independiente",
            "indio",
            "india",
            "irlandés",
            "irlandesa",
            "italiano",
            "italiana",
            "japonés",
            "japonesa",
            "judío",
            "judía",
            "coreano",
            "coreana",
            "latino",
            "latina",
            "liberal",
            "mexicano",
            "mexicana",
            "musulmán",
            "musulmana",
            "noruego",
            "noruega",
            "portugués",
            "portuguesa",
            "progresista",
            "protestante",
            "republicano",
            "republicana",
            "ruso",
            "rusa",
            "escocés",
            "escocesa",
            "español",
            "española",
            "sueco",
            "sueca",
            "galés",
            "galesa",
        }
    ),
    "it": frozenset(
        {
            "africano",
            "africana",
            "americano",
            "americana",
            "asiatico",
            "asiatica",
            "ateo",
            "atea",
            "australiano",
            "australiana",
            "brasiliano",
            "brasiliana",
            "britannico",
            "britannica",
            "buddista",
            "canadese",
            "cattolico",
            "cattolica",
            "cinese",
            "cristiano",
            "cristiana",
            "conservatore",
            "conservatrice",
            "danese",
            "democratico",
            "democratica",
            "olandese",
            "inglese",
            "europeo",
            "europea",
            "francese",
            "tedesco",
            "tedesca",
            "indù",
            "ispanico",
            "ispanica",
            "indipendente",
            "indiano",
            "indiana",
            "irlandese",
            "italiano",
            "italiana",
            "giapponese",
            "ebreo",
            "ebrea",
            "coreano",
            "coreana",
            "latino",
            "latina",
            "liberale",
            "messicano",
            "messicana",
            "musulmano",
            "musulmana",
            "norvegese",
            "portoghese",
            "progressista",
            "protestante",
            "repubblicano",
            "repubblicana",
            "russo",
            "russa",
            "scozzese",
            "spagnolo",
            "spagnola",
            "svedese",
            "gallese",
        }
    ),
    "fr": frozenset(
        {
            "africain",
            "africaine",
            "américain",
            "américaine",
            "asiatique",
            "athée",
            "australien",
            "australienne",
            "brésilien",
            "brésilienne",
            "britannique",
            "bouddhiste",
            "canadien",
            "canadienne",
            "catholique",
            "chinois",
            "chinoise",
            "chrétien",
            "chrétienne",
            "conservateur",
            "conservatrice",
            "danois",
            "danoise",
            "démocrate",
            "néerlandais",
            "néerlandaise",
            "anglais",
            "anglaise",
            "européen",
            "européenne",
            "français",
            "française",
            "allemand",
            "allemande",
            "hindou",
            "hispanique",
            "indépendant",
            "indépendante",
            "indien",
            "indienne",
            "irlandais",
            "irlandaise",
            "italien",
            "italienne",
            "japonais",
            "japonaise",
            "juif",
            "juive",
            "coréen",
            "coréenne",
            "latino",
            "libéral",
            "libérale",
            "mexicain",
            "mexicaine",
            "musulman",
            "musulmane",
            "norvégien",
            "norvégienne",
            "portugais",
            "portugaise",
            "progressiste",
            "protestant",
            "protestante",
            "républicain",
            "républicaine",
            "russe",
            "écossais",
            "écossaise",
            "espagnol",
            "espagnole",
            "suédois",
            "suédoise",
            "gallois",
            "galloise",
        }
    ),
    "de": frozenset(
        {
            "afrikanisch",
            "afrikaner",
            "amerikanisch",
            "amerikaner",
            "amerikanerin",
            "asiatisch",
            "atheist",
            "atheistin",
            "australisch",
            "australier",
            "brasilianisch",
            "brasilianer",
            "britisch",
            "brite",
            "buddhist",
            "buddhistin",
            "kanadisch",
            "kanadier",
            "katholisch",
            "katholik",
            "chinesisch",
            "chinese",
            "christlich",
            "christ",
            "konservativ",
            "dänisch",
            "däne",
            "demokrat",
            "demokratisch",
            "niederländisch",
            "niederländer",
            "englisch",
            "engländer",
            "europäisch",
            "europäer",
            "französisch",
            "franzose",
            "deutsch",
            "deutscher",
            "hindu",
            "hispanisch",
            "unabhängig",
            "indisch",
            "inder",
            "irisch",
            "ire",
            "italienisch",
            "italiener",
            "japanisch",
            "japaner",
            "jüdisch",
            "jude",
            "koreanisch",
            "koreaner",
            "latino",
            "liberal",
            "mexikanisch",
            "mexikaner",
            "muslim",
            "muslimisch",
            "norwegisch",
            "norweger",
            "portugiesisch",
            "portugiese",
            "progressiv",
            "protestantisch",
            "protestant",
            "republikaner",
            "republikanisch",
            "russisch",
            "russe",
            "schottisch",
            "schotte",
            "spanisch",
            "spanier",
            "schwedisch",
            "schwede",
            "walisisch",
            "waliser",
        }
    ),
    "pt": frozenset(
        {
            "africano",
            "africana",
            "americano",
            "americana",
            "asiático",
            "asiática",
            "ateu",
            "ateia",
            "australiano",
            "australiana",
            "brasileiro",
            "brasileira",
            "britânico",
            "britânica",
            "budista",
            "canadense",
            "canadiano",
            "católico",
            "católica",
            "chinês",
            "chinesa",
            "cristão",
            "cristã",
            "conservador",
            "conservadora",
            "dinamarquês",
            "dinamarquesa",
            "democrata",
            "holandês",
            "holandesa",
            "inglês",
            "inglesa",
            "europeu",
            "europeia",
            "francês",
            "francesa",
            "alemão",
            "alemã",
            "hindu",
            "hispânico",
            "hispânica",
            "independente",
            "indiano",
            "indiana",
            "irlandês",
            "irlandesa",
            "italiano",
            "italiana",
            "japonês",
            "japonesa",
            "judeu",
            "judia",
            "coreano",
            "coreana",
            "latino",
            "latina",
            "liberal",
            "mexicano",
            "mexicana",
            "muçulmano",
            "muçulmana",
            "norueguês",
            "norueguesa",
            "português",
            "portuguesa",
            "progressista",
            "protestante",
            "republicano",
            "republicana",
            "russo",
            "russa",
            "escocês",
            "escocesa",
            "espanhol",
            "espanhola",
            "sueco",
            "sueca",
            "galês",
            "galesa",
        }
    ),
}


# =============================================================================================
# SHOW, ORGANISATION AND PLACE WORDS
# =============================================================================================

#: The word that ends a show's own name, stripped when matching a title ("... the Trail podcast").
#: "podcast" is a loanword in all five and stays; the second word is the one that moves.
SHOW_TAIL_WORDS: Dict[str, FrozenSet[str]] = {
    "en": frozenset({"podcast", "show"}),
    "es": frozenset({"podcast", "programa", "emisión"}),
    "it": frozenset({"podcast", "programma", "trasmissione"}),
    "fr": frozenset({"podcast", "émission", "programme"}),
    "de": frozenset({"podcast", "sendung", "folge"}),
    "pt": frozenset({"podcast", "programa", "emissão"}),
}

#: Words that end an ORGANISATION's name. A two-word "name" ending in one of these is an org, not
#: a person ("Timmerman Report", "Africa Tech Summit").
ORG_TAIL_TOKENS: Dict[str, FrozenSet[str]] = {
    "en": frozenset(
        {
            "committee",
            "research",
            "project",
            "exchange",
            "context",
            "ceo",
            "institute",
            "council",
            "society",
            "foundation",
            "initiative",
            "association",
            "commission",
            "book",
            "conquest",
            "experience",
            "media",
            "network",
            "online",
            "podcast",
            "report",
            "show",
            "studios",
            "summit",
            "tech",
        }
    ),
    "es": frozenset(
        {
            "comité",
            "investigación",
            "proyecto",
            "intercambio",
            "contexto",
            "ceo",
            "instituto",
            "consejo",
            "sociedad",
            "fundación",
            "iniciativa",
            "asociación",
            "comisión",
            "libro",
            "conquista",
            "experiencia",
            "medios",
            "red",
            "línea",
            "podcast",
            "informe",
            "programa",
            "estudios",
            "cumbre",
            "tecnología",
        }
    ),
    "it": frozenset(
        {
            "comitato",
            "ricerca",
            "progetto",
            "scambio",
            "contesto",
            "ceo",
            "istituto",
            "consiglio",
            "società",
            "fondazione",
            "iniziativa",
            "associazione",
            "commissione",
            "libro",
            "conquista",
            "esperienza",
            "media",
            "rete",
            "online",
            "podcast",
            "rapporto",
            "programma",
            "studi",
            "vertice",
            "tecnologia",
        }
    ),
    "fr": frozenset(
        {
            "comité",
            "recherche",
            "projet",
            "échange",
            "contexte",
            "ceo",
            "institut",
            "conseil",
            "société",
            "fondation",
            "initiative",
            "association",
            "commission",
            "livre",
            "conquête",
            "expérience",
            "médias",
            "réseau",
            "ligne",
            "podcast",
            "rapport",
            "émission",
            "studios",
            "sommet",
            "tech",
        }
    ),
    "de": frozenset(
        {
            "komitee",
            "ausschuss",
            "forschung",
            "projekt",
            "austausch",
            "kontext",
            "ceo",
            "institut",
            "rat",
            "gesellschaft",
            "stiftung",
            "initiative",
            "verband",
            "kommission",
            "buch",
            "eroberung",
            "erfahrung",
            "medien",
            "netzwerk",
            "online",
            "podcast",
            "bericht",
            "sendung",
            "studios",
            "gipfel",
            "technik",
        }
    ),
    "pt": frozenset(
        {
            "comité",
            "comitê",
            "investigação",
            "pesquisa",
            "projeto",
            "intercâmbio",
            "contexto",
            "ceo",
            "instituto",
            "conselho",
            "sociedade",
            "fundação",
            "iniciativa",
            "associação",
            "comissão",
            "livro",
            "conquista",
            "experiência",
            "média",
            "rede",
            "linha",
            "podcast",
            "relatório",
            "programa",
            "estúdios",
            "cúpula",
            "tecnologia",
        }
    ),
}

#: Region words. The English row's rule carries over and has to be RE-DECIDED per language rather
#: than translated: a country word that is also a surname is left out, and which ones those are
#: differs — English excludes Brazil and France (Sam Brazil, Anatole France), Portuguese must
#: exclude `Franca`, Spanish `Alemán`, German `Deutsch`.
PLACE_TAIL_TOKENS: Dict[str, FrozenSet[str]] = {
    "en": frozenset({"china", "india", "africa", "america", "americas", "asia", "europe"}),
    "es": frozenset({"china", "india", "áfrica", "américa", "américas", "asia", "europa"}),
    "it": frozenset({"cina", "india", "africa", "america", "americhe", "asia", "europa"}),
    "fr": frozenset({"chine", "inde", "afrique", "amérique", "amériques", "asie", "europe"}),
    "de": frozenset({"china", "indien", "afrika", "amerika", "asien", "europa"}),
    "pt": frozenset({"china", "índia", "áfrica", "américa", "américas", "ásia", "europa"}),
}

#: Counting words. A "name" that starts with one is not a person ("Two Carnegie Mellon").
#: Gendered numerals and the "both" word are spelled out per language.
NUMBER_WORDS: Dict[str, FrozenSet[str]] = {
    "en": frozenset(
        {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "both"}
    ),
    "es": frozenset(
        {
            "uno",
            "una",
            "dos",
            "tres",
            "cuatro",
            "cinco",
            "seis",
            "siete",
            "ocho",
            "nueve",
            "diez",
            "ambos",
            "ambas",
        }
    ),
    "it": frozenset(
        {
            "uno",
            "una",
            "due",
            "tre",
            "quattro",
            "cinque",
            "sei",
            "sette",
            "otto",
            "nove",
            "dieci",
            "entrambi",
            "entrambe",
        }
    ),
    "fr": frozenset(
        {
            "un",
            "une",
            "deux",
            "trois",
            "quatre",
            "cinq",
            "six",
            "sept",
            "huit",
            "neuf",
            "dix",
            "tous",
            "toutes",
        }
    ),
    "de": frozenset(
        {
            "eins",
            "ein",
            "eine",
            "zwei",
            "drei",
            "vier",
            "fünf",
            "sechs",
            "sieben",
            "acht",
            "neun",
            "zehn",
            "beide",
        }
    ),
    "pt": frozenset(
        {
            "um",
            "uma",
            "dois",
            "duas",
            "três",
            "quatro",
            "cinco",
            "seis",
            "sete",
            "oito",
            "nove",
            "dez",
            "ambos",
            "ambas",
        }
    ),
}


# =============================================================================================
# PROPER NOUNS — the English row PLUS each market's own, never a translation
# =============================================================================================

#: Product and company names used as a single word. SPELLED THE SAME EVERYWHERE, so these rows are
#: NOT translations — "Apple" is Apple in Spanish. Each non-English row is the English row plus the
#: brands that market actually mentions, which is the only thing that can differ.
BRAND_MONONYMS: Dict[str, FrozenSet[str]] = {
    "en": frozenset(
        {
            "alexa",
            "amazon",
            "apple",
            "chatgpt",
            "claude",
            "copilot",
            "gemini",
            "google",
            "meta",
            "microsoft",
            "nvidia",
            "openai",
            "siri",
        }
    ),
    "es": frozenset(
        {
            "alexa",
            "amazon",
            "apple",
            "chatgpt",
            "claude",
            "copilot",
            "gemini",
            "google",
            "meta",
            "microsoft",
            "nvidia",
            "openai",
            "siri",
            "movistar",
            "telefónica",
            "mercadona",
            "santander",
            "bbva",
            "iberdrola",
        }
    ),
    "it": frozenset(
        {
            "alexa",
            "amazon",
            "apple",
            "chatgpt",
            "claude",
            "copilot",
            "gemini",
            "google",
            "meta",
            "microsoft",
            "nvidia",
            "openai",
            "siri",
            "tim",
            "enel",
            "eni",
            "intesa",
            "unicredit",
            "fiat",
        }
    ),
    "fr": frozenset(
        {
            "alexa",
            "amazon",
            "apple",
            "chatgpt",
            "claude",
            "copilot",
            "gemini",
            "google",
            "meta",
            "microsoft",
            "nvidia",
            "openai",
            "siri",
            "orange",
            "sncf",
            "edf",
            "renault",
            "carrefour",
            "doctolib",
        }
    ),
    "de": frozenset(
        {
            "alexa",
            "amazon",
            "apple",
            "chatgpt",
            "claude",
            "copilot",
            "gemini",
            "google",
            "meta",
            "microsoft",
            "nvidia",
            "openai",
            "siri",
            "sap",
            "siemens",
            "telekom",
            "bosch",
            "lufthansa",
            "zalando",
        }
    ),
    "pt": frozenset(
        {
            "alexa",
            "amazon",
            "apple",
            "chatgpt",
            "claude",
            "copilot",
            "gemini",
            "google",
            "meta",
            "microsoft",
            "nvidia",
            "openai",
            "siri",
            "nos",
            "meo",
            "galp",
            "edp",
            "sonae",
            "nubank",
        }
    ),
}

#: Podcast networks and publishers, used to recognise an ORG author tag rather than a person.
#: Proper nouns again — the English row is kept in every language because an English-language
#: network is named in a Spanish show's credits just as often, and each row ADDS its own market's
#: outlets. Not a translation.
KNOWN_NETWORKS: Dict[str, FrozenSet[str]] = {
    "en": frozenset(
        {
            "a16z",
            "acast",
            "andreessen horowitz",
            "associated press",
            "audible",
            "barstool",
            "bloomberg",
            "cadence13",
            "crooked media",
            "earwolf",
            "financial times",
            "gimlet",
            "headgum",
            "iheart",
            "iheartradio",
            "kaleidoscope",
            "maximum fun",
            "maximumfun",
            "megaphone",
            "new york times",
            "npr",
            "patreon",
            "pushkin",
            "pushkin industries",
            "radiotopia",
            "reuters",
            "ringer",
            "spotify",
            "stitcher",
            "substack",
            "the atlantic",
            "the economist",
            "the guardian",
            "the new york times",
            "the ringer",
            "the wall street journal",
            "the washington post",
            "vox",
            "washington post",
            "wondery",
        }
    ),
    "es": frozenset(
        {
            "rtve",
            "radio nacional",
            "cadena ser",
            "onda cero",
            "cope",
            "prisa",
            "prisa audio",
            "el país",
            "el mundo",
            "la vanguardia",
            "podimo",
            "spotify",
            "podium podcast",
            "cuonda",
            "bluper",
        }
    ),
    "it": frozenset(
        {
            "rai",
            "rai radio",
            "radio24",
            "chora media",
            "il sole 24 ore",
            "corriere della sera",
            "la repubblica",
            "il post",
            "sky tg24",
            "mediaset",
            "spotify",
            "storielibere",
            "piano p",
        }
    ),
    "fr": frozenset(
        {
            "radio france",
            "france inter",
            "france culture",
            "france info",
            "rtl",
            "europe 1",
            "rfi",
            "binge audio",
            "louie media",
            "nouvelles écoutes",
            "le monde",
            "les échos",
            "arte radio",
            "spotify",
            "slate.fr",
        }
    ),
    "de": frozenset(
        {
            "ard",
            "zdf",
            "deutschlandfunk",
            "deutsche welle",
            "br",
            "wdr",
            "ndr",
            "swr",
            "rbb",
            "funk",
            "der spiegel",
            "die zeit",
            "süddeutsche zeitung",
            "faz",
            "spotify",
            "studio bummens",
            "viertausendhertz",
        }
    ),
    "pt": frozenset(
        {
            "rtp",
            "antena 1",
            "antena 3",
            "renascença",
            "tsf",
            "observador",
            "público",
            "expresso",
            "jornal de notícias",
            "sic",
            "tvi",
            "spotify",
            "fumaça",
            "bumerangue",
        }
    ),
}


# =============================================================================================
# PHRASE TEMPLATES
#
# These are the RECALL side, and they are templates rather than compiled patterns on purpose.
# Each one wraps the same name-shape capture, and that shape — `_STATED_NAMES`, built from the
# Unicode-aware `_STATED_UC` — is STRUCTURE and stays in `hosts.py`. Only the prose around it
# belongs here. `hosts.py` interpolates with `%`-substitution, not `str.format`, because these
# patterns contain regex quantifiers like `{0,6}` that `format` would read as fields.
#
# RECALL IS THE RISKY HALF, so this is the part to be most sceptical of. A too-loose host phrase
# is how a person an episode is merely ABOUT becomes a named voice (#876), which is why the
# English rows are gold-gated against #2269's development set and the five others are not gated
# against anything. What protects them is the PRECISION side above — `NOT_A_NAME_TOKEN`,
# `NOT_A_MONONYM`, `ORG_TAIL_TOKENS`, `ROLE_OR_FILLER_TOKENS` — which now has its own rows, so a
# loose match in Spanish meets a Spanish reject filter rather than an English one.
# =============================================================================================

#: "hosted by X", "presented by X", "with X" — the show's own statement of who presents it.
#: Placeholders: ``%(lead)s`` bounded filler, ``%(names)s`` the name-shape capture.
HOST_PHRASE_TEMPLATES: Dict[str, Tuple[str, ...]] = {
    "en": (
        r"\bhosted\s+by\s+%(lead)s(?P<names>%(names)s)",
        r"\b(?:run|presented)\s+by\s+(?P<names>%(names)s)",
        r"\b(?:co-?)?hosts?\s+(?P<names>%(names)s)",
        r"\bjoin\s+%(lead)s(?P<names>%(names)s)",
        r"\bjournalists?\s+(?P<names>%(names)s)",
        r"\bwith\s+(?P<names>%(names)s)(?:,\s*(?:PhD|Ph\.D\.?|MD|M\.D\.?))?"
        r"(?:\s*\([^)]*\))?\s*$",
    ),
    "es": (
        r"\bpresentado\s+por\s+%(lead)s(?P<names>%(names)s)",
        r"\b(?:conducido|dirigido|realizado)\s+por\s+(?P<names>%(names)s)",
        r"\b(?:co)?(?:anfitri(?:ó|o)n(?:es)?|anfitriona|presentador(?:es|a|as)?)"
        r"\s+(?P<names>%(names)s)",
        r"\b(?:acompa(?:ñ|n)an?|(?:ú|u)nete\s+a)\s+%(lead)s(?P<names>%(names)s)",
        r"\bperiodistas?\s+(?P<names>%(names)s)",
        r"\bcon\s+(?P<names>%(names)s)(?:,\s*(?:PhD|Ph\.D\.?|MD|M\.D\.?|Dr\.?a?))?"
        r"(?:\s*\([^)]*\))?\s*$",
    ),
    "it": (
        r"\bcondotto\s+da\s+%(lead)s(?P<names>%(names)s)",
        r"\b(?:presentato|realizzato|diretto)\s+da\s+(?P<names>%(names)s)",
        r"\b(?:co)?(?:conduttor(?:e|i|ice|ici)|presentator(?:e|i|ice|ici))"
        r"\s+(?P<names>%(names)s)",
        r"\b(?:unisciti\s+a|accompagna(?:no)?)\s+%(lead)s(?P<names>%(names)s)",
        r"\bgiornalist[ai]\s+(?P<names>%(names)s)",
        r"\bcon\s+(?P<names>%(names)s)(?:,\s*(?:PhD|Ph\.D\.?|MD|M\.D\.?|Dott\.?))?"
        r"(?:\s*\([^)]*\))?\s*$",
    ),
    "fr": (
        r"\b(?:anim(?:é|e)|pr(?:é|e)sent(?:é|e))\s+par\s+%(lead)s(?P<names>%(names)s)",
        r"\b(?:r(?:é|e)alis(?:é|e)|dirig(?:é|e))\s+par\s+(?P<names>%(names)s)",
        r"\b(?:co)?(?:animateur(?:s)?|animatrice(?:s)?|pr(?:é|e)sentateur(?:s)?"
        r"|pr(?:é|e)sentatrice(?:s)?)\s+(?P<names>%(names)s)",
        r"\b(?:rejoignez|accompagn(?:e|ent))\s+%(lead)s(?P<names>%(names)s)",
        r"\bjournalistes?\s+(?P<names>%(names)s)",
        r"\bavec\s+(?P<names>%(names)s)(?:,\s*(?:PhD|Ph\.D\.?|MD|M\.D\.?|Dr\.?))?"
        r"(?:\s*\([^)]*\))?\s*$",
    ),
    "de": (
        r"\b(?:moderiert|pr(?:ä|a)sentiert)\s+von\s+%(lead)s(?P<names>%(names)s)",
        r"\b(?:produziert|geleitet)\s+von\s+(?P<names>%(names)s)",
        r"\b(?:ko)?(?:gastgeber(?:in|innen)?|moderator(?:in|en|innen)?)" r"\s+(?P<names>%(names)s)",
        r"\b(?:begleite(?:t|n)?|komm(?:t|en)\s+dazu)\s+%(lead)s(?P<names>%(names)s)",
        r"\bjournalist(?:in|en|innen)?\s+(?P<names>%(names)s)",
        r"\bmit\s+(?P<names>%(names)s)(?:,\s*(?:PhD|Ph\.D\.?|MD|M\.D\.?|Dr\.?))?"
        r"(?:\s*\([^)]*\))?\s*$",
    ),
    "pt": (
        r"\bapresentado\s+p(?:or|el[oa])\s+%(lead)s(?P<names>%(names)s)",
        r"\b(?:conduzido|dirigido|realizado)\s+p(?:or|el[oa])\s+(?P<names>%(names)s)",
        r"\b(?:co)?(?:anfitri(?:ã|a)o(?:s)?|anfitri(?:ã|a)(?:s)?|apresentador(?:es|a|as)?)"
        r"\s+(?P<names>%(names)s)",
        r"\b(?:junte-se\s+a|acompanha(?:m)?)\s+%(lead)s(?P<names>%(names)s)",
        r"\bjornalistas?\s+(?P<names>%(names)s)",
        r"\bcom\s+(?P<names>%(names)s)(?:,\s*(?:PhD|Ph\.D\.?|MD|M\.D\.?|Dr\.?a?))?"
        r"(?:\s*\([^)]*\))?\s*$",
    ),
}

#: The verb for "these people make the show". English folds singular and plural into one `s?`;
#: the Romance rows spell BOTH conjugations because the third-person plural is not the singular
#: plus an s (`presenta`/`presentano`).
#:
#: "HELP" IS IN THE ROMANCE ROWS ONLY ("Eliezer Budasoff y Silvia Viñas te ayudan a entender",
#: El Hilo). Measured over 585 chart feeds' channel descriptions (2026-10-10): es/pt/it/fr gain
#: one host statement, El Hilo's, and no wrong one; German `hilft`/`helfen` produced two false
#: hosts ("Lustige Experimente helfen"), because German capitalises the nouns before it.
PRESENTS: Dict[str, str] = {
    #: The verb a description uses for "these people make the show". English folds singular and
    #: plural into one `s?`; the Romance rows need BOTH conjugations spelled out because the
    #: third-person plural is not the singular plus an s (`presenta`/`presentano`), and German
    #: puts the verb in a different place entirely but the token still has to be matchable.
    "en": (
        r"(?:explore|explain|discuss|talk|cover|host|present|bring|engage|interview|uncover"
        r"|tackle)s?\b"
    ),
    "es": (
        r"(?:explora|exploran|explica|explican|discute|discuten|habla|hablan|cubre|cubren"
        r"|presenta|presentan|trae|traen|entrevista|entrevistan|aborda|abordan|conduce"
        r"|conducen|ayuda|ayudan)\b"
    ),
    "it": (
        r"(?:esplora|esplorano|spiega|spiegano|discute|discutono|parla|parlano|copre|coprono"
        r"|presenta|presentano|porta|portano|intervista|intervistano|affronta|affrontano"
        r"|conduce|conducono|aiuta|aiutano)\b"
    ),
    "fr": (
        r"(?:explore|explorent|explique|expliquent|discute|discutent|parle|parlent|couvre"
        r"|couvrent|présente|présentent|apporte|apportent|interviewe|interviewent|aborde"
        r"|abordent|animent|anime|aide|aident)\b"
    ),
    "de": (
        r"(?:erkund(?:et|en)|erklär(?:t|en)|diskutier(?:t|en)|sprech(?:t|en)|spricht"
        r"|behandel(?:t|n)|präsentier(?:t|en)|bring(?:t|en)|interview(?:t|en)|moderier(?:t|en))"
        r"\b"
    ),
    "pt": (
        r"(?:explora|exploram|explica|explicam|discute|discutem|fala|falam|cobre|cobrem"
        r"|apresenta|apresentam|traz|trazem|entrevista|entrevistam|aborda|abordam|conduz"
        r"|conduzem|ajuda|ajudam)\b"
    ),
}

#: A presenting formula that names THE SHOW: "This is X", "You're listening to X",
#: "welcome back to X", "today on X". Four shapes, kept across all six.
SHOW_INTRO_CUE: Dict[str, str] = {
    #: A presenting formula that names THE SHOW. The English row is the measured one; the five
    #: translations keep the same four shapes — "this is X", "you are listening to X", "welcome
    #: (back) to X", "today/this week on X" — because those are the shapes a presenter uses in
    #: any of them, and a greeting that opens the episode counts too.
    "en": (
        r"(?:this is|you'?re listening to|you are listening to|welcome(?: back)?"
        r" to(?: (?:another|this|today'?s) (?:episode|edition) of)?|(?:today|tonight|this week"
        r"|this time|this season|this month|next (?:few )?\w+) on|here on|(?:hello|hi)"
        r"(?:,?\s+(?:everyone|everybody|there|folks|all))?,?)"
    ),
    "es": (
        r"(?:(?:esto|este) es|est(?:á|a)s escuchando|est(?:á|a)is escuchando"
        r"|bienvenid[oa]s?(?: de nuevo)? a(?: (?:otro|este|el) (?:episodio|programa) de)?"
        r"|(?:hoy|esta noche|esta semana|esta temporada|este mes) en|aqu(?:í|i) en"
        r"|hola(?:,?\s+(?:a todos|a todas|gente|amigos))?,?)"
    ),
    "it": (
        r"(?:quest[oa] (?:è|e'|e)|stai ascoltando|state ascoltando"
        r"|ben(?:venut|tornat)[oiae]+(?: di nuovo)? (?:a|su|nel|nella)"
        r"(?: (?:un altro|questo) (?:episodio|programma) di)?"
        r"|(?:oggi|stasera|questa settimana|questa stagione|questo mese) (?:a|su|in)"
        r"|qui (?:a|su)|ciao(?:,?\s+(?:a tutti|a tutte|ragazzi))?,?)"
    ),
    "fr": (
        r"(?:(?:c'?est|voici)|vous (?:é|e)coutez|tu (?:é|e)coutes"
        r"|bienvenue(?: (?:à|a) nouveau)? (?:à|a|dans|sur)"
        r"(?: (?:un autre|ce|cet) (?:(?:é|e)pisode|num(?:é|e)ro) de)?"
        r"|(?:aujourd'?hui|ce soir|cette semaine|cette saison|ce mois) (?:dans|sur)"
        r"|ici (?:à|a|sur)|(?:bonjour|salut)(?:,?\s+(?:(?:à|a) tous|(?:à|a) toutes"
        r"|tout le monde))?,?)"
    ),
    "de": (
        r"(?:(?:dies|das) ist|(?:du h(?:ö|o)rst|sie h(?:ö|o)ren|ihr h(?:ö|o)rt)"
        r"|willkommen(?: zur(?:ü|u)ck)? (?:bei|zu|in)"
        r"(?: (?:einer weiteren|dieser|der) (?:folge|episode|ausgabe) von)?"
        r"|(?:heute|heute abend|diese woche|diese staffel|diesen monat) (?:bei|auf|in)"
        r"|hier (?:bei|auf)|(?:hallo|hi)(?:,?\s+(?:zusammen|alle|leute))?,?)"
    ),
    "pt": (
        r"(?:(?:isto|este) (?:é|e)|est(?:á|a)s a ouvir|est(?:ã|a)o a ouvir|voc(?:ê|e) est(?:á|a)"
        r" ouvindo|bem-vind[oa]s?(?: de novo)? a(?:o)?"
        r"(?: (?:outro|este|o) (?:epis(?:ó|o)dio|programa) de)?"
        r"|(?:hoje|esta noite|esta semana|esta temporada|este m(?:ê|e)s) (?:n[oa]|em)"
        r"|aqui (?:n[oa]|em)|(?:ol(?:á|a)|oi)(?:,?\s+(?:a todos|a todas|pessoal))?,?)"
    ),
}

#: A guest-introduction cue where the CUE comes first ("my guest today is X").
CUE_FIRST_BODY: Dict[str, str] = {
    #: The ENGLISH row is hosts.py's measured text, unchanged. The last two alternatives — the
    #: progressive ("I'm speaking with") and the inverted ("with us today is") — are easy to drop
    #: when hand-copying and each one is a whole introduction shape, so every other row carries
    #: both as well rather than stopping at the first five.
    "en": (
        r"(?:my|our)\s+guests?\s+(?:today\s+)?(?:is|are)|joined\s+(?:today\s+)?by"
        r"|joining\s+(?:me|us)(?:\s+(?:today|now|this\s+week))?\s+(?:is|are)"
        r"|(?:i'?m|we'?re)\s+(?:here\s+)?(?:joined\s+)?with|(?:please\s+)?welcome\s+(?:back\s+)?"
        r"|here\s+with\s+me\s+(?:is|are)|(?:my|our)\s+colleague"
        r"|(?:i'?m|we'?re)\s+(?:here\s+)?(?:chatting|talking|speaking|sitting\s+down)\s+(?:with|to)"
        r"|with\s+us\s+(?:today\s+)?(?:is|are)"
    ),
    "es": (
        r"(?:mi|nuestr[oa])s?\s+invitad[oa]s?\s+(?:de\s+hoy\s+)?(?:es|son)"
        r"|(?:me|nos)\s+acompa(?:ñ|n)an?(?:\s+(?:hoy|ahora|esta\s+semana))?"
        r"|(?:recibimos|damos\s+la\s+bienvenida)\s+a|conmigo\s+(?:est(?:á|a)|hoy)"
        r"|(?:mi|nuestr[oa])\s+coleg[a]"
        r"|(?:estoy|estamos)\s+(?:charlando|hablando|conversando|sentad[oa]s?)\s+con"
        r"|con\s+nosotros\s+(?:hoy\s+)?(?:est(?:á|a)|est(?:á|a)n)"
    ),
    "it": (
        r"(?:il\s+mio|il\s+nostro|i\s+nostri)\s+ospit[ei]\s+(?:di\s+oggi\s+)?(?:(?:è|e'|e)|sono)"
        r"|(?:mi|ci)\s+accompagna(?:no)?(?:\s+(?:oggi|ora|questa\s+settimana))?"
        r"|diamo\s+il\s+benvenuto\s+a|con\s+me\s+(?:c'(?:è|e)|oggi)"
        r"|(?:il\s+mio|il\s+nostro)\s+colleg[ah]"
        r"|(?:sto|stiamo)\s+(?:chiacchierando|parlando|conversando)\s+con"
        r"|con\s+noi\s+(?:oggi\s+)?(?:c'(?:è|e)|ci\s+sono)"
    ),
    "fr": (
        r"(?:mon|notre|mes|nos)\s+invit(?:é|e)e?s?\s+(?:d'?aujourd'?hui\s+)?(?:est|sont)"
        r"|(?:m'?|nous\s+)?accompagn(?:e|ent)(?:\s+(?:aujourd'?hui|maintenant|cette\s+semaine))?"
        r"|(?:nous\s+)?(?:recevons|accueillons)"
        r"|avec\s+moi\s+(?:aujourd'?hui\s+)?(?:est|se\s+trouve)"
        r"|(?:mon|notre)\s+coll(?:è|e)gue"
        r"|(?:je\s+(?:discute|parle)|nous\s+(?:discutons|parlons))\s+avec"
        r"|avec\s+nous\s+(?:aujourd'?hui\s+)?(?:est|sont|se\s+trouve)"
    ),
    "de": (
        r"(?:mein|unser)e?\s+g(?:ä|a)st(?:e|in)?\s+(?:heute\s+)?(?:ist|sind)"
        r"|(?:begleite(?:t|n)|ist\s+dabei)(?:\s+(?:heute|jetzt|diese\s+woche))?"
        r"|(?:wir\s+)?begr(?:ü|u)(?:ß|ss)en|bei\s+mir\s+(?:ist|sitzt)"
        r"|(?:mein|unser)e?\s+kolleg(?:e|in)"
        r"|(?:ich\s+(?:unterhalte\s+mich|spreche)|wir\s+(?:unterhalten\s+uns|sprechen))\s+mit"
        r"|bei\s+uns\s+(?:heute\s+)?(?:ist|sind)"
    ),
    "pt": (
        r"(?:o\s+meu|a\s+minha|o\s+nosso|a\s+nossa)s?\s+convidad[oa]s?"
        r"\s+(?:de\s+hoje\s+)?(?:é|e|s(?:ã|a)o)"
        r"|(?:me|nos)\s+acompanha(?:m)?(?:\s+(?:hoje|agora|esta\s+semana))?"
        r"|(?:recebemos|damos\s+as\s+boas-vindas)\s+a|comigo\s+est(?:á|a)"
        r"|(?:o\s+meu|a\s+minha)\s+colega"
        r"|(?:estou|estamos)\s+(?:a\s+)?(?:conversando|falando|conversar|falar)\s+com"
        r"|com\s+(?:n(?:ó|o)s|nosco)\s+(?:hoje\s+)?(?:est(?:á|a)|est(?:ã|a)o)"
    ),
}

#: The same cue in the PAST tense ("I spoke with X").
CUE_FIRST_PAST_BODY: Dict[str, str] = {
    "en": r"(?:i|we)\s+(?:spoke|talked|sat\s+down)\s+with",
    "es": r"(?:habl(?:é|e)|hablamos|convers(?:é|e)|conversamos|me\s+sent(?:é|e))\s+con",
    "it": r"(?:ho|abbiamo)\s+(?:parlato|chiacchierato|conversato)\s+con",
    "fr": r"(?:j'?ai|nous\s+avons)\s+(?:parl(?:é|e)|discut(?:é|e)|(?:é|e)chang(?:é|e))\s+avec",
    "de": (
        r"(?:ich\s+habe|wir\s+haben)\s+(?:gesprochen|geredet|unterhalten)(?:\s+mit)?"
        r"|sprach\s+mit"
    ),
    "pt": r"(?:falei|falamos|conversei|conversámos|conversamos)\s+com",
}

#: A greeting that follows a name ("Alice, welcome", "Bob, thanks for coming").
GREETED_TAIL: Dict[str, str] = {
    "en": (
        r"welcome\b|thanks?(?:\s+so\s+much)?\s+for\s+(?:coming|joining|being)"
        r"|thank\s+you(?:\s+so\s+much)?\s+for\s+(?:coming|joining|being)"
    ),
    "es": (
        r"bienvenid[oa]s?\b|gracias(?:\s+(?:mil|muchas))?\s+por\s+(?:venir|acompa(?:ñ|n)arnos"
        r"|estar)"
    ),
    "it": r"ben(?:venut|tornat)[oiae]+\b|grazie(?:\s+mille)?\s+per\s+(?:essere|esserci|avermi)",
    "fr": (r"bienvenue\b|merci(?:\s+beaucoup)?\s+(?:d'?(?:avoir|(?:ê|e)tre)|de\s+(?:venir|nous))"),
    "de": r"willkommen\b|danke(?:\s+sehr)?(?:,)?\s+(?:dass|f(?:ü|u)r)\b",
    "pt": r"bem-vind[oa]s?\b|obrigad[oa](?:\s+(?:muito|demais))?\s+por\s+(?:vir|estar|nos)",
}

#: A cue where the NAME comes first ("X is here with me", "X joins us").
NAME_FIRST_TAIL: Dict[str, str] = {
    "en": (
        r"(?:is|are)\s+(?:here\s+)?with\s+(?:me|us)|(?:is|are)\s+(?:my|our)\s+guests?"
        r"|(?:is|are)\s+joining\s+(?:me|us)|joins?\s+(?:me|us)|(?:is|are)\s+here\s+to\b"
    ),
    "es": (
        r"(?:est(?:á|a)|est(?:á|a)n)\s+(?:aqu(?:í|i)\s+)?con(?:migo|\s+nosotros)"
        r"|(?:es|son)\s+(?:mi|nuestr[oa])s?\s+invitad[oa]s?"
        r"|(?:me|nos)\s+acompa(?:ñ|n)an?|(?:est(?:á|a)|est(?:á|a)n)\s+aqu(?:í|i)\s+para\b"
    ),
    "it": (
        r"(?:(?:è|e'|e)|sono)\s+(?:qui\s+)?con\s+(?:me|noi)"
        r"|(?:(?:è|e'|e)|sono)\s+(?:il\s+mio|il\s+nostro|i\s+nostri)\s+ospit[ei]"
        r"|(?:mi|ci)\s+accompagna(?:no)?|(?:(?:è|e'|e)|sono)\s+qui\s+per\b"
    ),
    "fr": (
        r"(?:est|sont)\s+(?:ici\s+)?avec\s+(?:moi|nous)"
        r"|(?:est|sont)\s+(?:mon|notre|mes|nos)\s+invit(?:é|e)e?s?"
        r"|(?:m'?|nous\s+)?rejoin(?:t|gnent)|(?:est|sont)\s+(?:ici\s+|l(?:à|a)\s+)?pour\b"
    ),
    "de": (
        r"(?:ist|sind)\s+(?:hier\s+)?bei\s+(?:mir|uns)"
        r"|(?:ist|sind)\s+(?:mein|unser)e?\s+g(?:ä|a)st(?:e|in)?"
        r"|(?:begleite(?:t|n)|kommt\s+dazu)|(?:ist|sind)\s+hier,?\s+um\b"
    ),
    "pt": (
        r"(?:est(?:á|a)|est(?:ã|a)o)\s+(?:aqui\s+)?com(?:igo|\s+n(?:ó|o)s)"
        r"|(?:é|e|s(?:ã|a)o)\s+(?:o\s+meu|a\s+minha|o\s+nosso|a\s+nossa)s?\s+convidad[oa]s?"
        r"|(?:me|nos)\s+acompanha(?:m)?|(?:est(?:á|a)|est(?:ã|a)o)\s+aqui\s+para\b"
    ),
}

#: A REPORTING verb after a name ("X explains", "X walks us through"). Weaker than a guest cue —
#: it marks someone the episode quotes rather than someone in the room.
NAME_FIRST_REPORT_TAIL: Dict[str, str] = {
    "en": (
        r"explains?|reports?|tells\s+us|walks\s+us\s+through|talks\s+us\s+through"
        r"|takes\s+us\s+(?:through|inside)|breaks\s+(?:it\s+|this\s+)?down"
    ),
    "es": (
        r"explica(?:n)?|informa(?:n)?|nos\s+cuenta(?:n)?|nos\s+gu(?:í|i)a(?:n)?\s+por"
        r"|nos\s+lleva(?:n)?\s+(?:por|dentro)|desglosa(?:n)?|analiza(?:n)?"
    ),
    "it": (
        r"spiega(?:no)?|riferisce|riferiscono|ci\s+racconta(?:no)?|ci\s+guida(?:no)?\s+attraverso"
        r"|ci\s+porta(?:no)?\s+(?:dentro|attraverso)|scompone|analizza(?:no)?"
    ),
    "fr": (
        r"explique(?:nt)?|rapporte(?:nt)?|nous\s+raconte(?:nt)?"
        r"|nous\s+guide(?:nt)?\s+(?:à|a)\s+travers"
        r"|nous\s+(?:emm(?:è|e)ne|emm(?:è|e)nent)\s+(?:dans|(?:à|a) l'int(?:é|e)rieur)"
        r"|d(?:é|e)crypte(?:nt)?|analyse(?:nt)?"
    ),
    "de": (
        r"erkl(?:ä|a)r(?:t|en)|berichte(?:t|n)|erz(?:ä|a)hl(?:t|en)\s+uns"
        r"|f(?:ü|u)hr(?:t|en)\s+uns\s+durch|nimm(?:t|en)\s+uns\s+mit"
        r"|schl(?:ü|u)ssel(?:t|n)\s+auf|analysier(?:t|en)"
    ),
    "pt": (
        r"explica(?:m)?|relata(?:m)?|conta(?:m)?-nos|nos\s+conta(?:m)?"
        r"|guia(?:m)?-nos\s+por|leva(?:m)?-nos\s+(?:por|dentro)|decomp(?:õ|o)e|analisa(?:m)?"
    ),
}

#: A HYPOTHETICAL frame. A name inside one is an example, not a participant ("let's say Alice...").
HYPOTHETICAL_LEAD: Dict[str, str] = {
    "en": (
        r"(?:\blet'?s\s+say|\blet\s+us\s+say|\bsuppose|\bsupposing|\bimagine|\bpretend|\bif)"
        r"(?:\s+that)?,?\s*$"
    ),
    "es": (
        r"(?:\bdigamos|\bsupongamos|\bsup(?:ó|o)n|\bimagina|\bimaginemos|\bfinge|\bsi)"
        r"(?:\s+que)?,?\s*$"
    ),
    "it": (
        r"(?:\bdiciamo|\bsupponiamo|\bsupponi|\bimmagina|\bimmaginiamo|\bfingi|\bse)"
        r"(?:\s+che)?,?\s*$"
    ),
    "fr": (
        r"(?:\bdisons|\bsupposons|\bsuppose|\bimagine|\bimaginons|\bfais\s+semblant|\bsi)"
        r"(?:\s+que)?,?\s*$"
    ),
    "de": (
        r"(?:\bsagen\s+wir|\bnehmen\s+wir\s+an|\bangenommen|\bstell(?:\s+dir)?\s+vor"
        r"|\bstellen\s+wir\s+uns\s+vor|\bwenn|\bfalls)(?:\s+dass)?,?\s*$"
    ),
    "pt": (
        r"(?:\bdigamos|\bsuponhamos|\bsup(?:õ|o)e|\bimagina|\bimaginemos|\bfinge|\bse)"
        r"(?:\s+que)?,?\s*$"
    ),
}

#: A REPORTED-SPEECH frame ("Alice said, 'hello...'"). The greeting belongs to the quote, not to
#: the conversation, so a name introduced this way is not in the room.
REPORTED_LEAD: Dict[str, str] = {
    "en": (
        r"(?P<who>\b\w+)(?:\s+\w+ly)?\s+(?:said|says|told\s+(?:me|us|him|her)|asked)\b,?"
        r"\s*[\"“]?\s*(?:(?:hello|hi|hey)\W{0,2})?\s*$"
    ),
    "es": (
        r"(?P<who>\b\w+)(?:\s+\w+mente)?\s+(?:dijo|dice|me\s+dijo|nos\s+dijo|pregunt(?:ó|o))\b,?"
        r"\s*[\"“]?\s*(?:(?:hola|buenas)\W{0,2})?\s*$"
    ),
    "it": (
        r"(?P<who>\b\w+)(?:\s+\w+mente)?\s+(?:ha\s+detto|dice|mi\s+ha\s+detto|ci\s+ha\s+detto"
        r"|ha\s+chiesto)\b,?\s*[\"“]?\s*(?:(?:ciao|salve)\W{0,2})?\s*$"
    ),
    "fr": (
        r"(?P<who>\b\w+)(?:\s+\w+ment)?\s+(?:a\s+dit|dit|m'?a\s+dit|nous\s+a\s+dit"
        r"|a\s+demand(?:é|e))\b,?\s*[\"“]?\s*(?:(?:bonjour|salut)\W{0,2})?\s*$"
    ),
    "de": (
        r"(?P<who>\b\w+)\s+(?:sagte|sagt|hat\s+(?:mir|uns)\s+gesagt|fragte)\b,?"
        r"\s*[\"“]?\s*(?:(?:hallo|hi)\W{0,2})?\s*$"
    ),
    "pt": (
        r"(?P<who>\b\w+)(?:\s+\w+mente)?\s+(?:disse|diz|disse-me|disse-nos|perguntou)\b,?"
        r"\s*[\"“]?\s*(?:(?:ol(?:á|a)|oi)\W{0,2})?\s*$"
    ),
}

#: Tokens that mark an author tag as an ORGANISATION rather than a person. The punctuation and
#: digit classes are structural and shared; the WORDS are what move.
#:
#: Every row is the ENGLISH BASE *PLUS* its own additions, never a translation — and that is not
#: laziness, it is how these tags are actually written. "Cadena SER Media", "Radio Globo
#: Produções", "ZDF Studios" all carry English or borrowed markers inside a non-English feed, so
#: a row that replaced the base would stop matching them. The base below is lifted VERBATIM from
#: the single live regex, including the three measured subsets its comments record (news-outlet
#: suffixes; the institution tokens measured across 4,307 roster entries and 13,642 Person nodes,
#: matching 5 names, all organisations; and `plus`, the only carrier across 17,949 person names
#: being "China Plus" itself) — see `hosts.py` for those notes in full.
NONPERSON_AUTHOR_BASE: Tuple[str, ...] = (
    "podcasts?",
    "media",
    "networks?",
    "productions?",
    "studios?",
    "radio",
    "fm",
    "news",
    "inc",
    "llc",
    "ltd",
    "co",
    "company",
    "corp",
    "shows?",
    "entertainment",
    "audio",
    "broadcasting",
    "group",
    "labs?",
    "times",
    "journal",
    "tribune",
    "gazette",
    "herald",
    "chronicle",
    "magazine",
    "quarterly",
    "newspaper",
    "gmbh",
    "plc",
    "centers?",
    "centres?",
    "universit(?:y|ies)",
    "colleges?",
    "institutes?",
    "foundations?",
    "committees?",
    "councils?",
    "associations?",
    "societies",
    "society",
    "museums?",
    "librar(?:y|ies)",
    "plus",
)

NONPERSON_AUTHOR_WORDS: Dict[str, Tuple[str, ...]] = {
    "en": NONPERSON_AUTHOR_BASE,
    "es": NONPERSON_AUTHOR_BASE
    + (
        "medios?",
        "redes?",
        "producciones?",
        "noticias",
        "sa",
        "sl",
        "slu",
        "compa(?:ñ|n)(?:í|i)a",
        "programas?",
        "entretenimiento",
        "difusi(?:ó|o)n",
        "grupo",
        "laboratorios?",
        "diario",
        "revista",
        "peri(?:ó|o)dico",
        "gaceta",
        "cr(?:ó|o)nica",
        "centros?",
        "universidad(?:es)?",
        "colegios?",
        "institutos?",
        "editorial",
        "fundaci(?:ó|o)n(?:es)?",
        "comit(?:é|e)s?",
        "consejos?",
        "asociaci(?:ó|o)n(?:es)?",
        "sociedad(?:es)?",
        "museos?",
        "bibliotecas?",
    ),
    "it": NONPERSON_AUTHOR_BASE
    + (
        "reti?",
        "produzioni?",
        "studi",
        "notizie",
        "spa",
        "srl",
        "societ(?:à|a)",
        "programmi?",
        "intrattenimento",
        "emittente",
        "gruppo",
        "laboratori?",
        "giornale",
        "rivista",
        "quotidiano",
        "gazzetta",
        "cronaca",
        "centri?",
        "universit(?:à|a)",
        "collegi?",
        "istituti?",
        "editrice",
        "fondazion[ei]",
        "comitat[oi]",
        "consigli?",
        "associazion[ei]",
        "musei?",
        "bibliotec[ah]e?",
    ),
    "fr": NONPERSON_AUTHOR_BASE
    + (
        "m(?:é|e)dias?",
        "r(?:é|e)seaux?",
        "actualit(?:é|e)s?",
        "sas",
        "sarl",
        "soci(?:é|e)t(?:é|e)",
        "(?:é|e)missions?",
        "divertissement",
        "diffusion",
        "groupe",
        "laboratoires?",
        "revue",
        "quotidien",
        "chronique",
        "centres?",
        "universit(?:é|e)s?",
        "coll(?:è|e)ges?",
        "instituts?",
        "(?:é|e)ditions?",
        "fondations?",
        "comit(?:é|e)s?",
        "conseils?",
        "mus(?:é|e)es?",
        "biblioth(?:è|e)ques?",
    ),
    "de": NONPERSON_AUTHOR_BASE
    + (
        "medien",
        "netzwerke?",
        "produktionen?",
        "nachrichten",
        "ag",
        "kg",
        "ohg",
        "e\\.?v",
        "sendungen?",
        "unterhaltung",
        "rundfunk",
        "gruppe",
        "labore?",
        "zeitung",
        "zeitschrift",
        "magazin",
        "anzeiger",
        "chronik",
        "zentren?",
        "universit(?:ä|a)ten?",
        "hochschulen?",
        "institute?",
        "verlag",
        "stiftung(?:en)?",
        "aussch(?:ü|u)sse?",
        "r(?:ä|a)te?",
        "vereine?",
        "gesellschaft(?:en)?",
        "museen",
        "bibliotheken?",
    ),
    "pt": NONPERSON_AUTHOR_BASE
    + (
        "m(?:é|e)dia",
        "redes?",
        "produ(?:ç|c)(?:õ|o)es",
        "est(?:ú|u)dios?",
        "r(?:á|a)dio",
        "not(?:í|i)cias",
        "lda",
        "unipessoal",
        "companhia",
        "programas?",
        "entretenimento",
        "(?:á|a)udio",
        "radiodifus(?:ã|a)o",
        "grupo",
        "laborat(?:ó|o)rios?",
        "jornal",
        "revista",
        "di(?:á|a)rio",
        "gazeta",
        "cr(?:ó|o)nica",
        "centros?",
        "universidade(?:s)?",
        "col(?:é|e)gios?",
        "institutos?",
        "editora",
        "funda(?:ç|c)(?:õ|o)es",
        "comit(?:é|e)s?",
        "conselhos?",
        "associa(?:ç|c)(?:õ|o)es",
        "sociedade(?:s)?",
        "museus?",
        "bibliotecas?",
    ),
}

#: Nobiliary and patronymic particles inside a name ("van der Berg", "de la Cruz").
#:
#: DELIBERATELY ONE SHARED LIST, not six. A Dutch `van` appears in an English show's guest list and
#: a Spanish `de la` in a French one, so narrowing each row to its own language's particles would
#: lose names the English row already catches. And the English list is ALREADY a cross-lingual
#: union — it carries Italian `della`/`di`, French `du`/`le`, German `von`/`der`, Portuguese
#: `dos`/`das`, Spanish `el` — so every row is that list, UNCHANGED, in its measured order.
#:
#: Candidates NOT added here on purpose: Italian `dei`/`degli`/`delle`, German `zu`/`zum`, French
#: `des`, Portuguese `do`. Each would widen what counts as a name in EVERY language including
#: English, and widening the name shape is a recall decision that needs a measurement, not a
#: side effect of moving vocabulary into this module.
STATED_PARTICLES: Dict[str, Tuple[str, ...]] = {
    lang: (
        "van",
        "von",
        "de",
        "da",
        "del",
        "della",
        "di",
        "du",
        "la",
        "le",
        "der",
        "den",
        "ter",
        "ten",
        "al",
        "bin",
        "ibn",
        "dos",
        "das",
        "el",
    )
    for lang in TIER_1_LANGUAGES
}


# =============================================================================================
# SELF-INTRODUCTION CUES
#
# What a host says to name THEMSELF. The guards that make each of these safe (the show
# condition on the branded form, the required comma on the with-me form) are STRUCTURE and stay
# in `hosts.py` with the prose explaining why — only the words move here.
# =============================================================================================

#: "I'm X" / "my name is X". ``%(names)s`` is the capture.
HOST_SELF_INTRO: Dict[str, str] = {
    "en": (
        r"\b(?:I'?m|I am|[Mm]y name is|[Mm]y name['’]s)\s+"
        r"(?:(?:your|the)\s+(?:co-?)?host(?:\s+(?:for\s+)?today)?,?\s+)?"
        r"(%(names)s)"
    ),
    "es": (
        r"\b(?i:soy|me llamo|mi nombre es)\s+"
        r"(?:(?:tu|su|el|la)\s+(?:co)?(?:anfitri(?:ó|o)n|anfitriona|presentador(?:a)?)"
        r"(?:\s+de\s+hoy)?,?\s+)?"
        r"(?P<names>%(names)s)"
    ),
    "it": (
        r"\b(?i:sono|mi chiamo|il mio nome (?:è|e'|e))\s+"
        r"(?:(?:il|la|tuo|tua)\s+(?:co)?(?:conduttor(?:e|ice)|presentator(?:e|ice))"
        r"(?:\s+di\s+oggi)?,?\s+)?"
        r"(?P<names>%(names)s)"
    ),
    "fr": (
        r"\b(?i:je suis|je m'appelle|mon nom est)\s+"
        r"(?:(?:votre|ton|le|la)\s+(?:co)?(?:animateur|animatrice|pr(?:é|e)sentateur"
        r"|pr(?:é|e)sentatrice)(?:\s+d'?aujourd'?hui)?,?\s+)?"
        r"(?P<names>%(names)s)"
    ),
    "de": (
        r"\b(?i:ich bin|ich hei(?:ß|ss)e|mein name ist)\s+"
        r"(?:(?:dein|ihr|euer|der|die)\s+(?:ko)?(?:gastgeber(?:in)?|moderator(?:in)?)"
        r"(?:\s+von\s+heute)?,?\s+)?"
        r"(?P<names>%(names)s)"
    ),
    "pt": (
        r"\b(?i:sou|eu sou|chamo-me|me chamo|o meu nome (?:é|e)|meu nome (?:é|e))\s+"
        r"(?:(?:o|a|teu|tua|seu|sua)\s+(?:co)?(?:anfitri(?:ã|a)o|anfitri(?:ã|a)|apresentador(?:a)?)"
        r"(?:\s+de\s+hoje)?,?\s+)?"
        # The article before a NAME: "Eu sou a Branca Vianna", "eu sou o Eduardo" — ordinary
        # Portuguese, and how Rádio Novelo's host opens both measured episodes (2026-10-09).
        # The name capture still needs a capital, so "sou a primeira" stays out.
        r"(?:(?:o|a)\s+)?"
        r"(?P<names>%(names)s)"
    ),
}

#: The BRANDED open: "it's X with <Show>". ``%(names)s`` is the person, ``%(show)s`` the show.
#: The trailing word is what turns a statement of fact into a byline, so each row spells out its
#: language's equivalents of "with / from / for" — and the show condition in `hosts.py` is what
#: keeps a sponsor read ("it's <Name> from <Company>") out.
HOST_BRANDED_INTRO: Dict[str, str] = {
    "en": r"\b[Ii]t'?s\s+(%(names)s)\s+(?:with|from|for)\s+(%(show)s)",
    "es": (
        r"\b(?i:es|soy|aqu(?:í|i) est(?:á|a))\s+(?P<names>%(names)s)"
        r"\s+(?:con|de|desde|para)\s+(?P<show>%(show)s)"
    ),
    "it": (
        r"\b(?i:(?:è|e'|e)|sono|qui (?:c'(?:è|e)))\s+(?P<names>%(names)s)"
        r"\s+(?:con|da|di|per)\s+(?P<show>%(show)s)"
    ),
    "fr": (
        r"\b(?i:c'?est|voici)\s+(?P<names>%(names)s)"
        r"\s+(?:avec|de|depuis|pour)\s+(?P<show>%(show)s)"
    ),
    "de": (
        r"\b(?i:hier ist|das ist|es ist)\s+(?P<names>%(names)s)"
        r"\s+(?:mit|von|bei|f(?:ü|u)r)\s+(?P<show>%(show)s)"
    ),
    "pt": (
        r"\b(?i:(?:é|e)|sou|aqui (?:é|e)|aqui est(?:á|a))\s+(?P<names>%(names)s)"
        r"\s+(?:com|de|d[oa]|para)\s+(?P<show>%(show)s)"
    ),
}

#: "with me, X" — the broadcast idiom for naming ONESELF. The COMMA is required and the
#: "joining me" form is excluded in every row, for the reason `hosts.py` documents: "joining me,
#: X" introduces somebody ELSE, and admitting it paints a guest's name onto the host's voice.
HOST_WITH_ME_INTRO: Dict[str, str] = {
    "en": r"\b(?:with|and)\s+me,\s+(%(names)s)",
    "es": r"\b(?i:(?:y\s+)?conmigo),\s+(?P<names>%(names)s)",
    "it": r"\b(?i:con|e)\s+me,\s+(?P<names>%(names)s)",
    "fr": r"\b(?i:avec|et)\s+moi,\s+(?P<names>%(names)s)",
    "de": r"\b(?i:mit|und)\s+mir,\s+(?P<names>%(names)s)",
    "pt": r"\b(?i:(?:e\s+)?comigo),\s+(?P<names>%(names)s)",
}

#: The episode's own prose handing off to a named host: "X speaks with", "X is joined by".
#: Measured over the 2,256-episode production snapshot — see `hosts.py` for why the capture is
#: held to TWO tokens. Only the VERB TAIL and the "and" joiner are language-dependent.
EPISODE_HOST_CUE: Dict[str, str] = {
    "en": r"(?:(?:is|are)\s+joined\s+by|speaks?\s+with|sits?\s+down\s+with)",
    "es": (
        r"(?:(?:est(?:á|a)|est(?:á|a)n)\s+acompa(?:ñ|n)ad[oa]s?\s+por"
        r"|habla\s+con|conversa\s+con|se\s+sienta\s+con)"
    ),
    "it": (
        r"(?:(?:è|e'|e)\s+accompagnat[oa]\s+da|sono\s+accompagnati\s+da"
        r"|parla\s+con|conversa\s+con|si\s+siede\s+con)"
    ),
    "fr": (
        r"(?:est\s+accompagn(?:é|e)e?\s+(?:par|de)|sont\s+accompagn(?:é|e)e?s\s+(?:par|de)"
        r"|parle\s+avec|s'?entretient\s+avec|discute\s+avec)"
    ),
    "de": (
        r"(?:wird\s+begleitet\s+von|werden\s+begleitet\s+von"
        r"|spricht\s+mit|unterh(?:ä|a)lt\s+sich\s+mit|setzt\s+sich\s+mit)"
    ),
    "pt": (
        r"(?:(?:est(?:á|a)|est(?:ã|a)o)\s+acompanhad[oa]s?\s+por"
        r"|fala\s+com|conversa\s+com|senta-se\s+com)"
    ),
}


# =============================================================================================
# FUNCTION WORDS
#
# Articles, prepositions and conjunctions. These are the smallest entries in this module and the
# easiest to leave behind, because an English article inside a regex does not LOOK like
# vocabulary — but `^(?:the|a|an)\s+` strips nothing off "El Podcast de Trilha", so the show
# fails to match its own name and seats itself as its own host, which is the exact #876 defect
# in a different language.
# =============================================================================================

#: A leading article, stripped before comparing a candidate against a show's name.
LEADING_ARTICLE: Dict[str, str] = {
    "en": r"^(?:the|a|an)\s+",
    "es": r"^(?:el|la|los|las|un|una|unos|unas|lo)\s+",
    "it": r"^(?:il|lo|la|i|gli|le|un|uno|una|un')\s*",
    "fr": r"^(?:le|la|les|un|une|des|l')\s*",
    "de": r"^(?:der|die|das|den|dem|des|ein|eine|einen|einem|einer|eines)\s+",
    "pt": r"^(?:o|a|os|as|um|uma|uns|umas)\s+",
}

#: An article or genitive IMMEDIATELY BEFORE a candidate ("...the X", "...of X"), which means the
#: capitalised run is a thing being referred to rather than a person being named. The Romance rows
#: carry the contracted forms, which is most of the work: Italian `dello`, French `du`/`des` are a
#: preposition and an article fused into one token.
#:
#: GERMAN DELIBERATELY OMITS `von`/`vom`, AND THE OMISSION IS THE WHOLE ROW. `von` is German's
#: genitive, so it belongs here by translation — but it is ALSO German's by-agent marker, the one
#: word every German host statement puts directly in front of the host ("Pfadgespräche wird
#: moderiert von Lena Hofmann"). Measured 2026-10-03: with `von` in this row, all four other
#: languages named their host and German named NOBODY — the guard meant to stop "…Council of the
#: Americas Online" ate the host instead, on every German feed, silently. German expresses the
#: genitive with `des`/`der` in this position anyway, and both are still here.
#:
#: No other row has the collision: Spanish and Portuguese mark the agent with `por`/`pel[oa]`,
#: Italian with `da`, French with `par`, and none of those is an article.
ARTICLE_BEFORE: Dict[str, str] = {
    "en": r"\b(?:the|of)\s+$",
    "es": r"\b(?:el|la|los|las|de|del|de\s+l[ao]s?)\s+$",
    "it": r"\b(?:il|lo|la|i|gli|le|di|del|dello|della|dei|degli|delle)\s+$",
    "fr": r"\b(?:le|la|les|de|du|des|d')\s*$",
    "de": r"\b(?:der|die|das|den|dem|des|zum|zur)\s+$",
    "pt": r"\b(?:o|a|os|as|de|do|da|dos|das)\s+$",
}

#: A preposition that makes the following capitalised run a PLACE or an employer, not a person
#: ("At Planet Money", "From Brussels"). German and Portuguese fuse these with the article too.
PLACE_PREPOSITION: Dict[str, str] = {
    "en": r"(?:At|From|In|On)\s+",
    "es": r"(?:En|Desde|De|Dentro\s+de)\s+",
    "it": r"(?:A|Ad|In|Da|Dal|Presso)\s+",
    "fr": r"(?:(?:À|A)|Au|Aux|De|Depuis|Dans|Chez)\s+",
    "de": r"(?:In|Im|Bei|Beim|Von|Vom|Aus|Auf)\s+",
    "pt": r"(?:Em|No|Na|De|Desde|Dentro\s+de)\s+",
}

#: The conjunction joining two names in a list ("Alexandra Karppi, and Nina Panikova"). The
#: ampersand and comma are punctuation and stay in `hosts.py`; only the WORD lives here.
NAME_LIST_CONJUNCTION: Dict[str, str] = {
    "en": "and",
    "es": "y|e",
    "it": "e|ed",
    "fr": "et",
    "de": "und",
    "pt": "e",
}

#: The preposition in "<Name> in <Place>" ("Eric Olander in Vietnam"), which marks a place that
#: is extracted as a name of its own only to be refused by the person check.
PLACE_PREPOSITION_IN: Dict[str, str] = {
    "en": "in",
    "es": "en",
    "it": "in|a",
    "fr": "(?:à|a)|en",
    "de": "in",
    "pt": "em|n[oa]",
}

#: A trailing "with <Name>" in a TITLE names the host, not the show ("Invest Like the Best with
#: Patrick O'Shaughnessy"). Stripped before comparing a candidate against the show's name.
TITLE_WITH_SUFFIX: Dict[str, str] = {
    "en": r"\s+with\s+.+$",
    "es": r"\s+con\s+.+$",
    "it": r"\s+con\s+.+$",
    "fr": r"\s+avec\s+.+$",
    "de": r"\s+mit\s+.+$",
    "pt": r"\s+com\s+.+$",
}


# =============================================================================================
# THE ROSTER AND RESOLUTION VOCABULARY
#
# `providers/ml/diarization/roster.py`, `speaker_detectors/resolution.py` and `gi/speakers.py`
# each carried their own English-only copies of the cues below. They are listed here as WORDS
# rather than anchored patterns because the three modules anchor them differently on purpose —
# `roster.py` matches a case-FOLDED transcript, `resolution.py` anchors to the end of a window,
# `hosts.py` matches capitalised prose — and the anchor is structure, not vocabulary.
# =============================================================================================

#: "I'm X" / "my name is X" — the words only. Every site adds its own anchor and name capture.
#:
#: NOT "this is": that is how a host introduces a GUEST ("this is Matthew Cobb's seventh book"),
#: and framed as a self-introduction it told the model the host was the guest (#2075). The
#: exclusion holds in every language, which is why `THIS_IS_INTRO` below is a separate map that
#: only ever binds against metadata.
#:
#: THE ENGLISH ROW WIDENS THREE SITES, DELIBERATELY. `roster.py`'s match-form detector had
#: `i'm|i am|my name is`, `roster.py`'s sign-off reader had `I['’]?m` alone, and
#: `resolution.py`'s window anchor had a third variant — three hand-written copies of one speech
#: act, each slightly different, while `roster.py`'s own comment says the match-form regexes
#: "match the SAME cue vocabulary (imported, so they cannot drift)". They had drifted; they were
#: not imported. This row is the union `hosts.HOST_SELF_INTRO` already used, so all four sites
#: now read it: `i['’]?m` (which also admits the apostrophe-stripped "im" a folded transcript
#: produces) and `my name's`, which only the hosts.py copy had.
SELF_INTRO_WORDS: Dict[str, str] = {
    "en": r"i['’]?m|i am|my name is|my name['’]s",
    "es": r"soy|yo soy|me llamo|mi nombre es",
    "it": r"sono|io sono|mi chiamo|il mio nome (?:è|e'|e)",
    "fr": r"je suis|je m'appelle|mon nom est",
    "de": r"ich bin|ich hei(?:ß|ss)e|mein name ist",
    "pt": r"sou|eu sou|chamo-me|me chamo|o meu nome (?:é|e)|meu nome (?:é|e)",
}

#: "This is X" — a show naming ITSELF as often as a person naming themselves ("This is Unhedged",
#: "This is Planet Money"), so a match binds ONLY when the episode metadata states X as a person.
#: The ambiguity is identical in all six: every one of these doubles as a station ident.
THIS_IS_INTRO: Dict[str, str] = {
    "en": r"[Tt]his is",
    "es": r"(?:esto|este|esta) es|aqu(?:í|i) (?:est(?:á|a)|es)",
    "it": r"quest[oa] (?:è|e'|e)|qui (?:è|e'|e)",
    "fr": r"c'?est|voici|ici",
    "de": r"(?:dies|das) ist|hier ist",
    "pt": r"(?:isto|este|esta) (?:é|e)|aqui (?:é|e|est(?:á|a))",
}

#: A TEMPORAL marker that turns a past-tense cue into a RECAP. "last month we spoke with X" names
#: nobody in this episode, and admitting it misattributes the named person to whatever voice
#: speaks next.
RECAP_MARKERS: Dict[str, str] = {
    "en": (
        r"last\s+(?:week|month|year|night|time)|earlier|previously|recently|yesterday"
        r"|back\s+then|a\s+while\s+ago|the\s+other\s+(?:day|week)"
    ),
    "es": (
        r"(?:la\s+semana|el\s+mes|el\s+a(?:ñ|n)o)\s+pasad[oa]|anoche|antes|anteriormente"
        r"|recientemente|ayer|en\s+aquel\s+momento|hace\s+un\s+tiempo|el\s+otro\s+d(?:í|i)a"
    ),
    "it": (
        r"(?:la\s+settimana|il\s+mese|l'anno)\s+scors[oa]|ieri\s+sera|prima|in\s+precedenza"
        r"|recentemente|ieri|allora|qualche\s+tempo\s+fa|l'altro\s+giorno"
    ),
    "fr": (
        r"(?:la\s+semaine|le\s+mois|l'ann(?:é|e)e)\s+derni(?:è|e)re?|hier\s+soir|avant"
        r"|pr(?:é|e)c(?:é|e)demment|r(?:é|e)cemment|hier|(?:à|a)\s+l'(?:é|e)poque"
        r"|il\s+y\s+a\s+quelque\s+temps|l'autre\s+jour"
    ),
    "de": (
        r"(?:letzte|letzten|letztes)\s+(?:woche|monat|jahr|nacht|mal)|gestern\s+abend|fr(?:ü|u)her"
        r"|zuvor|k(?:ü|u)rzlich|gestern|damals|vor\s+einiger\s+zeit|am\s+anderen\s+tag"
    ),
    "pt": (
        r"(?:a\s+semana|o\s+m(?:ê|e)s|o\s+ano)\s+passad[oa]|ontem\s+(?:(?:à|a)\s+)?noite|antes"
        r"|anteriormente|recentemente|ontem|naquela\s+(?:altura|(?:é|e)poca)"
        r"|h(?:á|a)\s+algum\s+tempo|no\s+outro\s+dia"
    ),
}

#: FUNCTION WORDS between a name and an affiliation ("Jane Smith of the Financial Times"). Used to
#: decide a token is not a contradicting surname, so the bare-first-name relaxation may still
#: apply; mis-classification errs toward abstain, not a wrong name.
#:
#: THE ENGLISH ROW IS THE LIVE SET, VERBATIM, and the two-letter words in it are the point: a
#: genuine SHORT surname (Ng, Wu, Li, Xu — common on an AI-podcast corpus) must not be mistaken
#: for a function word, which is why `by`, `we`, `me`, `us`, `go`, `he`, `do`, `if`, `no`, `so`,
#: `or`, `up`, `it` are all present (second advisor review). An earlier draft of this map replaced
#: them with English pronouns and possessives and dropped thirteen of the thirty — the kind of
#: silent narrowing a hand-copy produces.
INTRO_AFFILIATION_TOKENS: Dict[str, FrozenSet[str]] = {
    "en": frozenset(
        {
            "of",
            "from",
            "at",
            "with",
            "and",
            "the",
            "our",
            "a",
            "an",
            "in",
            "on",
            "for",
            "to",
            "here",
            "as",
            "is",
            "by",
            "or",
            "so",
            "if",
            "up",
            "it",
            "my",
            "me",
            "us",
            "do",
            "go",
            "no",
            "he",
            "we",
        }
    ),
    "es": frozenset(
        {
            "de",
            "del",
            "desde",
            "en",
            "con",
            "y",
            "e",
            "el",
            "la",
            "los",
            "las",
            "un",
            "una",
            "nuestro",
            "nuestra",
            "para",
            "a",
            "al",
            "aquí",
            "aqui",
            "como",
            "es",
            "por",
            "o",
            "u",
            "si",
            "lo",
            "mi",
            "me",
            "nos",
            "yo",
            "él",
            "el",
            "no",
            "ya",
            "va",
        }
    ),
    "it": frozenset(
        {
            "di",
            "del",
            "dello",
            "della",
            "dei",
            "degli",
            "delle",
            "da",
            "dal",
            "in",
            "con",
            "e",
            "ed",
            "il",
            "lo",
            "la",
            "i",
            "gli",
            "le",
            "un",
            "uno",
            "una",
            "nostro",
            "nostra",
            "per",
            "a",
            "ad",
            "qui",
            "come",
            "è",
            "e'",
            "per",
            "o",
            "se",
            "mi",
            "ci",
            "io",
            "lui",
            "non",
            "va",
            "su",
        }
    ),
    "fr": frozenset(
        {
            "de",
            "du",
            "des",
            "depuis",
            "à",
            "a",
            "au",
            "aux",
            "chez",
            "avec",
            "et",
            "le",
            "la",
            "les",
            "un",
            "une",
            "notre",
            "nos",
            "pour",
            "ici",
            "comme",
            "est",
            "par",
            "ou",
            "si",
            "me",
            "moi",
            "nous",
            "je",
            "il",
            "ne",
            "va",
            "en",
            "y",
        }
    ),
    "de": frozenset(
        {
            "von",
            "vom",
            "aus",
            "bei",
            "beim",
            "in",
            "im",
            "mit",
            "und",
            "der",
            "die",
            "das",
            "den",
            "dem",
            "des",
            "ein",
            "eine",
            "einen",
            "unser",
            "unsere",
            "für",
            "fur",
            "zu",
            "zum",
            "zur",
            "hier",
            "als",
            "ist",
            "durch",
            "oder",
            "ob",
            "mir",
            "mich",
            "uns",
            "ich",
            "er",
            "wir",
            "nie",
            "so",
            "ja",
            "an",
        }
    ),
    "pt": frozenset(
        {
            "de",
            "do",
            "da",
            "dos",
            "das",
            "desde",
            "em",
            "no",
            "na",
            "com",
            "e",
            "o",
            "a",
            "os",
            "as",
            "um",
            "uma",
            "nosso",
            "nossa",
            "para",
            "aqui",
            "como",
            "é",
            "por",
            "ou",
            "se",
            "me",
            "nos",
            "eu",
            "ele",
            "não",
            "nao",
            "vai",
            "ao",
            "à",
        }
    ),
}

#: GENERATIONAL suffixes — the tail of a person's own name ("Martin Luther King Jr", "Louis XIV").
#:
#: NOT the same set as `NAME_SUFFIXES`, and the difference matters: that one holds CREDENTIALS
#: (`md`, `phd`, `esq.`, with their dotted forms) for stripping a qualification off a stated name,
#: while this one holds `jr`/`sr` and the roman numerals and keeps `v`, which the credential set
#: does not have. Pointing one at the other loses `v` and gains three credentials.
#:
#: One shared row: roman numerals and the borrowed `jr`/`sr` are written the same in all six.
GENERATIONAL_SUFFIXES: Dict[str, FrozenSet[str]] = {
    lang: frozenset({"jr", "sr", "ii", "iii", "iv", "v"}) for lang in TIER_1_LANGUAGES
}

#: What follows a SIGN-OFF self-introduction: "I'm Tracy Allaway. You can follow me at…". The
#: self-introduction reader looks only at the start of each voice, which is right for an opening
#: "I'm <host>" — but a publisher's diarization can fold the whole opening into one fragment and
#: leave the host's main voice identified only at the close (#2075, Odd Lots' Tungsten episode).
SIGN_OFF_CUES: Dict[str, str] = {
    "en": (
        r"(?:And\s+)?(?:you\s+can\s+follow|follow\s+(?:me|us)|see\s+you"
        r"|talk\s+(?:to\s+you\s+)?soon|thanks?\s+(?:you\s+)?for\s+listening|until\s+next)"
    ),
    "es": (
        r"(?:Y\s+)?(?:puedes\s+seguirme|s(?:í|i)gueme|s(?:í|i)guenos|nos\s+vemos"
        r"|hablamos\s+pronto|gracias\s+por\s+escuchar|hasta\s+la\s+pr(?:ó|o)xima)"
    ),
    "it": (
        r"(?:E\s+)?(?:puoi\s+seguirmi|seguimi|seguici|ci\s+vediamo|a\s+presto"
        r"|grazie\s+per\s+l'ascolto|alla\s+prossima)"
    ),
    "fr": (
        r"(?:Et\s+)?(?:vous\s+pouvez\s+me\s+suivre|suivez-(?:moi|nous)|(?:à|a)\s+bient(?:ô|o)t"
        r"|merci\s+d'?(?:avoir\s+)?(?:é|e)cout(?:é|e)|(?:à|a)\s+la\s+prochaine)"
    ),
    "de": (
        r"(?:Und\s+)?(?:du\s+kannst\s+mir\s+folgen|folg(?:t|e)\s+(?:mir|uns)|bis\s+bald"
        r"|wir\s+sprechen\s+bald|danke\s+f(?:ü|u)rs\s+zuh(?:ö|o)ren|bis\s+zum\s+n(?:ä|a)chsten)"
    ),
    "pt": (
        r"(?:E\s+)?(?:podes\s+seguir-me|segue-me|sigam-nos|at(?:é|e)\s+(?:j(?:á|a)|breve)"
        r"|falamos\s+em\s+breve|obrigad[oa]\s+por\s+(?:ouvir|escutar)"
        r"|at(?:é|e)\s+(?:à|a)\s+pr(?:ó|o)xima)"
    ),
}

#: A GREETING that opens a turn, before a first name ("Hey, Jordan. Good morning."). The voice
#: that greets a name is NOT that name (#2078: ChinaTalk published `Jordan Schneider` on the voice
#: saying "Hey, Jordan" while Jordan's real voice went unnamed).
GREETING_AT_OPEN: Dict[str, str] = {
    "en": r"hey|hi|hello|good\s+(?:morning|afternoon|evening)|morning",
    "es": r"hola|buenas|buen(?:os)?\s+(?:d(?:í|i)as|tardes|noches)|qu(?:é|e)\s+tal",
    "it": r"ciao|salve|buon(?:giorno|asera|a\s+sera)|buon\s+pomeriggio",
    "fr": r"salut|bonjour|bonsoir|coucou|all(?:ô|o)",
    "de": r"hallo|hi|hey|guten\s+(?:morgen|tag|abend)|moin|servus",
    "pt": r"ol(?:á|a)|oi|bom\s+dia|boa\s+(?:tarde|noite)|viva",
}

#: Tokens that mark a "host" string as a PUBLISHER rather than a person, for the GI speaker
#: reader. English base plus each market's own words, for the same reason as
#: `NONPERSON_AUTHOR_WORDS`: a non-English feed's label still carries borrowed markers.
NON_PERSON_LABEL_BASE: FrozenSet[str] = frozenset(
    {
        "bloomberg",
        "industries",
        "media",
        "podcast",
        "podcasts",
        "network",
        "news",
        "studios",
        "inc",
        "llc",
    }
)
NON_PERSON_LABEL_TOKENS: Dict[str, FrozenSet[str]] = {
    lang: NON_PERSON_LABEL_BASE | extra
    for lang, extra in {
        "en": frozenset(),
        "es": frozenset(
            {"medios", "red", "redes", "noticias", "estudios", "industrias", "emisora", "cadena"}
        ),
        "it": frozenset({"rete", "reti", "notizie", "studi", "industrie", "emittente"}),
        "fr": frozenset(
            {"médias", "medias", "réseau", "reseau", "actualités", "actualites", "industries"}
        ),
        "de": frozenset(
            {"medien", "netzwerk", "nachrichten", "industrie", "rundfunk", "gmbh", "verlag"}
        ),
        "pt": frozenset(
            {
                "média",
                "midia",
                "rede",
                "redes",
                "notícias",
                "noticias",
                "estúdios",
                "estudios",
                "indústrias",
                "industrias",
                "emissora",
            }
        ),
    }.items()
}


# =============================================================================================
# DESCRIPTORS THAT ARE NOT NAMES
# =============================================================================================

#: A phrase ABOUT a person that arrived where a name should be: "Pulitzer Prize-winning" (main,
#: 11e1e425e — the only such published name across the prod corpus, Freakonomics 2026-10-03).
#: Searched case-insensitively anywhere in the candidate name; any hit refuses it.
#:
#: SHAPES DIFFER BY LANGUAGE, which is why this is not one regex with translated words. English
#: and German glue the descriptor onto the prize/place with a hyphen ("Emmy-winning",
#: "Grammy-prämierte", "Berlin-based", "Oscar-nominierte"). The Romance languages put it in a
#: word of its own ("periodista galardonada", "journaliste primée", "giornalista premiata"), so
#: their rows match the whole word. The English row is main's pattern verbatim.
#:
#: AUTHORED, NOT MEASURED, like every non-English row here: no non-English prod data has shown
#: one of these in a name yet. Each word is one a real person's name does not contain.
DESCRIPTOR_PATTERNS: Dict[str, str] = {
    "en": r"\w-(?:winning|nominated|based|born|selling|renowned|acclaimed)\b",
    "es": (
        r"\b(?:galardonad[oa]s?|premiad[oa]s?|nominad[oa]s?|aclamad[oa]s?|afincad[oa]s?"
        r"|nacid[oa]s?|reconocid[oa]s?)\b"
    ),
    "it": (
        r"\b(?:premiat[oaie]|pluripremiat[oaie]|candidat[oaie]\s+all'oscar|acclamat[oaie]"
        r"|rinomat[oaie]|nat[oaie]\s+a)\b"
    ),
    "fr": (
        r"\b(?:prim(?:é|e)e?s?|r(?:é|e)compens(?:é|e)e?s?|nomin(?:é|e)e?s?\s+aux?"
        r"|acclam(?:é|e)e?s?|renomm(?:é|e)e?s?|bas(?:é|e)e?s?\s+(?:à|a)|n(?:é|e)e?\s+(?:à|a))\b"
    ),
    "de": (
        r"\w-(?:pr(?:ä|ae)miert|gekr(?:ö|oe)nt|nominiert|ausgezeichnet|basiert|geboren)\w*"
        r"|\b(?:preisgekr(?:ö|oe)nt|ausgezeichnet|preistr(?:ä|ae)ger)\w*"
    ),
    "pt": (
        r"\b(?:premiad[oa]s?|galardoad[oa]s?|nomead[oa]s?|aclamad[oa]s?|renomad[oa]s?"
        r"|radicad[oa]s?|nascid[oa]s?)\b"
    ),
}
