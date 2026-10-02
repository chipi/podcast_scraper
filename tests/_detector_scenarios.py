"""Per-language detector scenarios, as DATA — the test-side mirror of the vocabulary maps.

WHY THIS EXISTS. The detector vocabularies became language-keyed maps on 2026-10-02
(``gi.filters.AD_PATTERNS_BY_LANGUAGE`` and friends). The tests exercising them did not: each test
file carried its own inline dicts of Spanish, Italian, French, German and Portuguese sentences, so
adding a sixth language meant editing test bodies in several files and hoping none were missed.
That is the same flat-list problem the source side just fixed, one layer up.

So the scenario is a row and the test is one body replayed over the rows. Adding a language means
adding a row HERE and a row in each vocabulary map — no new test code, and no language named inside
a test function.

FAIL-CLOSED, NOT FAIL-QUIET. ``TIER_1`` in the consuming tests derives from
``config/languages.yaml``, so enabling a language immediately fails the suite until its scenario
row exists. The alternative — tests that simply do not run for the new language — is the state this
replaces, and it looks identical to passing.

=== WHAT THESE SCENARIOS CAN AND CANNOT PROVE ===

They prove a vocabulary row is WELL-FORMED and INTERNALLY CONSISTENT: the ad patterns reach the
>= 2-distinct threshold on a sponsor read and stay under it on editorial prose; a leading cue
reaches the guest's name while the mentioned-only guard catches a name the episode is merely about;
the host and guest speech acts do not fire on each other; name cleanup strips prepositions without
eating name particles.

They CANNOT prove recall or precision on real speech, and the reason is structural rather than a
matter of writing better sentences: **the same person wrote the patterns and these fixtures.** A
phrase a real Spanish podcast uses and I did not think of is absent from both, so the test passes
and the detector misses it in production. Every non-English sentence below was authored 2026-10-02
alongside the patterns.

That measurement needs real material, and it is tracked per language:

* Spanish — #2255   * Italian — #2256   * French — #2257   * German — #2258   * Portuguese — #2259

all of which depend on #2187, the first non-English ASR run of any kind.

THE SAME CAVEAT APPLIES, SLIGHTLY WEAKER, TO FIXTURE EPISODES BUILT FROM THESE ROWS. Generating a
fixture transcript out of ``ad_read`` and asserting the ad filter cuts it is partly circular — it
cannot discover a phrase nobody wrote down. What it does add over a unit test is that the text goes
through the whole chain first: diarized into a screenplay with speaker labels, segmented,
offset-indexed, then excised. A pattern that matches a bare string and fails once the text carries
``Name:`` prefixes and segment boundaries is a real defect, and that class is exactly what the
English fixture corpus catches today. So: integration evidence, not recall evidence. Do not let a
green fixture test become a claim that a language is validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class DetectorScenario:
    """One language's worth of detector-triggering content.

    Bundled per language rather than split into six parallel dicts because the pieces are only
    meaningful together: the recall fixtures (``introduction``) and the precision fixture
    (``mention``) are a matched pair, and so are ``ad_read`` and ``clean_content``. A row with one
    half of a pair is a row that tests only the comfortable direction.
    """

    language: str

    #: A sponsor read carrying one phrase from each of the three ad categories — a disclosure, a
    #: spoken or hybrid URL, and a promo CTA. Must reach ``_AD_HITS_THRESHOLD`` distinct patterns.
    ad_read: str
    #: Ordinary editorial prose in the same language. Must stay UNDER the threshold: the threshold
    #: exists so a guest describing their own product is not cut, and this is the other direction.
    clean_content: str

    #: The host's opening — welcome, self-identification, and introducing the guest. These are the
    #: performed role, which is what the detector reads; talk time is deliberately not a signal.
    host_opening: str
    #: The guest's reply — the thanks-for-having-me act. Must NOT perform a host act: a guest
    #: misread as host is the #1169 failure, where the dominant voice was crowned.
    guest_reply: str

    #: The guest a description introduces, lowercase, as the detector sees it after normalization.
    guest_name: str
    #: A description that INTRODUCES ``guest_name`` with a leading cue. The cue must reach the name
    #: across the bounded gap the detector allows.
    introduction: str

    #: Someone the episode is ABOUT and who never speaks.
    mentioned_name: str
    #: A description mentioning ``mentioned_name``. The mentioned-only guard must fire, or a
    #: lawsuit defendant becomes a podcast guest (#876).
    mention: str

    #: Real names that begin with a particle and must survive cleanup VERBATIM. English can strip a
    #: leading "The"/"From" safely; these languages cannot, because their names start with exactly
    #: the words a symmetric translation would have added to the junk list.
    name_particles: Tuple[str, ...]
    #: ``(raw, cleaned)`` pairs of genuine junk the cleanup must still remove — so keeping the
    #: particles above does not cost the cleanup its actual job.
    leading_junk: Tuple[Tuple[str, str], ...]


DETECTOR_SCENARIOS: Dict[str, DetectorScenario] = {
    "en": DetectorScenario(
        language="en",
        ad_read=(
            "This episode is sponsored by Ramp. Go to ramp.com slash invest and use code POD to "
            "save 20 percent. Free trial for a limited time."
        ),
        clean_content="We talk about inflation and how the central bank responds to the crisis.",
        host_opening=(
            "Hello and welcome to Planet Money. I'm your host Alexi, and my guest today is Brian."
        ),
        guest_reply="Thank you very much for having me. Glad to be here.",
        guest_name="brian chesky",
        introduction="in this episode we are joined by Brian Chesky, the CEO of Airbnb",
        mentioned_name="brian chesky",
        mention="this week we discuss Brian Chesky's decision to restructure Airbnb",
        name_particles=(),
        leading_junk=(("At Planet Money", "Planet Money"),),
    ),
    "es": DetectorScenario(
        language="es",
        ad_read=(
            "Este episodio está patrocinado por Ramp. Entra a ramp punto com barra oferta y usa "
            "el código POD para ahorrar 20 por ciento. Prueba gratis por tiempo limitado."
        ),
        clean_content=("Hablamos de la inflación y de cómo el banco central responde a la crisis."),
        host_opening=(
            "Hola y bienvenidos a Planet Money. Soy tu anfitrión Alexi, y mi invitado de hoy es "
            "Brian."
        ),
        guest_reply="Muchas gracias por invitarme. Encantado de estar aquí.",
        guest_name="brian chesky",
        introduction="en este episodio nos acompaña Brian Chesky, el director de Airbnb",
        mentioned_name="brian chesky",
        mention="esta semana analizamos la decisión de Brian Chesky de reestructurar Airbnb",
        name_particles=("de la Fuente",),
        leading_junk=(("En Planet Money", "Planet Money"),),
    ),
    "it": DetectorScenario(
        language="it",
        ad_read=(
            "Questo episodio è sponsorizzato da Ramp. Vai su ramp punto com barra offerta e usa "
            "il codice POD per risparmiare 20 per cento. Prova gratuita per un tempo limitato."
        ),
        clean_content=("Parliamo dell'inflazione e di come la banca centrale risponde alla crisi."),
        host_opening=(
            "Ciao e benvenuti a Planet Money. Sono il vostro conduttore Alexi, e il mio ospite di "
            "oggi è Brian."
        ),
        guest_reply="Grazie mille per l'invito. Felice di essere qui.",
        guest_name="brian chesky",
        introduction="in questo episodio parliamo con Brian Chesky, il direttore di Airbnb",
        mentioned_name="brian chesky",
        mention="questa settimana analizziamo la decisione di Brian Chesky",
        name_particles=("Da Vinci", "Di Stefano"),
        leading_junk=(("Su Planet Money", "Planet Money"),),
    ),
    "fr": DetectorScenario(
        language="fr",
        ad_read=(
            "Cet épisode est sponsorisé par Ramp. Allez sur ramp point com slash offre et "
            "utilisez le code POD pour économiser 20 pour cent. Essai gratuit pour une durée "
            "limitée."
        ),
        clean_content=(
            "Nous parlons de l'inflation et de la réponse de la banque centrale à la crise."
        ),
        host_opening=(
            "Bonjour et bienvenue dans Planet Money. Je suis votre hôte Alexi, et mon invité du "
            "jour est Brian."
        ),
        guest_reply="Merci beaucoup de m'avoir invité. Ravi d'être là.",
        guest_name="brian chesky",
        introduction="dans cet épisode nous recevons Brian Chesky, le directeur d'Airbnb",
        mentioned_name="brian chesky",
        mention="cette semaine nous analysons la décision de Brian Chesky",
        name_particles=("De Gaulle", "Le Pen", "Du Bois"),
        leading_junk=(("Chez Planet Money", "Planet Money"),),
    ),
    "de": DetectorScenario(
        language="de",
        ad_read=(
            "Werbung. Diese Folge wird von Ramp gesponsert. Geht auf ramp punkt com schrägstrich "
            "angebot, mit dem Code POD spart ihr 20 Prozent. Kostenlose Testphase, nur für kurze "
            "Zeit."
        ),
        clean_content=(
            "Wir sprechen über die Inflation und wie die Zentralbank auf die Krise reagiert."
        ),
        host_opening=(
            "Hallo und willkommen bei Planet Money. Ich bin euer Gastgeber Alexi, und mein Gast "
            "heute ist Brian."
        ),
        guest_reply="Vielen Dank für die Einladung. Freut mich, hier zu sein.",
        guest_name="brian chesky",
        # GERMAN IS V2, and this fixture is the one that caught it: the subject follows the verb
        # the moment anything is fronted, so "in dieser Folge SPRECHEN WIR MIT ..." is the ordinary
        # shape rather than an inversion. The cue row originally matched subject-first only.
        introduction="in dieser Folge sprechen wir mit Brian Chesky, dem Chef von Airbnb",
        mentioned_name="brian chesky",
        mention="diese Woche sprechen wir über Brian Chesky und seine Entscheidung",
        name_particles=("von Neumann", "zu Guttenberg"),
        leading_junk=(("Bei Planet Money", "Planet Money"),),
    ),
    "pt": DetectorScenario(
        language="pt",
        ad_read=(
            "Este episódio é patrocinado por Ramp. Acesse ramp ponto com barra oferta e use o "
            "código POD para economizar 20 por cento. Teste grátis por tempo limitado."
        ),
        clean_content="Falamos sobre a inflação e como o banco central responde à crise.",
        host_opening=(
            "Olá e bem-vindos ao Planet Money. Eu sou o seu apresentador Alexi, e meu convidado "
            "de hoje é Brian."
        ),
        guest_reply="Muito obrigado por me receber. É um prazer estar aqui.",
        guest_name="brian chesky",
        introduction="neste episódio conversamos com Brian Chesky, o diretor do Airbnb",
        mentioned_name="brian chesky",
        mention="esta semana analisamos a decisão de Brian Chesky",
        name_particles=("da Silva", "dos Santos"),
        leading_junk=(("Em Planet Money", "Planet Money"),),
    ),
}

#: The languages that have a scenario row. Consumers compare this against the ENABLED set from
#: ``config/languages.yaml`` rather than trusting it — a row here for a language nobody ingests is
#: harmless, a missing row for one we do ingest is the failure worth catching.
SCENARIO_LANGUAGES = frozenset(DETECTOR_SCENARIOS)


def scenario_for(language: str) -> DetectorScenario:
    """The row for ``language``, raising rather than defaulting.

    No English fallback, matching ``ad_patterns_for`` and ``interview_cue_patterns_for``: a test
    silently re-running the English scenario for a language with no row would report coverage it
    does not have, which is the whole failure mode this module exists to prevent.
    """
    code = language.strip().lower().split("-")[0]
    try:
        return DETECTOR_SCENARIOS[code]
    except KeyError:
        raise KeyError(
            f"no detector scenario for {language!r}. A language enabled in config/languages.yaml "
            "needs a row in tests/_detector_scenarios.py AND a row in each vocabulary map "
            "(gi.filters.AD_PATTERNS_BY_LANGUAGE, speaker_detectors.constants.INTERVIEW_*, "
            "speaker_detectors.hosts._{HOST,GUEST}_SPEECH_ACTS_BY_LANGUAGE). The row here is what "
            "lets the existing tests exercise it without new test code."
        ) from None
