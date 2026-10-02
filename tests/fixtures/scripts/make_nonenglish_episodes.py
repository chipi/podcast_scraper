#!/usr/bin/env python3
"""Author the p10..p14 companion episodes (``_e02``, ``_e03``) from a per-language content table.

WHY A GENERATOR RATHER THAN TEN HAND-WRITTEN FILES. The five non-English feeds are deliberate
parallel counterparts of p01 — the same trail-building conversation in five languages, so a
translation can be compared across all of them against one known meaning. Hand-authoring ten files
in five languages drifts: one language gets a sponsor read with three ad categories and another
gets two, and the resulting coverage difference looks like a detector finding when it is just
inconsistent fixture prose. The table below keeps the five parallel BY CONSTRUCTION.

``_e01`` PREDATES THIS and stays hand-authored — regenerating it would churn its committed
``.rttm`` / ``.groundtruth.json`` / ``.vtt`` siblings and the app-validation corpus derived from
it, for no gain.

=== WHAT EACH EPISODE IS FOR ===

``_e02`` — the full-featured episode. Carries, in every language:

  * a host welcome that the host-speech-act row actually matches,
  * a FIRST-PERSON guest introduction ("conmigo está", "con me c'è", "bei mir ist") — the form
    every cue row originally missed,
  * a GUEST speech act (thanks-for-having-me). This is the gap that made all five languages score
    zero guest acts: ``_e01``'s guest says only a bare "Gracias, Lucía", which is not the act and
    could as easily be a host,
  * a sponsor read carrying one phrase from each of the three ad categories, so the >= 2-distinct
    threshold is crossed and the filter — not merely the patterns — is exercised,
  * a MENTIONED-ONLY DISTRACTOR whose name carries a language-specific particle. One person who
    is discussed and never speaks, doing double duty: the precision guard has something to catch,
    and name cleanup has a particle it must not eat.

``_e03`` — the negative case, and the reason it exists is that a corpus of ads-everywhere episodes
cannot distinguish a working ad filter from one that cuts indiscriminately. No sponsor read at all,
so the correct behaviour is to cut NOTHING.

=== THE SPEAKERS ARE NOT NEW, DELIBERATELY ===

Each language reuses ``_e01``'s host and guest. macOS ships ONE Italian voice and ONE German
voice, and ``transcripts_to_mp3.VOICE_PITCH_SHIFT`` separates host from guest by a measured pitch
shift because of it. A third speaking voice would have no synthesis left to take and would land on
top of one of the existing two, so the distractor is MENTIONED and never speaks — which is what a
distractor is anyway.

=== AFTER RUNNING THIS ===

    python tests/fixtures/scripts/make_groundtruth.py        # the .groundtruth.json sidecars
    python tests/fixtures/scripts/transcripts_to_vtt.py      # the .vtt the pipeline prefers
    python tests/fixtures/scripts/transcripts_to_mp3.py ...  # audio + .rttm

Then the RSS fixtures need items for the new episodes, and the app-validation corpus needs a
rebuild to pick them up.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[3]
TRANSCRIPTS = REPO / "tests" / "fixtures" / "transcripts" / "v3"


@dataclass(frozen=True)
class LanguageContent:
    """One language's phrases. Parallel across languages by position, not by translation quality.

    Every field is prose a reader of that language would write; where a field exists to trigger a
    detector, the comment on the field in this dataclass says which one. A phrase that stops
    matching after a vocabulary change should be FIXED HERE rather than by loosening the pattern —
    the fixture is the claim about what the language looks like.
    """

    code: str
    feed: str
    show: str
    host: str
    guest: str

    #: Someone discussed and never present. Their name carries a particle the junk list must not
    #: strip ("de la", "Di", "Du", "von", "da").
    distractor: str

    # --- episode framing -----------------------------------------------------
    e02_title: str
    e03_title: str
    #: Host welcome. Must match a row in ``_HOST_SPEECH_ACTS_BY_LANGUAGE``.
    welcome: str
    #: First-person guest introduction. Must match a leading cue; this is the form the plural-only
    #: rows missed.
    introduce_guest: str
    #: Guest's thanks-for-having-me. Must match a row in ``_GUEST_SPEECH_ACTS_BY_LANGUAGE``.
    guest_thanks: str
    #: Mentions the distractor with a mentioned-only marker, cue BEFORE name, so the precision
    #: guard catches them and no interview cue reaches them.
    mention_distractor: str

    # --- the sponsor read, one phrase per ad category ------------------------
    ad_disclosure: str
    ad_url: str
    ad_promo: str

    # --- conversational body -------------------------------------------------
    host_turns: Tuple[str, ...]
    guest_turns: Tuple[str, ...]
    #: Closing lines, so the episode does not end mid-sentence.
    host_outro: str
    guest_outro: str


CONTENT: Dict[str, LanguageContent] = {
    "es": LanguageContent(
        code="es",
        feed="p10",
        show="Sesiones de Sendero",
        host="Lucía Herrera",
        guest="Javier Benavides",
        distractor="Elena de la Fuente",
        e02_title="Drenaje, Pendiente y el Tercer Año",
        e03_title="Permisos y Paciencia",
        welcome="Bienvenidos de nuevo a Sesiones de Sendero.",
        introduce_guest="Hoy conmigo está Javier Benavides, constructor de senderos.",
        guest_thanks="Gracias por invitarme. Encantado de estar aquí.",
        mention_distractor=(
            "También hablamos sobre el informe de Elena de la Fuente, que no pudo acompañarnos "
            "hoy."
        ),
        ad_disclosure="Este episodio está patrocinado por Ramp.",
        ad_url="Entra a ramp punto com barra oferta para empezar.",
        ad_promo="Usa el código SENDERO y ahorra 20 por ciento, por tiempo limitado.",
        host_turns=(
            "Empecemos por el drenaje. ¿Por qué es la primera decisión?",
            "¿Y qué pasa cuando el presupuesto no alcanza para hacerlo bien?",
            "Cuéntame del error más común que ves en equipos nuevos.",
            "¿Cómo se ve ese fallo tres años después?",
            "Antes de cerrar, ¿qué puede probar alguien esta semana?",
        ),
        guest_turns=(
            "Porque el agua decide todo lo demás. Si el agua no sale del sendero, el sendero se "
            "convierte en el cauce.",
            "Entonces se recorta en la capa de base, y eso es justo donde no se debe recortar.",
            "Excavar con demasiada pendiente. Parece más rápido el primer día y cuesta el triple "
            "al tercer año.",
            "Se ve como una zanja. Y reconstruir cuesta más que haberlo hecho bien desde el "
            "principio.",
            "Caminar su sendero después de una lluvia fuerte. El agua le va a mostrar cada error.",
        ),
        host_outro="Javier Benavides, gracias por la conversación.",
        guest_outro="Gracias a ti, Lucía.",
    ),
    "it": LanguageContent(
        code="it",
        feed="p11",
        show="Sentieri d'Autore",
        host="Giulia Ferrara",
        guest="Marco Bellini",
        distractor="Chiara Di Stefano",
        e02_title="Drenaggio, Pendenza e il Terzo Anno",
        e03_title="Permessi e Pazienza",
        welcome="Bentornati a Sentieri d'Autore.",
        introduce_guest="Oggi con me c'è Marco Bellini, costruttore di sentieri.",
        guest_thanks="Grazie per l'invito. Felice di essere qui.",
        mention_distractor=(
            "Parliamo anche del rapporto su Chiara Di Stefano, che oggi non è con noi."
        ),
        ad_disclosure="Questo episodio è sponsorizzato da Ramp.",
        ad_url="Vai su ramp punto com barra offerta per iniziare.",
        ad_promo="Usa il codice SENTIERO e risparmia 20 per cento, per un tempo limitato.",
        host_turns=(
            "Partiamo dal drenaggio. Perché è la prima decisione?",
            "E quando il budget non basta per farlo bene?",
            "Raccontami l'errore più comune che vedi nelle squadre nuove.",
            "Come si presenta quel guasto dopo tre anni?",
            "Prima di chiudere, cosa può provare qualcuno questa settimana?",
        ),
        guest_turns=(
            "Perché l'acqua decide tutto il resto. Se l'acqua non esce dal sentiero, il sentiero "
            "diventa il canale.",
            "Allora si taglia sullo strato di base, ed è proprio lì che non si deve tagliare.",
            "Scavare con una pendenza troppo forte. Sembra più rapido il primo giorno e costa il "
            "triplo al terzo anno.",
            "Si presenta come un fosso. E ricostruire costa più che farlo bene dall'inizio.",
            "Camminare sul proprio sentiero dopo una pioggia forte. L'acqua mostra ogni errore.",
        ),
        host_outro="Marco Bellini, grazie per la conversazione.",
        guest_outro="Grazie a te, Giulia.",
    ),
    "fr": LanguageContent(
        code="fr",
        feed="p12",
        show="Sessions Sentier",
        host="Camille Dubois",
        guest="Julien Mercier",
        distractor="Marc Du Bois",
        e02_title="Drainage, Pente et la Troisième Année",
        e03_title="Permis et Patience",
        welcome="Bienvenue à nouveau dans Sessions Sentier.",
        introduce_guest="Aujourd'hui je reçois Julien Mercier, constructeur de sentiers.",
        guest_thanks="Merci de m'avoir invité. Ravi d'être là.",
        mention_distractor=(
            "Nous parlons aussi à propos de Marc Du Bois, qui n'a pas pu nous rejoindre "
            "aujourd'hui."
        ),
        ad_disclosure="Cet épisode est sponsorisé par Ramp.",
        ad_url="Allez sur ramp point com slash offre pour commencer.",
        ad_promo="Utilisez le code SENTIER et économisez 20 pour cent, pour une durée limitée.",
        host_turns=(
            "Commençons par le drainage. Pourquoi est-ce la première décision ?",
            "Et quand le budget ne suffit pas pour bien le faire ?",
            "Raconte-moi l'erreur la plus courante chez les équipes nouvelles.",
            "À quoi ressemble cette défaillance trois ans plus tard ?",
            "Avant de terminer, qu'est-ce qu'on peut essayer cette semaine ?",
        ),
        guest_turns=(
            "Parce que l'eau décide de tout le reste. Si l'eau ne sort pas du sentier, le sentier "
            "devient le canal.",
            "Alors on coupe dans la couche de base, et c'est précisément là qu'il ne faut pas "
            "couper.",
            "Creuser avec une pente trop forte. Ça paraît plus rapide le premier jour et ça coûte "
            "le triple la troisième année.",
            "Ça ressemble à un fossé. Et reconstruire coûte plus cher que de bien faire dès le "
            "départ.",
            "Marcher sur son propre sentier après une forte pluie. L'eau montre chaque erreur.",
        ),
        host_outro="Julien Mercier, merci pour cette conversation.",
        guest_outro="Merci à toi, Camille.",
    ),
    "de": LanguageContent(
        code="de",
        feed="p13",
        show="Pfadgespräche",
        host="Katrin Vogel",
        guest="Stefan Brandt",
        distractor="Clara von Hofmann",
        e02_title="Drainage, Gefälle und das Dritte Jahr",
        e03_title="Genehmigungen und Geduld",
        welcome="Willkommen zurück bei Pfadgespräche.",
        introduce_guest="Heute bei mir ist Stefan Brandt, Wegebauer.",
        guest_thanks="Danke für die Einladung. Freut mich, hier zu sein.",
        mention_distractor=(
            "Wir sprechen auch über Clara von Hofmann, die heute nicht dabei sein konnte."
        ),
        ad_disclosure="Werbung. Diese Folge wird von Ramp gesponsert.",
        ad_url="Geht auf ramp punkt com schrägstrich angebot, um zu starten.",
        ad_promo="Mit dem Code PFAD spart ihr 20 Prozent, nur für kurze Zeit.",
        host_turns=(
            "Fangen wir mit der Drainage an. Warum ist das die erste Entscheidung?",
            "Und wenn das Budget nicht reicht, um es richtig zu machen?",
            "Erzähl mir vom häufigsten Fehler bei neuen Teams.",
            "Wie sieht dieser Fehler nach drei Jahren aus?",
            "Bevor wir schließen, was kann man diese Woche ausprobieren?",
        ),
        guest_turns=(
            "Weil das Wasser alles andere entscheidet. Wenn das Wasser nicht vom Weg abläuft, "
            "wird der Weg zum Bach.",
            "Dann wird an der Tragschicht gespart, und genau dort darf man nicht sparen.",
            "Zu steil graben. Am ersten Tag sieht es schneller aus und im dritten Jahr kostet es "
            "das Dreifache.",
            "Es sieht wie ein Graben aus. Und der Wiederaufbau kostet mehr, als es von Anfang an "
            "richtig zu machen.",
            "Den eigenen Weg nach starkem Regen abgehen. Das Wasser zeigt jeden Fehler.",
        ),
        host_outro="Stefan Brandt, danke für das Gespräch.",
        guest_outro="Danke dir, Katrin.",
    ),
    "pt": LanguageContent(
        code="pt",
        feed="p14",
        show="Sessões de Trilha",
        host="Beatriz Antunes",
        guest="Rafael Vasconcelos",
        distractor="Tiago da Silva",
        e02_title="Drenagem, Inclinação e o Terceiro Ano",
        e03_title="Licenças e Paciência",
        welcome="Bem-vindos de volta a Sessões de Trilha.",
        introduce_guest="Hoje comigo está Rafael Vasconcelos, construtor de trilhas.",
        guest_thanks="Obrigado por me receber. É um prazer estar aqui.",
        mention_distractor=(
            "Falamos também sobre Tiago da Silva, que não pôde nos acompanhar hoje."
        ),
        ad_disclosure="Este episódio é patrocinado pela Ramp.",
        ad_url="Acesse ramp ponto com barra oferta para começar.",
        ad_promo="Use o código TRILHA e economize 20 por cento, por tempo limitado.",
        host_turns=(
            "Vamos começar pela drenagem. Por que é a primeira decisão?",
            "E quando o orçamento não dá para fazer bem feito?",
            "Conte-me o erro mais comum que você vê em equipes novas.",
            "Como essa falha aparece três anos depois?",
            "Antes de encerrar, o que alguém pode testar esta semana?",
        ),
        guest_turns=(
            "Porque a água decide todo o resto. Se a água não sai da trilha, a trilha se torna o "
            "canal.",
            "Então se corta na camada de base, e é exatamente ali que não se deve cortar.",
            "Escavar com inclinação forte demais. Parece mais rápido no primeiro dia e custa o "
            "triplo no terceiro ano.",
            "Aparece como uma vala. E reconstruir custa mais do que ter feito bem desde o começo.",
            "Caminhar na própria trilha depois de uma chuva forte. A água mostra cada erro.",
        ),
        host_outro="Rafael Vasconcelos, obrigada pela conversa.",
        guest_outro="Obrigado a você, Beatriz.",
    ),
}


def _header(c: LanguageContent, title: str, failure_modes: str) -> List[str]:
    """The ``#fixture-v3`` preamble ``transcripts_to_mp3`` and ``make_groundtruth`` both read.

    ``voice`` / ``host_voice`` carry the language tag rather than an accent, matching ``_e01``:
    the voice map is keyed by language, so the tag is what resolves the per-language voice.
    """
    tag = {"es": "es-ES", "it": "it-IT", "fr": "fr-FR", "de": "de-DE", "pt": "pt-BR"}[c.code]
    return [
        f"# {c.show} — Episodio" if c.code in ("es", "pt") else f"# {c.show} — Episodio",
        f"## {title}",
        f"Host: {c.host}",
        f"Guest: {c.guest}",
        f"#fixture-v3: failure_modes={failure_modes}",
        f"#fixture-v3: voice={tag} host_voice={tag}",
        "",
    ]


def _e02(c: LanguageContent) -> str:
    """The full-featured episode: every detector has something to find."""
    lines = _header(c, c.e02_title, "none")
    lines += [
        "[00:00]",
        f"{c.host}: {c.welcome} {c.introduce_guest}",
        f"{c.guest}: {c.guest_thanks}",
        # The sponsor read is one speaker's turn so the ad filter sees the three categories in one
        # window — the >= 2-distinct threshold is evaluated over a window, not the whole episode.
        f"{c.host}: {c.ad_disclosure} {c.ad_url} {c.ad_promo}",
        "",
        "[02:00]",
    ]
    for i, (h, g) in enumerate(zip(c.host_turns, c.guest_turns)):
        lines.append(f"{c.host}: {h}")
        lines.append(f"{c.guest}: {g}")
        # The distractor is mentioned LATE and far from the introduction, so no leading interview
        # cue can reach across to them — which would make the precision fixture test the opposite
        # of what it is for.
        if i == 2:
            lines += ["", "[12:00]", f"{c.host}: {c.mention_distractor}", ""]
    lines += [
        "",
        "[26:00]",
        f"{c.host}: {c.host_outro}",
        f"{c.guest}: {c.guest_outro}",
    ]
    return "\n".join(lines) + "\n"


def _e03(c: LanguageContent) -> str:
    """The negative case: NO sponsor read anywhere, so the correct cut is nothing at all."""
    lines = _header(c, c.e03_title, "none")
    lines += [
        "[00:00]",
        f"{c.host}: {c.welcome} {c.introduce_guest}",
        f"{c.guest}: {c.guest_thanks}",
        "",
        "[01:30]",
        f"{c.host}: {c.mention_distractor}",
        "",
        "[03:00]",
    ]
    # Reversed pairing so this is not a re-run of _e02 with the ads deleted — a fixture that
    # differs from its sibling only by a removal tests the removal and nothing else.
    for h, g in zip(reversed(c.host_turns), reversed(c.guest_turns)):
        lines.append(f"{c.host}: {h}")
        lines.append(f"{c.guest}: {g}")
    lines += [
        "",
        "[24:00]",
        f"{c.host}: {c.host_outro}",
        f"{c.guest}: {c.guest_outro}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify the committed transcripts match this table; write nothing",
    )
    args = ap.parse_args()

    drift: List[str] = []
    for c in CONTENT.values():
        for suffix, body in (("e02", _e02(c)), ("e03", _e03(c))):
            path = TRANSCRIPTS / f"{c.feed}_{suffix}.txt"
            if args.check:
                current = path.read_text(encoding="utf-8") if path.exists() else ""
                if current != body:
                    drift.append(str(path.relative_to(REPO)))
                continue
            path.write_text(body, encoding="utf-8")
            print(f"  wrote {path.relative_to(REPO)} ({len(body)} chars)")

    if args.check:
        if drift:
            print("transcripts differ from the content table (re-run without --check):")
            for d in drift:
                print(f"  {d}")
            return 1
        print("all generated non-English transcripts match the content table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
