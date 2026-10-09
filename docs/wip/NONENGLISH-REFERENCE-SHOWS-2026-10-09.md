# Non-English shows with publisher transcripts, and the next arc (2026-10-09)

The #2187 checkpoint (PR #2301) shipped the pipeline hardening but measured no non-English
accuracy: the six V.6b feeds ran end to end, nothing scored their words (RFC-124 Q10). This note
lists shows whose publishers post a transcript, so our output can be scored against it, and the
plan for the arc that uses them.

## How the shows were found and checked

Searched through the iTunes search API and by grepping feeds for `<podcast:transcript>` (the
Podcast Index API needs a key we do not have, so the list is not exhaustive). For each show one
episode was checked: the feed fetches and has audio, the same episode's transcript is free to
read, and its word count fits the episode length (130-170 words per minute). Human vs
auto-generated was judged from the text (named speakers vs "Speaker N", typical ASR errors) and
the publisher's own wording where it says. **Not checked**: any transcript against its audio,
more than one episode per show.

## Human transcripts — the measuring set

| lang | show | feed | transcript | sample episode | words vs expected | fit |
| --- | --- | --- | --- | --- | --- | --- |
| es | **El Hilo** (Radio Ambulante Studios) | `omnycontent.com/.../e4eb1040-260e-4556-a9c7-b1ea01352a67/podcast.rss` | human, edited, named speakers; inline "Transcripción" on elhilo.audio (page also has an English version, AI-assisted — cut at "Transcript: The following") | "Marco Rubio en Los Andes", 47:51 | 7,759 vs 6.2-8.1k | good: LatAm geopolitics |
| es | **Radio Ambulante** (NPR) | `feeds.npr.org/510315/podcast.xml` | human, near-verbatim ("o sea", false starts kept); `radioambulante.org/transcripcion/<slug>-transcripcion` (the RSS tag points at Omny auto text on 47 of 369 items — do not use that) | "Bogotá pintada", 60:25 | 11,058 (incl. labels, a membership ad) vs 7.9-10.3k | ok: LatAm narrative journalism |
| fr | **RFI Journal en français facile** | `rfi.fr/fr/podcasts/journal-français-facile/podcast` | human; inline on francaisfacile.rfi.fr (PDF too); interview clips keep hesitations | 07/10/2026, 10:00 | 1,535 vs 1.3-1.7k | ok: international news, slow and clear |
| de | **SWR Das Wissen** | `swr.de/~podcast/swrkultur/programm/podcast-swr-das-wissen-102.xml` | human broadcast manuscript, PDF from the episode page; strip "Autorin:/O-Ton" labels, cover page, sources | "Organisierte Kriminalität in Europa (1/3)", 28:42 | 4,626 raw vs 3.7-4.9k | good: science, history, politics |
| de | DW Langsam gesprochene Nachrichten (backup) | `rss.dw.com/xml/DKpodcast_lgn_de` | human read script, in the page's embedded `__APOLLO_STATE__` | 07.10.2026, 9:13 | 586 (64 wpm) | ok: news; coverage unconfirmed, very slow speech |
| pt-BR | **Rádio Novelo Apresenta** | `feeds.megaphone.fm/NPP6869883964` | human, edited, named speakers; PDF from radionovelo.com.br | ep 198 "Recado recebido", 70:58 | 10,313 vs 9.2-12.1k | ok: narrative journalism |
| pt-BR | **Rádio Senado – Jornal do Senado** | `www12.senado.leg.br/radio/1/voz-do-brasil/podcast.xml` | human broadcast script, "Transcrição" on the episode page (anchor lines in capitals); the 60-min "Íntegra" items have none | 08/10/2026, 10:01 | ~1,350 vs 1.3-1.7k | good: Brazilian politics |
| it | Podcast Italiano (Davide Gemello) | `rss.buzzsprout.com/2413795.rss` | human prepared monologue; the page asks for a free login but the full text is in the HTML | "6 differenze tra italiano del nord e del sud", 12:19 | ~1,630 vs 1.6-2.1k | ok: language and culture |

**Italian has no clean, ungated human transcript.** Podcast Italiano works only if the login does
not block an automated fetch.

## Auto-generated transcripts — disagreement only, not accuracy

Another ASR system's output: a mismatch does not say which side is wrong. Good topic fit:

- fr: Monde Numérique (AI, tech; Audiomeans, 464 of 500 items).
- de: Lage der Nation (politics; 105 of 500), Logbuch:Netzpolitik (tech policy; named speakers,
  533 of 564), heise KI-Update (AI; visible errors).
- it: Digitalia (tech news; 45 of 45), Radio Radicale (politics; the station's FAQ says its
  transcripts contain many errors).
- pt-BR: Tecnocast (tech; poor ASR).

## Rejected

Paid or gated: Easy German, Easy Italian, Hoy Hablamos, L'Italiano Vero (Patreon); e-mail sign-up
for Teacher Stefano, Speaking Brazilian, Français avec Pierre. No transcripts: Radio France
programmes, Xataka, El Orden Mundial, Il Post, Il Sole 24 Ore, Café da Manhã, Naruhodo, Xadrez
Verbal, Les Echos, Thinkerview, Handelsblatt Today, and others. Unreachable from here: InnerFrench
(Cloudflare 403), Français Authentique (406), Binge Audio. Stale: Duolingo podcasts (bilingual
narration anyway), Portuguese With Carla, Italy Made Easy.

## What we are not yet happy with (input to the arc)

1. Non-English ASR accuracy is unmeasured (RFC-124 Q10).
2. Translation quality is unmeasured; the model pick is provisional (#2251).
3. Speaker naming: es hosts on first names or nicknames are swapped, de near-identical names
   merge, it and pt-PT guests unnamed, pt-BR introductions on one voice
   (`2187-NAMING-HANDOVER-2026-10-08.md`).
4. Non-English detector vocabularies unmeasured (#2255-#2259); punctuation-repair prompts exist
   only for en, es, it, fr, de, pt.
5. Gap recovery can bring back an ad in another language (the V.6b French feed's Dutch ad).
6. Summaries, insights and the KG run on the English translation; the loss against native English
   is unmeasured, and quotes are translations with no "translated" marker yet (D-36).

## The next arc — plan

**Goal:** know, per language, how good our transcript, speaker names and translation are against a
human reference, and which gaps block enabling that language on prod.

1. **Reference harness (no DGX).** Per show: fetch the transcript, strip what was not spoken
   (labels, cover pages, source lists, ads, the English half on El Hilo), normalise. Reuse the eval
   repo's WER scorer (`eval-data/scripts/eval/moss/score_transcription_wer.py` or
   `score/whisper_accent_wer_v1.py`) rather than writing one; keep the trigram bar/floor from the
   English check as the second view, because edited transcripts drop filler.
2. **Episode set.** Two episodes per human-transcript show (es x2 shows, fr, de, pt-BR x2): about
   12 episodes, 10-70 minutes. Check each reference covers its episode before spending DGX on it.
3. **Run as prod would (DGX — needs the operator's yes; ~12 episodes is over the 1-2 limit).**
   `prod_dgx_full`, feed-declared language, full chain: ASR, invented lines, gap recovery,
   punctuation repair, diarization, naming, translation. Keep the transcript after EACH step so
   every step's effect can be scored on its own. Start with one episode to measure time per
   episode, then the rest.
4. **Score.**
   - ASR: WER per episode and language against the reference, and against the English baseline
     (80k set raw WER 8.4-10.6%); per step: does punctuation repair / gap recovery move WER here as
     it did in English; error classes (names, numbers, code-switching, music beds).
   - Speaker naming: El Hilo, Radio Ambulante, Novelo and SWR name their speakers in the text —
     score who-said-what, the defects in item 3 above.
   - Translation: no English reference exists for most of these. First measure what ASR errors
     cost the translation (translate the human reference and our transcript, compare the two
     English texts); then decide how to judge translation itself (human spot-check of a sample or
     a stronger model as judge) — an open question for the operator.
5. **Gap list and fixes.** Rank the gaps by what they cost per language; fix in this arc what
   is a code fix, record what is a model choice (#2251) or a vocabulary (#2255-#2259).
6. **Readiness call per language.** Which languages could be enabled on prod, with what known
   limits. The operator decides.

**Caveats.** The measuring set is narrative journalism and news, not the long interviews of the
corpus; interview-style human references exist only in auto form (Lage der Nation, Logbuch:
Netzpolitik, Monde Numérique). Italian can be measured only through Podcast Italiano.
