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
| de | **SWR Das Wissen** | `swr.de/~podcast/swrkultur/programm/podcast-swr-das-wissen-102.xml` | human broadcast manuscript, PDF from the episode page; strip "Autorin:/O-Ton" labels, cover page, sources | "Organisierte Kriminalität in Europa (1/3)", 28:42 | 4,626 raw vs 3.7-4.9k | good: science, history, politics <!-- codespell:ignore programm --> |
| de | DW Langsam gesprochene Nachrichten (backup) | `rss.dw.com/xml/DKpodcast_lgn_de` | human read script, in the page's embedded `__APOLLO_STATE__` | 07.10.2026, 9:13 | 586 (64 wpm) | ok: news; coverage unconfirmed, very slow speech |
| pt-BR | **Rádio Novelo Apresenta** | `feeds.megaphone.fm/NPP6869883964` | human, edited, named speakers; PDF from radionovelo.com.br | ep 198 "Recado recebido", 70:58 | 10,313 vs 9.2-12.1k | ok: narrative journalism |
| pt-BR | **Rádio Senado – Jornal do Senado** | `www12.senado.leg.br/radio/1/voz-do-brasil/podcast.xml` | human broadcast script, "Transcrição" on the episode page (anchor lines in capitals); the 60-min "Íntegra" items have none | 08/10/2026, 10:01 | ~1,350 vs 1.3-1.7k | good: Brazilian politics <!-- codespell:ignore jornal --> |
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
programmes, Xataka, El Orden Mundial, Il Post, Il Sole 24 Ore, Café da Manhã, Naruhodo, Xadrez <!-- codespell:ignore ore -->
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

**Status (2026-10-09):** plan accepted by the operator; work started. Decisions:

- Translation quality is judged by a **stronger model as judge** (operator, 2026-10-09).
- The measuring code lives in the eval repo (`eval-data/`, private) as reusable tools, not one-offs
  (operator, 2026-10-09): `podcast_scraper_eval/publisher_transcripts.py` (one parser per
  publisher), `dataset_audio.py`, `wer_breakdown.py`, `translation_judge.py`;
  `scripts/eval/data/build_publisher_reference_set.py`,
  `scripts/eval/experiment/run_dataset_through_pipeline.py`, `scripts/eval/score/`
  (`asr_publisher_reference_wer_v1.py`, `translation_judge_v1.py`). Dataset
  `asr_publisher_ref_nonen_v1`, 12 episodes, for the private repo (not yet committed); run outputs under its
  ignored `cache/dataset_runs/`.
- Audio over GitHub's 100 MiB file limit (one Rádio Novelo episode) is not committed: the dataset
  records URL, sha256 and size and the runner re-fetches and verifies it (operator, 2026-10-09).
- Everything found while onboarding these feeds is fixed in this arc, on this branch, whether or
  not it is about translation (operator, 2026-10-09) — see "Defects found" below.
- Reference transcripts are stored only in the private eval repo's cache; never committed to the
  public repo or quoted in public.

| step | what | DGX | status |
| --- | --- | --- | --- |
| 1 | reference harness: fetch, clean, coverage-check the human transcripts | no | done |
| 2 | episode set, ~12 episodes, references verified to cover their audio | no | done, 12 |
| 3 | pipeline runs on `prod_dgx_full`, transcript kept after every step | yes (operator's yes for > 2 episodes) | running (baseline `d43559417`) |
| 4 | scoring: ASR WER, per-step effect, naming, translation (model judge) | judge only | WER, naming and judge built; judge waits for a key |
| 5 | gap list and fixes | — | — |
| 6 | readiness call per language | — | operator |

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
     English texts); then judge the translation itself with a stronger model as judge
     (operator's decision).
5. **Gap list and fixes.** Rank the gaps by what they cost per language; fix in this arc what
   is a code fix, record what is a model choice (#2251) or a vocabulary (#2255-#2259).
6. **Readiness call per language.** Which languages could be enabled on prod, with what known
   limits. The operator decides.

## Defects found while onboarding these feeds

Each row: what the run showed, the cause, the fix and its test. "Fixed" means a failing test first
and the unit suites green; the effect on real episodes is measured by re-running naming at the fix
commit against the baseline run.

| # | found on | defect | cause | status |
| --- | --- | --- | --- | --- |
| D1 | El Hilo (es) | host's "Soy <Name>" read as no self-introduction; 0 of 16 voices | `hosts.extract_self_introduced_host` / `distinct_self_introductions` compiled the English row only; the roster held the language and did not pass it | fixed `9cf7da0dc` |
| D2 | El Hilo (es) | same | every non-English cue in `HOST_SELF_INTRO`, `HOST_BRANDED_INTRO`, `HOST_WITH_ME_INTRO` lowercase in a case-sensitive pattern: sentence-initial "Soy", "Je suis", "Ich bin" never matched | fixed `9cf7da0dc` |
| D3 | (review of D2) | es/pt "with me" rows could never match | rows read "con migo" / "com igo"; the words are "conmigo" / "comigo" | fixed `9cf7da0dc` |
| D4 | El Hilo (es) | both real guests rejected as "named but never introduced as speaking" | guest corroboration read English cues, on the premise that the feed description is in the analysis language; D-44 translates the transcript, not the feed | fixed `9cf7da0dc` |
| D5 | El Hilo (es) | an organisation ("My Cultura", from the author tag "My Cultura and iHeartPodcasts") detected as the host | the org filter drops the known network and keeps the co-publisher, which has no organisation marker | no effect on naming: dropped before the roster (`known_hosts: []` in the diagnostics); only the `DETECTED HOSTS` log line misleads |
| D7 | El Hilo (es) | "DEADLINE EXCEEDED: metadata generation (summary+GI+KG)" at 1200 s with summary, GI and KG all off | translation runs inside that observed block; the credit #2230 designed was never built | fixed: `credit_deadline`, the translation stage credits its wall time |
| D8 | El Hilo (es) | translation of a 41-minute episode took over 20 minutes (~14 s per unit) | units sent one at a time to a vLLM server that batches concurrent requests | built: `translation_max_concurrency` (default 1 = unchanged); the prod value waits for a DGX measurement |
| D9 | (scoring El Hilo) | `Turn.to_dict` said `speaker_label` is anonymous and `speaker` resolved; on disk it is the reverse, as in `.segments.json` | docstring left from D-34 (naming after translation), reverted in #2234 | fixed: docstring states what the files hold |
| D10 | El Hilo (es) | BOTH episodes ended with no English, so no summary, GI or KG: 1 of 85 and 2 of 79 units refused as "model commentary" and the §5.3 gate withholds the whole set | (a) a FALSE refusal: the source said "según el contexto" and the faithful English "depending on the context" matched a marker; (b) two one-off commentary replies (on garbled ad audio) were never retried — re-sent, both came back clean twice | fixed: speech-like markers count only beside translator narration; a refused unit is retried once on both paths, then the whole-unit fallback |
| D11 | El Hilo (es), probe over 714 descriptions | the second of two guests ("hablamos con A, de X, y con B") never introduced | the es leading-cue row had no coordinated form; English has "(?:and|along) with" | fixed: `(?:y\|también) con`; measured: 34 new introductions, 24 real guests, 10 organisations (refused by the person check), 0 merely-mentioned people |
| D6 | El Hilo, Radio Ambulante | feed items carry `<podcast:transcript>` (Omny SRT/VTT/text); prod would download those and never run our ASR | by design (feed transcript first) | open. They are machine transcripts ("Speaker N" labels, timestamps); WER against the human reference: El Hilo 0.044 / 0.069, Radio Ambulante 0.115 / 0.149. Compare with our ASR on the same episodes when the baseline run lands |

Measurement tooling defects found on the way (eval repo, fixed there): the runner first served a
synthetic feed without descriptions (naming scored against less evidence than prod has); it then
served the publisher feed with its transcript links (ASR would not have run); and it ran the live
working tree, so a feed started after an edit ran different code. It now serves the publisher's
feed with only guid, enclosure and transcript links changed, and runs an export of a named commit.

**Caveats.** The measuring set is narrative journalism and news, not the long interviews of the
corpus; interview-style human references exist only in auto form (Lage der Nation, Logbuch:
Netzpolitik, Monde Numérique). Italian can be measured only through Podcast Italiano.
