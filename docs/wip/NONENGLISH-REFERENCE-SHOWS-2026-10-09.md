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
  `asr_publisher_ref_nonen_v1`, 12 episodes, committed to the private repo (`2692309`, not pushed); run outputs under its
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
| 3 | pipeline runs on `prod_dgx_full`, transcript kept after every step | yes (operator's yes for > 2 episodes) | done: baseline, ASR-only, translation retest, relabel (see Results) |
| 4 | scoring: ASR WER, per-step effect, naming, translation (model judge) | judge only | WER, per-step effect, naming scored; judge waits for its key |
| 5 | gap list and fixes | — | D1-D21 below; D6, D15, D16, D20 open for the operator |
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

## Results (2026-10-10)

Runs live in the eval repo's ignored `cache/dataset_runs/asr_publisher_ref_nonen_v1/`; each
`run.json` names the commit it ran. Those are pre-rebase SHAs, and each run's exported code is
kept under `cache/code/<sha>`. After the rebase onto main `3e0615172`: `d43559417` →
`59d3c29cc`, `22ea34a55` → `86ed99c84`, `283038972` → `0d0db93d6`.

| run | code | what |
| --- | --- | --- |
| `20261009T131355Z_full` | `d43559417` | baseline, full chain |
| `20261009T170818Z_asr_bare` | `d43559417` | ASR without gap recovery or punctuation repair, translation off |
| `20261009T174336Z_translate_fresh` | `22ea34a55` | El Hilo x2, translation from scratch, `translation_max_concurrency: 4` (D8 timing) |
| `20261009T175048Z_translate` | `283038972` | translate_only of the 4 Spanish episodes (D10, D10b) |
| `20261009T174920Z_relabel` | `283038972` | relabel_only of all 12 episodes (D1-D21 naming) |
| `20261009T230028Z_translate_fresh` | `36f8eefb0` | the 4 Spanish episodes translated from scratch (memory discarded, 4 in flight) |
| `20261009T233119Z_relabel` | `8ba7e8e1c` | relabel_only of El Hilo x2 (D22 + the "help" verb) |

**ASR.** Adjusted WER: numbers, long insertions and long deletions are split out (`wer_breakdown`),
normalisation unicode-v1. Baseline full chain: es 0.059, pt-BR 0.065, fr 0.045, de 0.071. Each
step's effect is measured as ASR alone (`asr_bare`) → full chain:

| episode | adjusted WER | long-deletion words |
| --- | --- | --- |
| El Hilo 09-25 | 0.0609 → 0.0591 | 53 → 51 |
| El Hilo 10-02 | 0.0183 → 0.0154 | 19 → 6 |
| Radio Ambulante 07-21 | 0.1136 → 0.1085 | 364 → 321 |
| Radio Ambulante 09-10 | 0.0515 → 0.0447 | 44 → 6 |
| Rádio Novelo 09-24 | 0.1060 → 0.0681 | 379 → 40 |
| Rádio Novelo 10-08 | 0.0792 → 0.0681 | 230 → 78 |
| RFI 10-07 | 0.0452 → 0.0473 | 0 |
| RFI 10-08 | 0.0427 → 0.0427 | 0 |
| Senado 10-07 | 0.0534 → 0.0534 | 10 → 10 |
| Senado 10-08 | 0.0230 → 0.0230 | 5 → 5 |
| SWR 10-07 | 0.0894 → 0.0839 | 68 → 64 |
| SWR 10-07 (ed6cd4) | 0.0688 → 0.0595 | 85 → 44 |

Gap recovery and punctuation repair lower adjusted WER on 8 of 12 episodes. They leave 3 unchanged
(RFI 10-08 and both Senado bulletins, which had 0-10 long-deleted words to begin with) and raise 1 (RFI 10-07, +0.002).
Most of the gain is recovered speech: Novelo 09-24 goes from 379 long-deleted words to 40.

**Translation.** Baseline: 3 of 4 Spanish episodes had no English (D10), so 9 of 12 episodes had
English. At `283038972`: 4 of 4 Spanish episodes are translated, 488 units, `units_failed: 0`, so 12
of 12 episodes have English. The retest re-sent only the 4 units that failed at the baseline; all 4
passed on the retry (`attempts: 2`, or 1 for a unit only the D10 guard fix released). The other 484
came from translation memory (`attempts: 0`). D10 only relaxes the guard, so units it accepted
before it still accepts. From scratch at `36f8eefb0` (memory discarded): the same 488 units,
`units_failed: 0`, all 4 episodes translated; 3 units were refused once as commentary and passed
on the numbered retry (`attempts: 2`). The earlier fresh run at `22ea34a55`, before D10b, lost 1
unit on each El Hilo episode. The translation itself is not judged yet: the judge waits for its key.

**Naming** (relabel at `283038972` against the baseline; fraction of the reference's speaking time
per outcome; two episodes per show):

| show | correct | misspelled | unnamed | wrong, after |
| --- | --- | --- | --- | --- |
| Radio Ambulante | 0 / 0 → 0.49 / 0.19 | — | 1.0 / 1.0 → 0.51 / 0.74 | 0 |
| El Hilo | 0.19 / 0.68 → unchanged | 0 → 0.28 / 0.30 (host, named as ASR heard him) | 0.81 / 0.32 → 0.53 / 0.03 | 0 |
| Rádio Novelo | 0.41 / 0.09 → 0.84 / 0.16 | — | 0.59 / 0.91 → 0.17 / 0.79 | 0 |

"Zentrum" (SWR) and "France Médias Monde" (RFI) are gone.

**Each figure above is ONE run of a step that is not deterministic, and an earlier version of this
section reported them as results ("no voice is named as a different person", El Hilo 10-02
"correct 0.97").** The LLM naming call runs at temperature 0, but the vLLM server's answers vary
between identical requests. Measured 2026-10-10 by repeating the relabel: five 12-episode relabels
at `d619fa099` and five at `664334560` give identical figures for every scored episode except El
Hilo 10-02. That episode, over 17 runs (the ten above, six El Hilo-only runs at `8ba7e8e1c` and
`d619fa099`, and the `8ba7e8e1c` run): correct 0.97 in 2, unnamed 0.53 in 10, and **wrong 0.50 in
5**: the guest's voice (Vanessa Torres) named "Silvia Viñas", a feed-stated host who is not on
the episode. D27 refuses the form where the model calls her a guest; the same error with the role
"host" remains (D29, open).

El Hilo 09-25 stays at its host "Elias Erbuda Sof", a three-token garbling the canonicaliser cannot
match.

**Why the rest stays unnamed (read on the transcripts, 2026-10-10):**

- Novelo 10-08's reporter (Vitor Hugo Brandalise, 5,288 words) and Radio Ambulante's (Mariano
  Pagella 2,436, Marco Avilés 1,642) are never introduced on air: they appear only in the credits
  ("Esta serie fue producida por…") and, for Novelo, the description's by-line ("Por Vitor Hugo <!-- codespell:ignore serie -->
  Brandalise."). Naming them needs a new rule binding a credited or by-lined name to the dominant
  non-host voice. Operator decision.
- SWR's interview clips are named only by the narration, verb first ("…, erklärt Oliver Huth.",
  then his clip); the metadata names none of them. The roster binds report-form names only to
  stated people (#876), so this needs binding UNSTATED names from narration attribution. Operator
  decision.
- El Hilo 09-25's "Elias Erbuda Sof" resembles the stated "Eliezer Budasoff" (ratio 0.75); the
  resemblance rule only ever refuses. Options: keep publishing the misspelling, or refuse it
  (unnamed). Operator decision.

**ASR, the steps' zero effect on fr and pt-BR:** RFI has no long deletion to recover (0 words).
Senado's long deletions are reporter sign-offs in the script ("Da Rádio Senado, Bruno Lourenço")
that the podcast audio does not hold: the diarizer hears no speech in the 5.2 s where one would be
(307.1-312.5 s), and the other runs straight into the next item. Gap recovery acts only on diarized
speech the transcript lacks, of which both bulletins have none (`untranscribed_speech_count: 0`).
The third sign-off (Raíssa Abreu, 10-08, checked 2026-10-10) is the same: the report's last
sentence runs 370.36-373.96 s (8 words in 3.6 s), the next item starts at 374.28 s, and the
diarizer's turns meet at 374.18 s, so there is no untranscribed speech where the sign-off would be.

**D15 re-measured at `283038972`.** With all four Spanish episodes now in English, the ad-free
English of each still holds the iHeart pre-roll, mid-rolls and post-roll: `chars_removed: 0` in all
four `.adfree.admap.json` files. At `7f7b8fb81` (D15, D30) the pre-roll and post-roll are cut in
all four (7,858 characters), with fragments left at the edges (a post-roll's first lines, which
precede its first tagged sentence; on El Hilo 10-02 a trailer's lines after the first cut); the
mid-rolls stay, since the region detector stays out of the middle by design.

## Defects found while onboarding these feeds

Each row: what the run showed, the cause, the fix and its test. "Fixed" means a failing test first
and the unit suites green; the effect on real episodes is measured by re-running naming at the fix
commit against the baseline run.

| # | found on | defect | cause | status |
| --- | --- | --- | --- | --- |
| D1 | El Hilo (es) | host's "Soy <Name>" read as no self-introduction; 0 of 16 voices | `hosts.extract_self_introduced_host` / `distinct_self_introductions` compiled the English row only; the roster held the language and did not pass it | fixed `b28cbc120` |
| D2 | El Hilo (es) | same | every non-English cue in `HOST_SELF_INTRO`, `HOST_BRANDED_INTRO`, `HOST_WITH_ME_INTRO` lowercase in a case-sensitive pattern: sentence-initial "Soy", "Je suis", "Ich bin" never matched | fixed `b28cbc120` |
| D3 | (review of D2) | es/pt "with me" rows could never match | rows read "con migo" / "com igo"; the words are "conmigo" / "comigo" | fixed `b28cbc120` |
| D4 | El Hilo (es) | both real guests rejected as "named but never introduced as speaking" | guest corroboration read English cues, on the premise that the feed description is in the analysis language; D-44 translates the transcript, not the feed | fixed `b28cbc120` |
| D5 | El Hilo (es) | an organisation ("My Cultura", from the author tag "My Cultura and iHeartPodcasts") detected as the host | the org filter drops the known network and keeps the co-publisher, which has no organisation marker | no effect on naming: dropped before the roster (`known_hosts: []` in the diagnostics); only the `DETECTED HOSTS` log line misleads |
| D7 | El Hilo (es) | "DEADLINE EXCEEDED: metadata generation (summary+GI+KG)" at 1200 s with summary, GI and KG all off | translation runs inside that observed block; the credit #2230 designed was never built | fixed: `credit_deadline`, the translation stage credits its wall time |
| D8 | El Hilo (es) | translation of a 41-minute episode took over 20 minutes (~14 s per unit) | units sent one at a time to a vLLM server that batches concurrent requests | built: `translation_max_concurrency` (default 1 = unchanged). MEASURED (El Hilo, 85 units): 4 in flight = 3.0 s/unit vs 15.5 sequential (5x). NOT byte-identical: 65 of 84 units matched, the rest differ in wording only (same sentence counts, lengths within 1.5%); the D8 commit's "changes wall time and nothing else" held for the pipeline, not for the server. Prod value: operator decision |
| D9 | (scoring El Hilo) | `Turn.to_dict` said `speaker_label` is anonymous and `speaker` resolved; on disk it is the reverse, as in `.segments.json` | docstring left from D-34 (naming after translation), reverted in #2234 | fixed: docstring states what the files hold |
| D10 | El Hilo (es) | BOTH episodes ended with no English, so no summary, GI or KG: 1 of 85 and 2 of 79 units refused as "model commentary" and the §5.3 gate withholds the whole set | (a) a FALSE refusal: the source said "según el contexto" and the faithful English "depending on the context" matched a marker; (b) two one-off commentary replies (on garbled ad audio) were never retried — re-sent, both came back clean twice | fixed: speech-like markers count only beside translator narration; a refused unit is retried once on both paths, then the whole-unit fallback |
| D10b | translation retest at `22ea34a55` | the D10 retry did not help: all three remaining failed units were refused on both attempts, so the three Spanish episodes still have no English | at temperature 0 the identical plain request gets the identical reply. Live translator: sent plain, all three (one garbled line of the "La Bestia" reggaeton promo) came back "Here's the translation:", twice followed by an invented story; sent numbered ("1. <sentence>"), a clean single line every time | fixed: the single-sentence retry sends the numbered form and reads one line (an unnumbered one-line answer is accepted) |
| D11 | El Hilo (es), probe over 714 descriptions | the second of two guests ("hablamos con A, de X, y con B") never introduced | the es leading-cue row had no coordinated form; English has "(?:and\|along) with" | fixed: `(?:y\|también) con`; measured: 34 new introductions, 24 real guests, 10 organisations (refused by the person check), 0 merely-mentioned people |
| D12 | SWR Das Wissen (de) | the narrator (95% of the episode's words) named "Zentrum" | baseline: the English row read German "Im Zentrum steht…" as "I'm Zentrum" (closed by D1); the German row then opened "Ich bin Polizist." → "Polizist" (a regression in `b28cbc120`): German capitalises every noun | fixed: one-word self-introductions refused in noun-capitalising languages (de); full names still read |
| D13 | RFI (fr) | "France Médias Monde" (the author tag, RFI's parent company) seated on the anchor's voice as host, in baseline and relabel | the author-tag organisation filter read the English markers only; the French row's `médias` was never consulted | fixed: English plus the feed's row (markers only refuse). The claim "every pipeline call passes the feed language" was FALSE: only `processing.py`'s calls did; eleven others did not, and Senado seated "Rádio Senado" as its host. Completed in D22 |
| D14 | Rádio Novelo (pt-BR), self-introduction probe over all labelled human turns | the host's "Eu sou a Branca Vianna" not read, in both episodes | the pt row allowed the article only before a role word, never before the name | fixed: optional `o`/`a` before the name; the probe read 11 introductions correctly and refused all 18 non-introductions |
| D17 | (code reading, after D1/D4) | a Spanish host's "Bienvenidos a El Hilo" or "nos acompaña hoy <name>" could never mark the host or name the guest | `roles_from_conversation` and `guests_introduced_by_the_host` read the English rows only, and the roster imported the English host-introduction pattern directly — the third instance of a per-language map built and then read in English | fixed: both take the language; every roster call passes it (AST delivery test); the English pin records the move |
| D18 | sweep of every `X_BY_LANGUAGE[TARGET_LANGUAGE]` alias (67) | the roster's guest/host speech-act checks (`_guests_by_their_own_words`, `_rescued_from_bleed`, `_name_host_voices`, `_presenter_voices_by_evidence`) and `performs_show_intro` read English on source-language text: "Esto es Radio Ambulante" was no show intro | same class as D1/D13/D17 | fixed: each reads its language's row; AST delivery test; English pin records the moves |
| D19 | Rádio Novelo (pt-BR) | "A Sous-titrage Société Radio-Canada" kept twice over the closing music | the invented-line filter drops a segment only when its WHOLE text is a known line; a stray leading "A" defeated it | fixed: one stray token of at most two characters at either end is allowed; a real sentence around the words is still speech |
| D20 | (measurement run) | `translate_api_base: null` in a run config does not switch translation off; the profile's translator stays | the CLI keeps only non-None values from a config file, so `null` cannot unset a key a profile sets — silently. `""` does (measured with `_build_config` on prod_dgx_full) | fixed `0252e8b97`: a key the file writes as `null` is carried as None, so it unsets the profile's value (option a: what an operator writing null means). Committed configs: the only explicit nulls are the profiles' own `transcription_coverage_failover_*`, which their profile layer already resolves to None; configs that exist only on prod were not checked |
| D21 | relabel at `22ea34a55`, Radio Ambulante (es) | relabel_only of a TRANSLATED episode ended with 0 named entries; the LLM proposed "Daniel Alarcón" from English text | relabel read `<base>.txt` — the English render under D-44 — named speakers on it with the source vocabulary, and the post-processing swap-back then moved that English aside, discarding the relabel's work. Every translated non-English episode is affected, and relabel is the repair path for naming fixes | fixed: relabel_only swaps the source back BEFORE reading (as translate_only already did); the memory is kept, so the English re-render costs no GPU |
| D22 | Senado (pt-BR) feed host; 585 chart feeds (es, mx, fr, de, br, it), eval repo `metadata_name_readers_probe_v1.py` | the metadata name readers read English only: a German cast "A und B" stayed one composite person (127 tags), a publisher with an article read as a person ("El País", "Le Monde", "Il Post"), an accented one-word name was refused ("Zoé", "Müller": an ASCII-only check); "Rádio Senado" / "RadioAgência Senado" seated as Senado's host | `split_author_names`, `names_the_show`, `looks_like_a_person_name`, `is_plausible_mononym`, the honorific list read the English row of per-language maps; eleven org-check calls never received the language (D13); markers glued into one token ("RadioAgência", "iHeartPodcasts") match no row | fixed: each takes the feed language, English plus that row; a name particle inside a name stays ("de la Garza", "von der Tann": the first model of the change refused them, so the probe's changed cases were read before building); glued markers refuse on non-English feeds (2,693 names: 11 refused, all organisations); AST test asserts every call passes the language, with four named English-by-design exceptions. The arc's 12 episodes: transcript-side naming unchanged; Senado's feed host goes from an organisation to none |
| D23 | translate_fresh run's events | every manifest and event of a pinned eval run recorded `"git_sha": "ced3bcf"`, the EVAL repo's commit | an export has no `.git`, so the pipeline's git probe walked up into the eval repo; and `corpus_version.resolve_git_commit_sha` (the `corpus_manifest.json` stamp) never read the image's `PODCAST_GIT_SHA`, so in the pipeline image (no git) it writes "unknown" | fixed: the stamp prefers `PODCAST_GIT_SHA` (one constant shared with the run manifest); the eval runner passes it. Prod's existing manifests were not inspected (no filesystem access) |
| D24 | (code reading after D22) | in the roster: the bleed rescue read guest replies in English only, so off English its first safeguard always passed; the show-mononym check dropped only an English article; name matching kept "Sra." / "Dott." / "Herr" as a given name; the greeting reclaim and turn scan used hosts' English greeting aliases | the same class as D17/D18 | fixed: each reads the feed's language (title stripping is one shared list, applied only while a given name and surname remain, with `m`/`general`/`colonel` kept out for English); English pin records the move. On the 12 episodes' 1,144 turns no guest act and no greeting row matches, so the effect there is nil; titled names do occur ("senador Paulo Paim") and are not measured without a relabel |
| D25 | El Hilo (es) | the feed states its hosts and the pipeline found none, so the host was named as the recogniser spelt him | no presenting verb in the Romance rows means "help" ("…te ayudan a entender") | fixed: `ayuda(n)`, `ajuda(m)`, `aiuta(no)`, `aide(nt)`; 585 chart feeds: one statement gained (El Hilo), none wrong; German `hilft/helfen` made two false hosts, so not added |
| D26 | the re-run alias sweep; 29,410 chart-feed item descriptions | `hosts_from_episode_description` read only the English host cue; `is_known_network` had no language ("ARD Kultur", "Deutschlandfunk Nova" seated as feed hosts); the one-word person check, `_first_name_presenters` and `guests_introduced_by_the_host` read English rows | the D17/D18 class, in readers the D18 sweep did not list | fixed: each reads English plus the feed's row; the cue's own row carries guards measured on the descriptions (a guest "with us", a manner after the cue, a capture inside a longer name, a genitive before it, the feed host's first name after the cue). Episode hosts: 0 items before, 178 after (35 feeds), about 6 wrong by reading — a news item, a guest where the feed states no host. Networks: 5 feeds lose a broadcaster seated as host. The 12 episodes: unchanged |
| D27 | El Hilo 10-02 relabels (es); prod snapshot 2026-10-09 | the LLM named a feed-stated HOST with the role "guest": "Silvia Viñas, guest" on the guest's voice (El Hilo, 42% of the episode), "Tom Holland, guest" on Dominic Sandbrook's voice (The Rest Is History, published on prod), "Elena Burger, guest" (a16z) | the verdict check refused invented and third-person names, not a self-contradicting one | fixed `664334560`: neither name nor role is kept. Prod: 2 of 507 recorded verdicts have this form, both wrong when read on the transcript |
| D28 | (writing the D27 test) | a voice saying "Soy Eliezer Budasoff" was refused the name it gave itself whenever the LLM named it | `_introduces_itself_as` read English cues only | fixed `664334560`: also the text language's `HOST_SELF_INTRO` row; English unchanged |
| D29 | El Hilo 10-02 relabels (es) | the same wrong name as D27 with the role "host" (in 2 of 5 runs at `664334560`) | the model's answer varies between identical requests at temperature 0; Silvia Viñas is uttered in the episode (the credits), so "never uttered" cannot refuse it, and "a stated guest left unbound while a second feed host is placed" fires on 7 prod episodes that are real two-host shows | OPEN: no deterministic signal found that separates it from correct answers |
| D30 | El Hilo (D15 measurement); prod snapshot | an ad cut that ends a screenplay turn took the next speaker's first sentence | the snap to a sentence end knew ". ", "! ", "? " and not the line break that ends a turn | fixed `7f7b8fb81`. Prod replay: 134 of 3,235 episodes move a boundary; read: 119 spans now kept, mostly sign-offs and openers, 3 of them ad sentences; 32 spans now cut, outro calls to action plus one closing line (Colombia Calling) |
| D15 | Radio Ambulante, El Hilo (es; English analysis base) | four iHeart network cross-promos (~750 source words: pre-roll, mid-rolls at 44% and 83%, post-roll) stay in the ad-free English that summary, GI and KG read: `chars_removed: 0` | the sponsor patterns do not match cross-promo wording ("Listen on iHeart Radio, Apple Podcasts, or wherever you listen to podcasts"); `detect_opening_crosspromo` handles only the OPENING and needs a voice that never recurs past 50% — the same promo as post-roll makes its voices "recur" (measured: SPEAKER_01/14 at 0-1% and 98-99%; detector returns None) | partly fixed `7f7b8fb81`: the iHeart network tags ("This is an iHeart Podcast", "Download the iHeart Radio app", "on iHeart Radio, Apple Podcasts") are English ad cues; pre-roll and post-roll cut on all four episodes, edge fragments left, mid-rolls kept (the detector stays out of the middle by design). Replayed on the prod snapshot of 2026-10-09 (3,235 stored episodes): the tags change none. Generic cues ("wherever you get your podcasts", "listen on Apple Podcasts") were measured and dropped: 106 episodes, 139k characters, including Hard Fork's and The Rest Is Science's own intros, cut whole from the start |
| D16 | Radio Ambulante (es), SWR (de) | narration lost under a "stretched word": "¿no?" timed 15.6 s over 11.5 s of speech at 20:31 hid 55 reference words; SWR's "einige" (3.9 s) hid 13 | by design (`stretched_words_over_speech`): recorded, never re-transcribed — 3 of 11 measured on 2026-10-08 hid speech and a splice would duplicate the word at an unknown position. This set: 2 of 4 hid speech (0 for "entonces" 3.1 s, "Europäer" 5.1 s) | OPEN, operator decision: re-transcribe the span alone and REPLACE the word only when the result holds the word plus clearly more speech — no duplication, no position guess |
| D6 | El Hilo, Radio Ambulante | feed items carry `<podcast:transcript>` (Omny SRT/VTT/text); prod would download those and never run our ASR | by design (feed transcript first) | measured 2026-10-10, same scorer (unicode-v1, `wer_breakdown`): feed transcript WER 0.054 / 0.019 (El Hilo), 0.119 / 0.066 (Radio Ambulante); ours 0.172 / 0.160 / 0.223 / 0.180, all of the difference ~750-785 words of ads in our audio that neither the reference nor the feed transcript holds. Adjusted: feed 0.052 / 0.018 / 0.113 / 0.065, ours 0.059 / 0.015 / 0.109 / 0.045 (ours better on 3 of 4). The figures this row gave before (0.044 / 0.069 / 0.115 / 0.149) were not from this scorer and are withdrawn. Feed-transcript-first or our ASR: operator decision |

**Discussion held for the end of the arc (operator, 2026-10-09):** the RFC-124 §5.3 all-or-nothing
translation gate. At the baseline one refused unit in ~80 withheld a Spanish episode's whole English
set (3 of 4 Spanish episodes, 0 of 8 others). Bring the retest's post-D10 failure rate and the
options: keep the gate (retries + guard fix only) or accept partial English with failed units
marked or re-tried by a fallback. From scratch at `36f8eefb0`: 0 of 488 Spanish units failed, 3
needed the retry, all 12 episodes have English. Four Spanish episodes are the whole sample.

**English-only readers left after D22-D26:** a re-run of the alias sweep (every
`X = X_BY_LANGUAGE[TARGET_LANGUAGE]` read inside a function where the language is in scope) finds
15 reads, each the English half of an "English plus the feed's row" union or the fallback for a
language with no row; I read each one. My first draft of this sentence said "none found" before
re-running the sweep, and the re-run found the five readers fixed in D26. Three English behaviours
were held back pending an English-corpus replay: accented mononyms and glued org markers on English
feeds ("José" refused, "OnePodcast" kept), and the title words `general`/`colonel` in name
matching. Replayed 2026-10-10 with `scripts/measure/roster_replay.py --repool` on the prod snapshot
of 2026-10-09 (2,454 English episodes; a no-op replay changed 0): each changes 0 published names.
The inputs are rare there, which is why: 2 accented one-word self-introductions in 3,236
transcripts, and of 89 author tags one ("BizNews") newly marked, already refused by the one-word
rule. Applied to English in `b5fc95281`. Kept English by design: `show_name_pattern` (the article is part of what a voice
says: "Esto es El Hilo"), the KG and GI readers of the English body, the ad patterns that build the
English ad-free base, and main's frozen migrations m0015/m0017/m0025.

Measurement tooling defects found on the way (eval repo, fixed there): the runner first served a
synthetic feed without descriptions (naming scored against less evidence than prod has); it then
served the publisher feed with its transcript links (ASR would not have run); and it ran the live
working tree, so a feed started after an edit ran different code. It now serves the publisher's
feed with only guid, enclosure and transcript links changed, and runs an export of a named commit.

**Caveats.** The measuring set is narrative journalism and news, not the long interviews of the
corpus; interview-style human references exist only in auto form (Lage der Nation, Logbuch:
Netzpolitik, Monde Numérique). Italian can be measured only through Podcast Italiano.
