# Onboarding a new language: measurement runbook

## Why this exists

The non-English arc of 2026-10-09/10 (es, fr, de, pt-BR; `docs/wip/NONENGLISH-REFERENCE-SHOWS-2026-10-09.md`)
measured ASR, speaker naming, ad cutting and translation against publishers' own transcripts, and
found 30 defects (D1-D30) on the way. Most of them were not language bugs but measurement traps:
a result read from one sample, a scorer that broke accents, a replay that changed nothing because
nothing on prod could trigger it. This page is the order of work and the traps, so the next
language starts where this one ended.

The measuring tools live in the private eval repo (`chipi/podcast-scraper-eval-data`; ask the
operator for access). Paths below are relative to its root. Nothing from a publisher's transcript
is quoted in this repo, in a commit message, or in an issue.

## Quick reference

| step | tool | gate |
| --- | --- | --- |
| 1. reference set | `scripts/eval/data/build_publisher_reference_set.py` | 2 episodes per show, human transcripts, at least one show with speaker labels |
| 2. vocabulary probe | `scripts/eval/probe/fetch_chart_feeds_v1.py`, `metadata_name_readers_probe_v1.py` | every changed case read before a row is wired |
| 3. baseline run | `scripts/eval/experiment/run_dataset_through_pipeline.py --variant full --code-ref <sha>` | every feed exit 0 |
| 4. ASR | `scripts/eval/score/asr_publisher_reference_wer_v1.py` | adjusted WER per episode; long deletions read |
| 5. naming | `--variant relabel`, `scripts/eval/score/naming_publisher_reference_v1.py --compare` | `wrong` 0 in EVERY repeated run |
| 6. translation | `--variant translate*`, `scripts/eval/score/translation_judge_v1.py` | `units_failed` 0; judged; low units read |
| 7. English side effects | `scripts/measure/roster_replay.py`, an ad-cut replay | no-op replay 0; every change read |

## Work in tiers, gated per language

ASR first, then translation, then naming, then prod (operator, 2026-10-10). Each layer reads the
one below it: translation translates the ASR text, naming reads the source transcript's
self-introductions. Working them at once makes a moved number ambiguous. Each tier has its own
gate per language (es, pt-BR, fr, de, it are judged separately; one can pass while another is
stuck), and each gate is set against what the English pipeline achieves on the same measure, not
against perfection.

**A tier does not close until the per-language lists it uses are checked** for every onboarded
language. A hand-written list cannot be shown optimal; it can be shown covered, sourced and
measured:

1. covered: each onboarded language has entries, or the list says it has none on purpose;
2. sourced: each entry was seen on our real data or kept by a probe whose changed cases were all
   read, never translated from the English row;
3. measured: adding or removing entries is scored against the references like any switch.

| tier | lists |
| --- | --- |
| ASR | `transcription/invented_lines.py` (`INVENTED_LINES`), `transcription/gap_recovery.py` (`_FILLERS`), `transcription/punctuation.py` (prompts), `providers/ml/diarization/ad_signatures.py` (stopwords); in the eval repo `SPOKEN_FORMS` and the bracketed-note rule |
| naming | `speaker_detectors/naming_vocabulary.py` (`X_BY_LANGUAGE`), `speaker_detectors/hosts.py`, `speaker_detectors/constants.py` (step 2 below is the method) |
| after prod | `gi/filters.py`, `search/query_language.py` (shape insights and search, not the transcript) |

**ASR lists, checked 2026-10-11** on the 24 reference episodes (en, es, pt, fr, de), and for
Italian on 2026-10-13 on its 6 episodes of the 2026-10-12 re-run (the Italian part of each row):

| list | covered | sourced and measured | change |
| --- | --- | --- | --- |
| `INVENTED_LINES` | en es pt fr de it | entries seen on our runs or standard subtitle credits; 0 hits in 303 gap decodes | none; it: 0 hits and 0 lines removed in the 6 Italian episodes; "Grazie a tutti." is not added (the chair says it to close a sitting; the invented copies sat in digital silence, now cut) |
| `_FILLERS` | hesitations for en de fr pt; es and pt discourse words ("né", "o sea", "pues") left out on purpose | Whisper writes no hesitations in any of the 24; adding the es/pt discourse words to the silent-gap floor changed nothing (-100 errors either way) | none; it: Whisper wrote "eh" 3 and "ah" 4 (both listed), "ehm" 0; discourse words ("cioè" 24, "allora" 29, "appunto" 13) are words and left out, as for es and pt; no Italian silent-gap recovery was fillers only |
| punctuation prompts | en es pt fr de it | window repairs ran on es and pt; the whole-transcript retry had only the English prompt, so RFI 10-09 (French, 2.0 sentence ends per 1,000 words) was never retried | the retry now uses the language's prompt; on the 2026-10-12 re-run it fired on 2 of 30 (RFI 10-09: 2.0 -> 70.0, +16 adjusted errors from two names); it: the retry fired on Ci vediamo in Piazza (0.0 -> 73.9) |
| ad-signature function words | en de es pt fr it nl | unique-per-language rule measured on prod 2026-10-02 | none |
| `SPOKEN_FORMS` (eval repo) | pt de en it | each entry seen as a substitution in the 24 | added de "hab'/habe", "sag/sage", en "ok/okay"; removed never-seen es "pa", en "gonna"/"wanna", pt "tavam"/"cê"/"cês"; it: euphonic d both ways (a/ad 10+4, e/ed 6), con il/col 6, apocope maggiore/maggior 3 and avere/aver 2, "è" typed "e'" 2: Camera's main-voice adjusted 0.0896 / 0.0808 / 0.0919 -> 0.0883 / 0.0789 / 0.0911, ON-Air unchanged |
| bracketed-note rule (eval repo) | all | notes in every reference set | added |

## Where each decision is recorded

| decision | file | field |
| --- | --- | --- |
| ASR model, coverage, language, gap and stretched-word recovery, punctuation | `<ep>.asr.json` | `speech_recovery`, `stretched_word_recovery`, `stretched_words`, `punctuation` |
| per-stage counts and flags | `<ep>.manifest.json` | `stages.asr.metrics`, `stages.translation.metrics`, `quality_flags` |
| every translation unit, its attempts and why any was refused | `<ep>.translation.json` | `units[].attempts`, `units[].refusals`, `units[].model`, `prompt` |
| ad cuts and the text cut | `<ep>.adfree.admap.json` | `excised_ranges`, `excised_texts` |
| every naming decision and the LLM's raw answer | `<ep>.speakers.diagnostics.json` | `decision_trace` |

## 1. Pick reference shows

- **Human transcripts only.** Machine transcripts (Omny SRT/VTT, "Speaker N") measure our ASR
  against someone else's ASR. Keep them as a comparison (D6), not as the reference.
- **At least one show whose transcript labels speakers.** Without labels, naming cannot be
  scored at all; in this arc RFI and Senado could only be scored for ASR.
- **Two episodes per show.** One episode makes a show-level conclusion out of one recording.
- **Check what the transcript IS.** Senado's transcripts are the broadcast script: reporter
  sign-offs ("Da Rádio Senado, <name>") are written down but not in the podcast audio. They show
  up as long deletions that no ASR change can recover. Check a long deletion against the
  diarization before treating it as lost speech.
- **Mix genres.** Narrative journalism and news are not the long interviews the corpus is made
  of. Interview-style human references were found only in auto form for de and fr.

## 2. Learn the language's vocabulary on real feeds first

Per-language rows live in `src/podcast_scraper/speaker_detectors/naming_vocabulary.py`
(`X_BY_LANGUAGE`). A translated row is a guess; a row activated without measurement regressed
real names twice (D12, D22's first model).

- Fetch real feeds: `fetch_chart_feeds_v1.py es mx fr de br it` (Apple top charts, about 100
  each; cached under `cache/probe_feeds/`).
- Probe the change before wiring it: `metadata_name_readers_probe_v1.py` prints every verdict
  that changes with the language added. Read every changed case.
- **The rule (D13):** read English PLUS the feed's row. Feeds in Spanish carry English metadata
  ("My Cultura and iHeartPodcasts"). Lists that refuse may only widen.
- **`vocabulary_row` semantics:** `None` means English; a language with no row reads nothing,
  never English silently.
- **Every reader must receive the language.** The arc claimed "every call passes the language"
  and 11 calls did not (D13 → D22). The AST tests
  `tests/unit/podcast_scraper/test_name_readers_receive_the_feed_language.py` and
  `test_the_feed_language_reaches_the_roster.py` enforce it; add a new reader to them. Re-run
  the alias sweep (every `X = X_BY_LANGUAGE[TARGET_LANGUAGE]` inside a function with the language
  in scope) before writing "none left": the re-run found five more (D26).
- English behaviour is pinned (`tests/unit/podcast_scraper/test_english_patterns_are_mains.py`,
  `tests/fixtures/naming/english_patterns.json`). Changing it needs step 7.

## 3. Run the pipeline as prod would

`run_dataset_through_pipeline.py DATASET --variant V --code-ref <sha>`:

- `--code-ref` runs an export of that commit, never the working tree (a feed started after an
  edit once ran different code). `run.json` records the commit (`PODCAST_GIT_SHA`).
- The publisher's own feed is served (metadata intact, guids/enclosures/transcript links
  rewritten). A synthetic feed scores naming against less evidence than prod has.
- Variants: `full` (prod chain), `asr_bare` (no gap recovery, no punctuation repair, no
  translation), `relabel` (naming only, on a copy of `--from-run`), `translate` (memory kept),
  `translate_fresh` (memory discarded, 4 in flight), `translate_fresh_serial` (1 at a time).
- The `translate*` variants re-translate EVERY episode of the copied feed; `--episodes` does not
  narrow a copy.
- GPU-host runs: 1-2 episodes; more needs approval. Before spending a GPU-host run on a
  phenomenon, check which episode carries it: the arc ran the stretched-word test on the wrong
  Radio Ambulante episode once (D16).
- A config `null` now unsets a profile value (D20); before `df5535a38` it was silently ignored.

## 4. ASR

- `asr_publisher_reference_wer_v1.py RUN_DIR`. Normalisation `unicode-v1`: `ascii-v1` breaks every
  accented word.
- Read **adjusted** WER: numbers, spoken contractions, long insertions and long deletions are
  split out. Raw WER on iHeart shows is inflated by ~750-785 words of ads the reference does not
  contain. A spoken contraction ("pra" / "para a", "tá" / "está", "gonna" / "going to") is what
  Whisper writes for what was said, and the publisher writes the standard form; both are correct,
  so a substitution span whose two sides match after contraction (`SPOKEN_FORMS`, per language,
  eval repo `fe06311`) leaves the adjusted figure. A new language adds its contractions there
  before gate B is read.
- The publisher's bracketed notes are not speech and are dropped from the reference before
  scoring (`reference_words`, eval repo `842606a`): "[MÚSICA: …]", "[ri]", "(Laughter)",
  "(Speaking Arabic)", "(ph)" after a name spelled by ear, SWR's "(Atmo abblenden)", "[de]" words
  an editor added, "[Gabriela Llanos]" an inline speaker change. Every reference set had some;
  Radio Ambulante 07-14 lost 30 main-voice errors to them.
- Measure each step's effect as `asr_bare` → `full` per episode (gap recovery, punctuation
  repair, stretched-word recovery). Whisper's sampling fallback makes two decodes differ slightly
  (ADR-161), so small deltas are noise.
- Read every long deletion: lost speech, an edited transcript, or a script line not in the audio.
- Stretched words (`stretched_words` in `.asr.json`): one word timed seconds long over diarized
  speech can hide a sentence (D16). The span is re-transcribed alone, aligned against the
  episode's own words around it, and the word is replaced by the clip words the episode does not
  already have, when there are at least 3 that are not fillers (`stretched_word_recovery` in
  `.asr.json`). The first version, which demanded the clip hear the stretched word itself,
  declined every real case: a stretched word is often a word Whisper invented ("¿no?") over speech
  it skipped. Scored against the references before shipping: adjusted errors -33, -3, -5, 0.
- Silent gaps (`asr_silent_gap_recovery` in `.asr.json`). The diarizer misses a voice over a music
  bed, so a gap with no speaker turn was never retried: SWR's intro ("Die Kriminellen. Von Beate
  Kroll."), RFI's ("Le journal en français facile. Adrien Delgrange."), El Hilo's date line, Rádio
  Novelo's outro, a Throughline narrator line, all decoded word for word when cut out alone. Every
  transcript gap of 3 s+ not already retried is now re-transcribed, and its text kept only when it
  holds 3+ words that are not fillers (the D16 rule; `transcription_recover_silent_gaps`). Measured
  2026-10-11 on the 24 reference episodes by cutting every gap and decoding it with the pipeline's
  clip transcriber, which on `prod_dgx_full` is the self-hosted Whisper server: main-voice adjusted errors
  2861 -> 2761 (2780 keeping every gap); 8 episodes better (SWR 10-07 -23, Radio Ambulante 07-21
  -26), 15 unchanged including 5 of 6 English, Rádio Novelo 09-24 +3. The 1-2 word decodes of
  music gaps ("for", "Hmm.", "Episodio") were what made episodes worse without the floor. Of the
  26 gaps kept where the reference has nothing, about 18 read as real speech the publisher did not
  transcribe (foreign-language tape, songs, promos) and about 4 as invented ("That's it for this."
  over 18.6 s of music, "Bye! Bye! Bye! Bye!"); not listened to. Checked on the full chain on the
  GPU host, main-voice adjusted: SWR 10-07 0.0454 -> 0.0402 (the estimate exactly), Throughline 10-08
  0.0389 -> 0.0379 (estimated 0.0362; run before the fix below). The first SWR run exposed a bug:
  a gap the diarized retry had partly covered (70.6-74.2 s of 74.0-81.2 s) was skipped whole, and
  the intro in its uncovered part was lost; retried stretches are now subtracted from a gap.
  Two invented recoveries on Throughline packed words faster than anyone speaks ("That's it for
  this." in 0.26 s, 15.4 words/s; "Stay warm." 16.7 and 100) where the real ones ran 1.1-6.2,
  so a recovered segment over 8 words per second is rejected (`too_fast`), for both triggers.
- Foreign-language tape in a recovered gap is decoded in the episode's declared language, as the
  whole episode is: the declared language is the truth, and a wrong one is caught by the language
  check (`asr_language_mismatch`), not by guessing per clip. Whisper then sometimes renders the
  foreign speech in the episode language instead of transcribing it (Rádio Novelo 09-24: a Spanish
  bolero came back as Portuguese, "Já perfumado, e os seus lábios..."). That is the model's
  behaviour, kept and recorded; it cost Novelo 09-24 3 main-voice errors in the measurement.
- To probe a span by hand, cut the episode audio at the transcript's times (preprocessing keeps
  the timeline, #1173) and compare the clip's words with the episode's words around the span,
  not with the stretched word: the episode text there is often what Whisper misheard.

### ASR gates: what is measured, on which voices, against what

How the tier-1 gate was built (2026-10-10), in the order it was worked out, so the reasoning is not
lost and the dead ends are not walked again.

**Two metric sets per episode.** The scorer (`asr_publisher_reference_wer_v1.py`) reports the whole
episode and, separately, its **main voices** (`main_voices` block per episode,
`main_voices_wer_adjusted_pooled` per language). The gate is set on the main voices. Nobody
expects archive clips, songs, film audio, field tape or ads to transcribe well, and on the first
reference set the voices other than the main ones carried 33-92% of an episode's errors while
being a minority of its speech.

**What a main voice is.** Presenters, narrators, readers, and guests in conversation with the
host. Not main ("tape"): voices cut into the narration — interview tape, field recordings, voice
messages, archive clips, songs, film. Four ways of finding them were tried; three failed:

| tried | why it failed |
| --- | --- |
| a share of diarized speech (e.g. every voice >= 10%) | no cut works across shows: Senado has 6-7 reporters at 9-24% each, SWR one narrator at 70% and nothing above 6%, Radio Ambulante a long tail at 4-10%; any threshold is invented |
| the roster's host/guest roles | naming is tier 3 and still wrong off English (SWR's narrator, 73% of the speech, unseated); it would tie the ASR gate to a layer above it |
| turn length (main voices speak in long turns) | field voices have 60 s turns, Senado reporters 6 s ones |
| the publisher's labels ("Archivo de audio", "O-Ton", "Música") | works only where the publisher writes clips into the transcript (SWR, a little Radio Ambulante); named interviewees are labelled like presenters, and English publishers mark clips inconsistently (Freakonomics labels an archival clip "PRESIDENT OBAMA") |

**What is used: hand tags, once per reference episode.** Every speaker label of a reference is
tagged `main`, `tape` or `unsure` by reading the reference, in
`data/eval/references/gold/asr_publisher_transcripts/<set>/main_voice_tags.json` (eval repo).
`tape_prefixes` covers families of labels ("O-Ton", "Archivo de audio"). `unsure` turns are on
neither side of the score. A reference with no speaker labels (RFI, Senado: one script) is
`unseparated`: its main-voice figure is the whole episode, and the report says so. A label the
tags do not cover is reported (`untagged_labels`), never counted silently. The label rule is the
fallback for an untagged set. Reference sets are small (a dozen episodes), so the tagging is
minutes per episode and the result is exact.

**Gate A: nothing left that is ours to fix** (per episode, independent of English):

| check | pass |
| --- | --- |
| language heard = language asked | 100% |
| long deletions (5+ reference words in a row) | each read and explained: not spoken (dates, sign-offs, stage directions in the script), music, a foreign-language original under a voice-over, or field tape |
| untranscribed diarized speech after recovery | 0, or each explained |
| stretched words still over speech after D16 | 0, or each read |
| each recovery step (gap, D16, punctuation) | no episode's MAIN VOICES worse on adjusted WER (`asr_bare` against full chain); recovering real speech inside tape the publisher did not transcribe raises the whole-episode figure and is not a fault (RFI 10-07: a 3-word interview clip) |
| invented lines left in the text | 0 |

**Gate B: good enough against English, per language and per kind of show.**
`gate = English main-voice WER on the same kind of show x the language's ratio`, where the ratio is
the Whisper paper's FLEURS rate for the language over English's (large-v2, Table 13; PRD-047
Appendix A): es 3.0, it 4.0, pt 4.3, de 4.5, fr 8.3 against en 4.2, floored at 1.0 so no language
is asked to beat English on podcasts. FLEURS is read speech, so only the RATIO is used, never the
absolute rate.

- **Genre-matched, or it misleads.** On the first set French "beat the paper" because its only
  show is slowly read news, the genre closest to FLEURS; Portuguese read news (Senado) matched
  English while Portuguese narrative (Novelo) did not. The same language spans 0.015-0.109
  adjusted across shows. English must be measured on each kind of show a language is gated on:
  studio interview, read news, narrative journalism with tape.
- **Speech rate, music share, voices and overlap per episode** are measured
  (`show_profile`-style: words per minute of speech, diarized speech over duration, voices, main
  voices' share), so a gate can be read against the show's nature, not only its language.
- **No tolerance is invented.** A first proposal used "x 1.5"; it was a round number, not a
  measurement, and is withdrawn. The margin has to come from measured variation: re-decoding the
  same audio is reproducible (Radio Ambulante 09-10 and SWR re-decoded on 2026-10-10 gave the
  same adjusted WER to the fourth decimal), so the margin is the spread between episodes of the
  same show and genre, measured once English and each language have 3+ episodes per show.
- **A language failing gate B only on plain misheard words** (gate A clean) gets one model
  comparison (#2251); if that does not close the gap, the gap is recorded and the language moves
  on to translation.

**Italian reference set (2026-10-11, eval `asr_publisher_ref_it_v2`)**: Italiano ON-Air (two hosts
in a prepared conversation, the school's own transcripts in the feed) and the Camera dei
deputati's Question Time (the stenographic record against Radio Radicale's recording of the
sitting, cut to the Question Time window read off its per-speech offsets), 3 episodes each. A show
with no feed can now supply its own (`Publisher.feed`) and its own audio (`fetch_audio`). Both
take English news as their gate B bar: Question Time is news; ON-Air is scripted studio speech
with no tape, closer to read news than to narrative with tape, and the stricter bar of the two.
Podcast Italiano, the earlier candidate, is gated behind a free login and one voice. The Camera's chair is labelled with
the presiding name the record gives ("PRESIDENZA DELLA VICEPRESIDENTE ANNA ASCANI"), not
"PRESIDENTE", so a name resolved for that voice can be scored (v2; v1 labelled the role).

**Evidence needed per language**: at least 2 shows x 3 episodes, one of them interview or news,
each with a human reference and its tags.

**First set (2026-10-10), full chain, adjusted WER, whole episode against main voices only:**

| language | whole episode | main voices | note |
| --- | --- | --- | --- |
| es | 0.015-0.109 | 0.011-0.060 | Radio Ambulante 07-21: 0.109 whole, 0.035 main (0.033 with D16) |
| pt-BR | 0.023-0.068 | 0.023-0.053 | Senado unseparated |
| de | 0.060-0.084 | 0.037-0.042 | SWR's "O-Ton" soundbites were the excess |
| fr | 0.043-0.047 | 0.043-0.047 | unseparated |
| en | — | 0.028-0.034 | 80,000 Hours, 3 episodes, ASR only: the one genre measured so far |

The large whole-episode figures are tape; on main voices every language sits near English's band.

**Gate B, first result (2026-10-11): five shows pass; Senado sits on its bar (inconclusive).** English was
measured on the two kinds of show the six non-English shows are: read news (NPR Up First) and
narrative journalism with tape (NPR Throughline), 3 episodes each, with hand tags (eval dataset
`asr_publisher_ref_en_v1`). Each non-English show got a third episode
(`asr_publisher_ref_nonen_v1b`). The rule, with nothing invented: a show passes when the
**median** of its 3 episodes' main-voice adjusted WER is at most `(English median + English
range) x ratio` for its kind of show, the range being the spread between English's 3 episodes of
that kind.

| English, main-voice adjusted | episodes | median | range |
| --- | --- | --- | --- |
| read news (Up First) | 0.0217 / 0.0247 / 0.0281 | 0.0247 | 0.0064 |
| narrative (Throughline) | 0.0224 / 0.0232 / 0.0382 | 0.0232 | 0.0158 |
| interview (80,000 Hours, for reference; no non-English show is this kind) | 0.028-0.034 | | |

| show | language, ratio | kind | episodes | median | bar | margin |
| --- | --- | --- | --- | --- | --- | --- |
| El Hilo | es, 1.0 | narrative | 0.060 / 0.016 / 0.038 | 0.0377 | 0.0390 | 0.0013 |
| Radio Ambulante | es, 1.0 | narrative | 0.034 / 0.011 / 0.048 | 0.0339 | 0.0390 | 0.0051 |
| Rádio Novelo | pt, 1.02 | narrative | 0.035 / 0.029 / 0.060 | 0.0351 | 0.0399 | 0.0048 |
| SWR Wissen | de, 1.07 | narrative | 0.037 / 0.038 / 0.045 | 0.0378 | 0.0418 | 0.0040 |
| RFI Journal en français facile | fr, 1.98 | news | 0.048 / 0.043 / 0.034 | 0.0427 | 0.0615 | 0.0188 |
| Senado | pt, 1.02 | news | 0.053 / 0.023 / 0.032 | 0.0318 | 0.0318 | 0.00004 |

Scored with the spoken and variant forms sourced from the data and the bracketed notes dropped.
Each ruler fix helped English too (notes: Throughline 09-24 0.0254 -> 0.0224; "ok/okay": Up
First), so the bars fell with every one and Senado's margin went 0.0012 -> 0.0001 -> 0.00004.
That last margin is inside the 4-decimal rounding of the stored WERs: Senado is ON its bar, not
over or under it. These are the runs before the silent-gap trigger and the punctuation fix; the
one-version re-run of all 24 decides it.

What this result does not show:

- **The margins are inside the noise of 3 episodes.** Senado is on its bar, El Hilo 0.0013
  under. One more English or non-English episode can move a median by more than that.
  The bar is as wide as English's own spread over 3 episodes, nothing more.
- **Rádio Novelo passed only after the spoken-contraction rule.** Before it Novelo's median was
  0.049 against 0.041. The rule was written after reading Novelo's errors (pra/para 43, tá/tava/tô
  17 of 303 on 09-17), so it was checked on every episode: it fired on Novelo only (105, 54 and 51
  words) and on no English, Spanish, French, German or Senado episode. Senado's newsreaders read
  standard Portuguese.
- **Two code versions.** The first two episodes per show ran on 2026-10-09 code, the third on
  the branch at `23956c651` (the run records that build; the same commit is `5541acd7e` after the
  rebase onto main). Re-decoding the same audio was reproducible (above), but the chain changed
  between them; they were not re-run on one version.
- **Residue left on the third episodes.** Novelo 09-17 (0.060 after the rule) is name spellings
  (Elder / Hélder 19, Adivaldo / Advaldo 7, Aurélio / Aurelio 7) and fillers the publisher edited
  out ("e", "né", "assim", "aí", "eu", about 60); its 20 "[ri]" notes sit in interviewees' tape
  turns, so dropping them moved the whole episode (0.097 -> 0.086), not the main voices. Senado
  10-07's 0.053 includes 10-word sign-offs ("Da Rádio Senado, …") that are in the script and not
  in the audio (a 0.12 s gap where two would be; the third, at 307 s, sits in a 5.2 s gap that
  transcribed alone holds no words); without them it is about 0.045. Radio Ambulante 07-14
  (0.048 after the notes) is about 25 name spellings (Iolany / Yolani, Azpiri / Aspiri, Annie /
  Ani), about 12 accent-only differences (cómo / como 6, sólo / solo 2), a word the reference
  spells letter by letter ("A-S-E-S-I-N-A-T-O", 9 errors), a line not in the audio ("Así lo
  afirma un hoy", no gap at 1480.86 s), and plain mishearings ("disc jokey" -> "dijoque y",
  "amoratado" -> "amor atado"). Senado 10-02's two-word deletion is the reading differing from
  the script (order of "Câmara" and "Tribunal de Contas", and the technician's name). El Hilo
  10-09 quotes an English ad inside Spanish ("they/them"); Whisper, decoding Spanish, renders
  part of it in Spanish: a model limit.

**Reading the residue (how gate A was closed on the first set).** Each item is read, not counted:

- Main-voice short errors (spans under 5 words) run 0.011-0.053 against English's 0.028-0.039,
  and the share that is a filler the publisher edited out is similar (2-29% against 9-23%): once
  tape is excluded nothing is language-specific. The highest, El Hilo 09-25, is Rioplatense voseo
  written as standard Spanish ("sabés" -> "sabes"), brand names ("ChatGPT y Claude"), and fillers.
- A long deletion can be text the publisher kept and the audio cut: El Hilo 09-25's 32 words sit
  where the transcript's segments run on without a gap (2126.98 s -> 2127.16 s), and appear
  nowhere else. Check the timeline before calling a deletion lost speech.
- Untranscribed stretches left after recovery: jingles, songs, an old record, field tape, the
  start of an interview clip. The last one (Novelo 09-24, 3609 s, on a main voice's diarized
  cluster, recovery rejected as low confidence) was settled with one clip call: a German archive
  recording inside a Portuguese episode, decoded in Portuguese, which the diarizer had merged into
  a main voice. Tape. Gate A passes on the whole first set (2026-10-10).

**Gate B, re-run of all 30 (2026-10-12): six shows pass; El Hilo and Camera are over.** All 30
reference episodes (English 6, Spanish, Portuguese, French and German 24, Italian 6) re-run on the
branch (two builds, below), scored with `asr_publisher_reference_wer_v1.py`, gated with `asr_gate_report_v1.py`
against the earlier runs re-scored with the same scorer.

| English, main-voice adjusted | episodes | median | range |
| --- | --- | --- | --- |
| read news (Up First) | 0.0217 / 0.0271 / 0.0281 | 0.0271 | 0.0064 |
| narrative (Throughline) | 0.0224 / 0.0232 / 0.0346 | 0.0232 | 0.0122 |

| show | language, ratio | kind | episodes | median | bar | result |
| --- | --- | --- | --- | --- | --- | --- |
| Italiano ON-Air | it, 1.0 | news | 0.0074 / 0.0119 / 0.0113 | 0.0113 | 0.0335 | pass |
| Radio Ambulante | es, 1.0 | narrative | 0.0271 / 0.0109 / 0.0475 | 0.0271 | 0.0354 | pass |
| RFI Journal en français facile | fr, 1.98 | news | 0.0476 / 0.0427 / 0.0454 | 0.0454 | 0.0662 | pass |
| SWR Wissen | de, 1.07 | narrative | 0.0287 / 0.0346 / 0.0402 | 0.0346 | 0.0379 | pass |
| Senado | pt, 1.02 | news | 0.0534 / 0.0230 / 0.0318 | 0.0318 | 0.0343 | pass |
| Rádio Novelo | pt, 1.02 | narrative | 0.0357 / 0.0289 / 0.0575 | 0.0357 | 0.0362 | pass by 0.0005: on the bar |
| El Hilo | es, 1.0 | narrative | 0.0588 / 0.0157 / 0.0373 | 0.0373 | 0.0354 | over by 0.0019 |
| Camera, Question Time | it, 1.0 | news | 0.0896 / 0.0808 / 0.0919 | 0.0896 | 0.0335 | over (0.0883 with the Italian spoken forms added 2026-10-13; still over) |

Against the earlier runs, main voices: 21 of 24 episodes equal or better (Radio Ambulante 07-21
0.0339 -> 0.0271 and SWR 0.0371 -> 0.0287 from stretched-word recovery). Worse: RFI 10-09
(0.0344 -> 0.0454, the punctuation retry below), Up First 10-09 (53 -> 58 errors) and Novelo
09-24 (262 -> 267), the last two not separated from decode variance.

Reading the two over:

- **El Hilo** is over on one episode: 09-25 (0.0588; Rioplatense voseo written as standard
  Spanish, brand names, fillers, as read on the first set). The other two are 0.0157 and 0.0373.
  A margin of 0.0019 on three episodes is inside the noise described above.
- **Camera** is over by a factor of 2.7 and most of it is the reference. The stenographic record
  is an edited record, not a verbatim transcript: the chair's spoken "prego, onorevole" and
  "grazie, grazie ministro" are recorded as "ha facoltà di replicare"; spoken fillers ("appunto",
  "allora", "prego": 13 insertions on one sitting) are not recorded; the record normalises
  euphony (a / ad, e / ed). The scorer adds to it: Whisper writes "54,3%", the digits are
  classed as a number difference, but the reference's "per cento" is left unmatched and counted
  (13-26 errors per sitting); "1.000" against "mila" likewise. Real ASR errors remain: names
  ("Aspides" -> "a spides"), compounds ("superbonus" -> "super bonus"). Camera stays in gate B and
  is reported over; whether the scorer should read "%" as the language's spoken word is a ruler
  change for every language and baseline at once, not made here.

What this result does not show:

- **Two builds.** English and the 24 Spanish, Portuguese, French and German episodes ran on one
  build; the Italian set ran on the next, which adds the digital-silence cut (below). Between the
  two, the only changes on the ASR path are that cut (it fires on the three Camera sittings only:
  checked on all 30 preprocessed files) and the punctuation repair's tail rule (punctuation only;
  WER is scored without punctuation). The recovered-word join fix (below) was measured by replay
  on the stored word lists, not by a run: 11 recovered segments change, and only Camera 09-23's
  figure moves (0.0896 -> 0.0882).
- **The punctuation retry costs one French episode.** RFI 10-09 came out unpunctuated (2.0
  sentence ends per 1,000 words); the retry with the French prompt now runs and punctuates it
  (70.0), and its decode spells "Pillay" as "pile" (5) and "Wembanyama" as "wambanyama" (4):
  +16 adjusted errors. The retry fired on 2 of 30 episodes.
- **Foreign tape is decoded in the declared language** (operator's rule, no language detection).
  Words of another language under a voice-over are not transcribed as that language: SWR 10-07's stretched "einige" hid 13 English
  words ("We have maybe a few cases ... you have to be very reactive"); the clip, decoded in
  German, returned "einige" and recovery correctly declined. RFI 10-09's English Nobel
  announcement (18 words) was recovered and scores as an insertion the reference does not hold.

Found and fixed on this run (each with a failing test first):

- **Digital silence at either end is cut off before ASR.** Camera 09-30 opens with 103 s of true
  zeros: Whisper wrote "Grazie a tutti." over the whole 66-minute sitting (417 words, on three
  attempts and three probes). With the silence cut the same audio gave 8,980 words against a
  reference of 8,836. The cut is for a -70 dB peak held 10 s at an end (MP3 coding noise inside
  true zeros peaks above -90 dB, so -90 found nothing), keeps 1 s of margin, shifts the result
  back to episode time, and keeps gap recovery out of the cut (on 10-07 silent-gap recovery had
  added invented words in 70 s of zeros at the end). Only the three Camera sittings are cut.
- **A failed transcription counts as failed.** 09-30's failure left the episode "ok" from
  queueing (`episodes=3 ok=3 failed=0`). The transcription-failure paths now write the episode
  ledger row; the guard test no longer accepts only a literal `0` as the refusal's third value.
- **A guardrail violation is raised at once.** The endpoint answered; the answer failed a
  content check. The reprocess policy retried it three times, opened the breaker, held 900 s and
  reported "endpoint did not recover", blocking the episodes behind it.
- **Recovered words are joined as Whisper wrote them.** Re-spacing stripped tokens split every
  elision ("l 'ordine", "l 'intégralité").
- **A punctuation repair is judged on the text it would splice in.** Up First 10-09's window
  (602-768 s) was refused as repetitive: over the clip's silent end Whisper re-emitted lines the
  credits had said, and its reported ratio counts text it never put in a segment. Lines an earlier
  decode window holds are dropped from the clip's last window and recorded; a window is repetitive
  only when both the reported ratio and its segments' own text pass 2.4. On the re-run the window
  was repaired.

Known and not fixed: **invented lines at the end of a first-pass transcript.** Up First 10-08
ended "and we'll see you next time" 13 times in one segment (85 words in 6.3 s); RFI 10-09
repeated a line twice at 0.08 s each; SWR 10-07 a line twice. No rule built on text and timing
alone separates these from real speech: checked on the restored prod corpus (2,628 transcripts),
"a segment repeating the one before it, faster than 8 words/s" matched 829 segments in 337
episodes, both invented outro and ad lines ("Godspeed, keep going and God bless." at 150 words/s)
and real repeated short replies ("Yeah.", "right,"); a phrase repeated three times inside a
segment matched rhetorical repetition as well as loops. The signal that generalises is the audio
(an invented line sits where diarization hears no voice); not built. Long insertions are already
out of the adjusted figures.

## 5. Naming

- `relabel` at the fix commit against the run it copied, scored with
  `naming_publisher_reference_v1.py RUN --compare BASE`: correct / misspelled / wrong / unnamed,
  as fractions of the reference's speaking time.
- **Repeat every relabel at least 5 times.** The LLM naming call runs at temperature 0 and still
  answers differently between identical requests. One El Hilo episode gave correct 0.97, unnamed
  0.53 and wrong 0.50 across 17 runs of the same code. A single run is not a result, and the arc
  reported one as one.
- `wrong` must be 0 in every run (#876: a wrong name is worse than an unnamed voice).
- A non-zero `wrong`: read `.speakers.diagnostics.json` → `decision_trace.inputs.llm_resolution`
  (raw answer and every verdict) and `decision_trace.voices.<SPEAKER>` (which rung named it).
- The same diagnostics exist on prod episodes: before a naming rule, count its cases on the prod
  snapshot (`WORKSPACE_DIR=<dir> PODCAST_BACKUP_TAG=<tag> make restore-corpus-prod`) and read them. D27 was decided on 2 prod
  cases, both wrong; a candidate for D29 fired on 7 prod episodes that were all correct and was
  dropped.

## 6. Translation

1. **Outcome**: every unit translated (`units_failed: 0` in `.translation.json`), and a refusal is
   retried (`attempts`). Before D10, one refused unit withheld a whole episode's English.
2. **Quality**: `translation_judge_v1.py RUN_DIR` (Opus judge; key
   `AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY` in the eval repo's `.env`, never prod's or another
   project's). Per unit: adequacy 1-5 (meaning kept), fluency 1-5, typed errors; means weighted by
   source words. About 45-50k tokens per episode.
   - Check `units_unparsed` and `cut` before reading a mean: a reply that does not parse loses
     its whole window. 102 of 192 units on one episode were lost because claude-opus-5-5 counts
     its reasoning against `max_tokens`: at 4,096 it ran out mid-JSON (now 16,000; eval repo
     `ad15018`). A partial score misleads: Radio Ambulante 07-21 read 4.26 with 37 units missing
     and 4.10 complete.
   - Read the low units (`adequacy <= 3`), not only the mean: the errors' types say whether the
     fix is the model, the prompt, or how turns were cut into units.
   - The judge reads the ASR text as the source: it measures translation, not hearing.
   - The judge varies too: the same El Hilo translations judged twice gave adequacy 3.58 and
     3.44, low units 18 and 22 of 79. A difference of that size between two runs is the judge,
     not the translation; compare settings over repeated judgements or the same windows.
3. **A setting that changes wording** (concurrency, model, prompt): compare against two runs of
   the unchanged setting first. If two identical serial runs already differ by N units, a
   difference of N under concurrency is not caused by concurrency.
   - Measured for `translation_max_concurrency` (2026-10-10, El Hilo x2, two runs each, eval
     variants `translate_fresh_serial` / `translate_fresh`): wording, serial against serial 68 of
     85 units identical, serial against 4 in flight 65-66; judged, serial 3.47-3.54 / 3.94-4.02
     adequacy, 4 in flight 3.55-3.61 / 3.97-4.02, low units 19-21 either way. No quality
     difference beyond the judge's noise; 4 in flight is about 3.5x faster (525-574 s against
     1,782-2,062 s).
4. **Anchor the judge on a human translation** before reading its numbers as good or bad:
   `translation_judge_human_v1.py DATASET` judges a publisher's own translation with the same
   judge and windows. El Hilo (es, the one show of the first set that publishes one): adequacy
   4.60-4.71, fluency 4.95-4.96, 0-2% low units. A careful human translation is the ceiling, not
   5. The pipeline on the same episodes: 3.45-4.12 adequacy, 18-28% low units; the gap (0.6-1.2)
   is many times the judge's noise and is where the tier-2 work is.
5. **Read the main voices, as for ASR.** The judge report carries a `main_voices` block per
   episode (`--rescore` adds it to an old report without judge calls): a unit counts when most of
   its words align to the reference's main-voice turns, so ads and archive clips are out. On El
   Hilo about half the low units are ads (another show's promo, iHeart spots), but they are short:
   main-voice adequacy moves by under 0.1. The comparison that matters, main voices against the
   human anchor's main voices: 3.38-3.54 / 3.95-4.06 against 4.59 / 4.70 (five unchanged runs).
6. **Sort the low units by cause before building a fix.** The model's output is the same run to
   run (31 of 51 low units low in all four baseline runs), so the errors are systematic and only a
   change of input moves them. On El Hilo (2026-10-10):
   - Unit length, NOT a cause after all. Units of 200+ source words were judged low 43-50% of
     the time against 12% for 50-100 words, but the judge grades a unit as a whole, so a long unit
     collects more errors per grade. Regrouping proved it (item 8): `translation_max_unit_words:
     100` gained nothing.
   - Short fragments, slang, backchannels have the highest error rate (about 5 per 100 words
     against the human 0.3): no surrounding context. Switch: `translation_context_sentences`.
   - Unpunctuated ASR text makes one "sentence" of hundreds of words (654, 339).
     `translation_max_sentence_words` cuts one at ASR segment ends; unmeasured, and with unit
     length not a cause its premise is gone.
   - Names: the show's own title as a common noun ("en el hilo" as "the thread").
   - Not fixable in code: idioms and slang the model does not know ("grandes ligas", "hacer
     agua"), and ASR errors the model invents around ("valesía").
7. **Measure each switch alone against unchanged runs at the same code**, and repeat a run before
   believing a single gain. Results so far (El Hilo, main-voice adequacy / low units, against
   3.38-3.54 / 11-14 and 3.95-4.06 / 7-8):
   - `translation_context_sentences: 2`: 3.42 / 12 and 4.19-4.21 / 4, two runs. A gain on
     10-02, none on 09-25. The two runs wrote identical English, so the repeat measures the
     judge (steady within 0.02), and the five unchanged runs measure the translation's own spread.
   - `translation_protect_names` over the episode metadata's capitalised names: 3.34 / 11 and
     3.82 / 9, a net loss. Places with an English name ("América Latina") came back untranslated,
     a placeholder lost the grammar around it ("Desde [1]" as "Since"), and the show title was
     missed because the host says it in lower case. Narrowed to the show title, case-blind when
     multi-word (`9452cd455`): 3.43 / 11 and 4.10 / 5. It touches 8 units, so read them, not
     the mean: all four show-name errors ("back on track", "back to the thread") are gone, three
     units rose (3->5, 4->5, 4->5), none fell.
   - `translation_max_unit_words: 100`: 3.82 / 4.24 on its own units, which looked like the
     largest gain, and was not one (item 8).
   - Context and show-title protection together: 3.49 / 11 and 4.18 / 3, the fewest low units
     of any run on 10-02.
   - The same pair on the other five shows (2026-10-10, one run each, main-voice adequacy against
     the switches-off run): Rádio Novelo 3.56/3.87 -> 3.74/3.91 (low units 47 -> 39), RFI
     4.07/4.23 -> 3.93/4.54, Senado 4.37/4.78 -> 4.38/4.47, Radio Ambulante 4.47/4.33 ->
     4.42/4.37, SWR 4.29/3.88 -> 4.28/3.79. Low units 90 -> 83 of 498. Mostly inside the 0.3 two
     unchanged runs differ by: context helps conversational shows with short turns (El Hilo,
     perhaps Rádio Novelo), not scripted news or documentary; show-title protection fixes its own
     case only. Both stay OFF in the profiles. What remains is the model's (vocabulary, idiom,
     meaning lost in long answers): the point where a model comparison is worth running.
8. **A switch that changes how turns are cut into units changes the judge's ruler too.** The
   judge grades per unit; the same English graded in smaller units scores higher. Compare such a
   switch only on IDENTICAL grading units: regroup the baseline's English (sentences are the
   alignment atom, so every sentence id exists in both runs) into the new run's units and judge
   that. Measured: the unchanged English regrouped into the 100-word units scored 4.02 / 4.44
   main-voice adequacy against 3.52 / 3.95 on its own units, and the capped run's English 3.82 /
   4.24, about 0.2 BELOW the baseline on the same ruler. Splitting a turn costs the model
   context. Context and name protection do not change units, so their comparisons stand.
9. **Italian, first judgement (2026-10-13)**, on the re-run of 2026-10-12, the first two episodes
   each show lists in the dataset, as for the other languages (2 per show): main-voice adequacy /
   fluency / low units: Italiano ON-Air 10-07 4.68 / 4.93 / 4 of 25, 07-01 4.24 / 4.77 / 4 of 35;
   Camera seduta 717 3.64 / 4.56 / 12 of 56, seduta 722 3.42 / 4.50 / 29 of 62. Every unit
   parsed, none cut; 114,301 + 43,436 tokens. Scripted studio speech is among the best of any
   language; the parliament is at the bottom with El Hilo 09-25. Camera's low units are the
   translator's, read on the source: invented acronym expansions ("PEFE" as an environmental
   assessment; it is the economic-financial plan; "iter" treated as an acronym; CIPES), the
   chamber's formulas taken literally ("ha facoltà di replicare" as "replicate statements",
   "Prego" as "you're welcome" for "please go ahead"), the formal "lei" as "she". The model's, as
   in item 7. The translation code judged on 2026-10-10 is the code that ran on 2026-10-12: every
   later translation change is behind a switch that is off in the profiles, so the other
   languages' judgements stand. Outcome on the re-run: all 24 translated episodes complete
   (`units_failed: 0`). 6 units needed a retry and all passed: 3 refused as commentary ("here's
   the translation"; El Hilo x2, Radio Ambulante), passed on the numbered retry; 3 returned fewer
   lines than sentences (SWR x2, Rádio Novelo), passed on the third attempt.

## 7. Anything that changes English output

Ad patterns and name rules are shared with the English corpus. Before shipping a change that can
touch English episodes:

- Replay it on the prod snapshot, old code against new, no LLM:
  `scripts/measure/roster_replay.py --corpus <snapshot>/corpus --repool --old hosts=<old.py> --new hosts=<new.py>`
  (roster, hosts, resolution); for ad cutting, run `build_adfree_artifacts` with each version and
  diff the excised text.
- **Run a no-op replay first** (old against old): it must change 0, or the replay itself is noisy.
- **A zero is a result about the inputs.** Count how often the rule's input occurs on prod before
  calling it safe: the three held-back English changes changed 0 of 2,454 episodes because prod has
  2 accented one-word self-introductions in 3,236 transcripts.
- **Read every changed case.** D15's generic cue ("wherever you get your podcasts") would have
  cut Hard Fork's and The Rest Is Science's own intros; only reading the replay showed it.

## Verification

A language is measured when, for each show: the baseline run exits 0; adjusted WER and its long
deletions are read; five relabels show `wrong` 0; every episode is translated and judged with
`units_unparsed` 0 and its low units read; and every change to shared English code has a read
replay.

## Troubleshooting

- **Judge HTTP 400 "temperature is deprecated"**: the model refuses the parameter; fixed in the
  eval repo's `podcast_scraper_eval/judges/sonnet46.py` (`67bfc31`).
- **Judge `anthropic` missing**: the eval venv lacks the SDK; the public repo's `.venv` has it
  (`../.venv/bin/python scripts/eval/score/translation_judge_v1.py`). Installing into either venv
  needs the operator's yes.
- **A relabel shows a name that was correct yesterday**: run it five more times before looking for
  a code cause (section 5).
- **Feed transcript published in the feed (`<podcast:transcript>`)**: prod uses it instead of our
  ASR (D6); score it with the same WER scorer before deciding which to keep.
