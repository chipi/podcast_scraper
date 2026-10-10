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
- DGX: 1-2 episodes without asking; more needs the operator's yes. Before spending a DGX run on a
  phenomenon, check which episode carries it: the arc ran the stretched-word test on the wrong
  Radio Ambulante episode once (D16).
- A config `null` now unsets a profile value (D20); before `0252e8b97` it was silently ignored.

## 4. ASR

- `asr_publisher_reference_wer_v1.py RUN_DIR`. Normalisation `unicode-v1`: `ascii-v1` breaks every
  accented word.
- Read **adjusted** WER: numbers, long insertions and long deletions are split out. Raw WER on
  iHeart shows is inflated by ~750-785 words of ads the reference does not contain.
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
Gate B numbers wait for English on narrative and read-news shows (parked until the DGX is free).

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
