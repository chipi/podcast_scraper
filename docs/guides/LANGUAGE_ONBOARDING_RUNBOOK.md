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
  speech can hide a sentence (D16). Recovery replaces the word only when its span re-transcribes
  as the word plus at least 3 non-filler words.

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
   - Check `units_unparsed` before reading a mean: a reply that does not parse loses its whole
     window (102 of 192 units on one episode before the retry, `4f2804b` in the eval repo).
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
