# Brief search: finding what a listener remembers (eval, 2026-10-10)

## The use case

Someone listened to an episode, remembers a few words or a term, and wants that part again: to
highlight a piece of the transcript, or to find the insight about it. The Brief's search is the
only search limited to one episode, so it is the one that has to do this.

## Method

- **Data:** 94 real prod episodes with timed transcripts (the Moments eval sample, kept locally in
  the gitignored `.test_outputs/`). Indexed locally with the prod indexer: 3,926 transcript
  passages, 4,653 insights, 10,293 other rows. The download had no plain-text transcripts, so each
  `.txt` was rebuilt by joining its Whisper segments in order (same words).
- **Search:** the Brief's own call, `structured_corpus_search(feed=, episode_id=, top_k=10)`, after
  the fix that scopes the query to the episode (`80dea232b`).
- **Queries:** 10 random passages per episode (~30 words each), five ways a listener might
  remember each: the most distinctive word (rarest in the episode); three distinctive words,
  shuffled; four consecutive words verbatim; the distinctive word in another form; the three words
  with one swapped for a near-synonym (written by Claude Haiku, ~$0.17 in all).
- **Found:** a transcript result contains the passage (a run of eight of its words). For the 324
  passages with an insight grounded in them, also: does that insight come back in the 10.
- **Baseline "exact":** a plain word search over the episode's transcript — the passages that
  contain every query word as a whole word.
- Script: `scripts/eval/brief_search_eval.py` (reruns in ~6 minutes; numbers only leave `--out`).

## Results (4,700 queries)

| Query | Found first | Found in top 3 | Found in top 10 | Insight found (of 324) | Exact search finds it | Transcript results in the 10 |
|---|---|---|---|---|---|---|
| one distinctive word | 23% | 72% | **95%** | 60% | 100% | 3.6 |
| three words, shuffled | 16% | 57% | **92%** | 71% | 100% | 4.1 |
| four words verbatim | 10% | 39% | **73%** | 55% | 100% | 3.4 |
| another word form | 21% | 67% | **89%** | 59% | 1% | 3.6 |
| one word misremembered | 14% | 54% | **92%** | 69% | 5% | 4.2 |

Two runs give identical numbers for the first four rows (seeded); the misremembered row moves by
under a point between runs because the LLM writes slightly different synonyms.

Across all queries the 10 slots held: insights 32%, transcript 38%, quotes 26%, summary /
description / title 5%.

## What it says

1. **It mostly finds the passage, but rarely first.** 89–95% in the top 10 for words, 10–23% first.
   A listener scanning the first few results misses it half the time.
2. **A remembered phrase is the weakest query (73%).** Four verbatim words are diluted by their
   common words, for keyword and meaning search alike — while a plain word search finds the same
   passage every time, usually as the only match (median 1).
3. **Meaning search earns its place on fuzzy memory.** A word in another form or one word
   misremembered: 89–92% found, where a plain word search finds 1–5%.
4. **Transcript gets about a third of the slots.** Insights and their supporting quotes (the same
   evidence twice) take most of the rest, so a good passage can sit 11th.
5. **The grounded insight comes back 55–71% of the time.**

## Defects found alongside (prod, checked read-only)

- **Transcript results have no time on prod.** Every transcript passage in the prod index has
  `timestamp_start_ms: 0` (the chunker times a passage only from segments with character offsets,
  which Whisper segments do not carry). The Brief then shows "▶ Play from 0:00" on a transcript
  result and jumps to the episode's start — so even a found passage cannot be played or
  highlighted where it was said.
- **A transcript result is a ~300-word passage** with nothing marking the remembered words, and
  offers no way to highlight it.

## Options

| | Change | Fixes |
|---|---|---|
| A | **Exact matches first:** search the episode's own timed transcript for the words (server, text files, no index change), and put those passages above the meaning results | phrase recall 73% → ~100%; each exact match has its real time |
| B | **Time every transcript result at read time:** map a passage back to the timed transcript by its text when serving it (rung 1 of the repair ladder, no re-index) | "Play from" lands on the passage, prod included |
| C | **Two groups instead of one mixed list:** transcript passages, then insights (quotes folded into their insight) | the passage and the insight both visible; no duplicate evidence crowding |
| D | **Show the sentence, not the passage:** the one or two sentences around the words, words highlighted, with Play from and Highlight | the use case's last step: listen, then save the piece |

Recommended: all four, A and B first (they decide whether the passage is found and playable).

## After A and B (same queries, `--brief`)

A (verbatim passages first) and B (a real time on every transcript result) are built
(`app_episode_search.py`); the eval's `--brief` mode composes the results as the route now does,
and counts a timed result overlapping the passage as finding it.

| Query | Found first: before → after | Top 3 | Top 10 |
|---|---|---|---|
| one distinctive word | 23% → **95%** | 72% → **100%** | 95% → **100%** |
| three words, shuffled | 16% → **85%** | 57% → **93%** | 92% → **98%** |
| four words verbatim | 10% → **99.6%** | 39% → **99.7%** | 73% → **99.7%** |
| another word form | 21% → 20% | 67% → 65% | 89% → 90% |
| one word misremembered | 14% → 18% | 54% → 56% | 92% → 92% |

What a listener says verbatim is now found first almost every time, at the moment it was said.
Word forms and misremembered words still rely on meaning search (unchanged, as expected); the
grounded insight still comes back 55–71% of the time — C and D next.


## After word forms and one misremembered word (2026-10-10, later)

`exact_passages` gained two tiers after the verbatim ones:
- **forms:** every remembered word in some form, by an inflection key (`_form`: plural, -ed, -ing;
  not a stemmer, so "police" and "policy" stay apart);
- **most:** for three or more words, all but one of them.

Each passage carries the words it matched (`marks`), so the Brief marks "investing" for a typed
"invested". Same 94 episodes, same seed, the cached synonyms (no LLM calls:
`synonym_tokens: 0/0`), `--brief`:

| Query | Found first | Top 3 | Top 10 |
|---|---|---|---|
| one distinctive word | 95.4% → 95.4% | 99.6% → 99.6% | 100% → 100% |
| three words, shuffled | 84.9% → **95.3%** | 93.4% → **98.4%** | 98.3% → **99.4%** |
| four words verbatim | 99.6% → 99.7% | 99.7% → 99.8% | 99.7% → 99.8% |
| another word form | 20.2% → **72.1%** | 65.4% → **85.9%** | 89.5% → **93.9%** |
| one word misremembered | 17.8% → **86.5%** | 55.9% → **94.9%** | 92.2% → **98.8%** |

The insight-found column is unchanged (55–71%): these tiers add transcript passages only.

Not measured:
- whether the extra "most" passages crowd out a better result on queries the eval does not
  generate;
- word forms the eval's generator does not produce (irregular verbs such as "ran" / "run").
